"""DefensiveTrajectoryDataset — Phase 2 Feature Engineering.

Wraps build_samples() output into a PyTorch Dataset. Enforces strict position
filtering, assembles social context from a pre-built index, and exposes the
Model A / Model B ablation boundary via include_ball_destination.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Module-level constants (hard-coded, not configurable)
# ---------------------------------------------------------------------------
STRICT_DEFENSIVE_POSITIONS: set[str] = {"CB", "FS", "SS", "LB"}
CONTEXT_PLAYERS: int = 21                # social-context slots (zero-padded)
CONTEXT_FEATURES_PER_PLAYER: int = 2    # (x, y) only
SEQUENCE_LENGTH: int = 25               # default T


class DefensiveTrajectoryDataset(Dataset):
    """PyTorch Dataset for defensive player trajectory prediction.

    Each item is a single (gameId, playId, nflId) triple. Items with positions
    outside STRICT_DEFENSIVE_POSITIONS are silently excluded at construction
    time.

    Args:
        samples: List of sample dicts produced by build_samples(). Each dict
            must have keys: gameId, playId, nflId, frames, padding_mask,
            target_xy, ball_target_xy, position.
        context_df: Preprocessed tracking DataFrame containing columns
            gameId, playId, frameId, nflId, x, y, ball_land_x, ball_land_y.
            Held only during __init__; set to None afterwards to prevent OOM
            issues with DataLoader workers.
        sequence_length: Number of timesteps T to return (default 25).
        include_ball_destination: If True, append 2 ball-landing columns to
            the feature dimension → shape (T, 52) instead of (T, 50).
    """

    def __init__(
        self,
        samples: list[dict],
        context_df: pd.DataFrame,
        sequence_length: int = SEQUENCE_LENGTH,
        include_ball_destination: bool = False,
    ) -> None:
        # Filter to strict defensive positions only
        self.samples: list[dict] = [
            s for s in samples if s["position"] in STRICT_DEFENSIVE_POSITIONS
        ]
        self.seq_len = sequence_length
        self.include_ball_destination = include_ball_destination

        # Build pre-computed indices from the DataFrame
        self._context_index, self._ball_index = self._build_context_index(context_df)

        # Drop reference to DataFrame to free memory when workers are forked
        self.context_df = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_context_index(
        self, context_df: pd.DataFrame
    ) -> tuple[dict, dict]:
        """Build nested position and ball-destination lookup dicts.

        Returns:
            context_index: {(gameId, playId): {frameId: {nflId: np.array([x, y])}}}
            ball_index:    {(gameId, playId): np.array([ball_land_x, ball_land_y])}
        """
        context_index: dict = {}
        ball_index: dict = {}

        required_cols = {"gameId", "playId", "frameId", "nflId", "x", "y"}
        available = set(context_df.columns)
        missing = required_cols - available
        if missing:
            raise ValueError(f"context_df missing columns: {missing}")

        # Build context_index
        for row in context_df[["gameId", "playId", "frameId", "nflId", "x", "y"]].itertuples(
            index=False
        ):
            key = (int(row.gameId), int(row.playId))
            fid = int(row.frameId)
            nid = int(row.nflId)
            xy = np.array([row.x, row.y], dtype=np.float32)

            if key not in context_index:
                context_index[key] = {}
            frame_dict = context_index[key]
            if fid not in frame_dict:
                frame_dict[fid] = {}
            frame_dict[fid][nid] = xy

        # Build ball_index from ball_land_x / ball_land_y columns if present
        if "ball_land_x" in available and "ball_land_y" in available:
            ball_df = (
                context_df[["gameId", "playId", "ball_land_x", "ball_land_y"]]
                .drop_duplicates(subset=["gameId", "playId"])
            )
            for row in ball_df.itertuples(index=False):
                key = (int(row.gameId), int(row.playId))
                ball_index[key] = np.array(
                    [row.ball_land_x, row.ball_land_y], dtype=np.float32
                )

        return context_index, ball_index

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        game_id = int(sample["gameId"])
        play_id = int(sample["playId"])
        target_nfl_id = int(sample["nflId"])

        # Own-kinematic block: shape (T, 8)
        own_kin: np.ndarray = sample["frames"][: self.seq_len]  # (T, 8)

        # Padding mask: shape (T,) bool
        mask: np.ndarray = sample["padding_mask"][: self.seq_len]  # (T,)

        # Determine frame IDs covered by this slice
        # Frames are ordered by frameId in the builder; we infer them from
        # the context index for this play. If the play has no index entry,
        # we use a fallback of range(1, T+1) — all social context will be zero.
        play_frames = self._context_index.get((game_id, play_id), {})
        sorted_frame_ids = sorted(play_frames.keys()) if play_frames else []
        frame_ids = sorted_frame_ids[: self.seq_len]

        # Pad frame_ids list to seq_len with sentinel -1 (will produce zero rows)
        while len(frame_ids) < self.seq_len:
            frame_ids.append(-1)

        # Social context: shape (T, 42)
        ctx = self._assemble_social_context(game_id, play_id, frame_ids, target_nfl_id)

        # Zero-out social context at padded timesteps
        ctx[~mask] = 0.0

        # Concat own-kin + context → (T, 50)
        tensor = np.concatenate([own_kin, ctx], axis=-1)

        # Optional Model B: append ball destination columns → (T, 52)
        if self.include_ball_destination:
            ball_xy = self._get_ball_destination(sample)  # (2,)
            ball_tile = np.tile(ball_xy, (self.seq_len, 1))  # (T, 2)
            tensor = np.concatenate([tensor, ball_tile], axis=-1)

        return {
            "input": torch.tensor(tensor, dtype=torch.float32),
            "padding_mask": torch.tensor(mask, dtype=torch.bool),
            "target_xy": torch.tensor(sample["target_xy"], dtype=torch.float32),
            "position": sample["position"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assemble_social_context(
        self,
        game_id: int,
        play_id: int,
        frame_ids: list[int],
        target_nfl_id: int,
    ) -> np.ndarray:
        """Assemble social context matrix of shape (T, CONTEXT_PLAYERS * 2).

        For each frame, retrieves all other players' (x, y) from the pre-built
        index, sorts them by nflId ascending (deterministic), excludes the
        target player, then zero-pads or truncates to exactly CONTEXT_PLAYERS
        slots.

        Args:
            game_id: Game identifier.
            play_id: Play identifier.
            frame_ids: List of frameIds of length T; sentinel -1 → zero row.
            target_nfl_id: nflId of the player being predicted (excluded).

        Returns:
            np.ndarray of shape (T, CONTEXT_PLAYERS * CONTEXT_FEATURES_PER_PLAYER).
        """
        T = len(frame_ids)
        n_context_features = CONTEXT_PLAYERS * CONTEXT_FEATURES_PER_PLAYER  # 42
        output = np.zeros((T, n_context_features), dtype=np.float32)

        play_dict = self._context_index.get((game_id, play_id), {})

        for t, fid in enumerate(frame_ids):
            if fid == -1:
                continue  # padded timestep — leave zeros
            frame_dict = play_dict.get(fid, {})
            if not frame_dict:
                continue  # frame not in index — leave zeros

            # Collect other players sorted by nflId
            others = [
                (nid, xy)
                for nid, xy in sorted(frame_dict.items())
                if nid != target_nfl_id
            ]

            # Fill up to CONTEXT_PLAYERS slots
            for slot, (_, xy) in enumerate(others[:CONTEXT_PLAYERS]):
                start = slot * CONTEXT_FEATURES_PER_PLAYER
                output[t, start : start + CONTEXT_FEATURES_PER_PLAYER] = xy

        return output

    def _get_ball_destination(self, sample: dict) -> np.ndarray:
        """Return the ball landing (x, y) for this sample's play.

        Lookup order:
          1. self._ball_index[(gameId, playId)]
          2. sample["ball_target_xy"]

        Raises:
            ValueError: If neither source is available.
        """
        game_id = int(sample["gameId"])
        play_id = int(sample["playId"])

        ball_xy = self._ball_index.get((game_id, play_id))
        if ball_xy is not None:
            return ball_xy

        fallback = sample.get("ball_target_xy")
        if fallback is not None:
            return np.asarray(fallback, dtype=np.float32)

        raise ValueError(
            f"No ball destination for game {game_id} play {play_id}"
        )
