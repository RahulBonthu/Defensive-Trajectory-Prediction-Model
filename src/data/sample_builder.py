"""
Sample builder: construct (player, play) tensors from preprocessed tracking data.

Phase 1 base features per frame (8 total):
    x, y, s, a_computed, dir_sin, dir_cos, o_sin, o_cos

Phase 2 will add social context features (other players, relative positions).

Ball landing coordinate source (verified against actual dataset on 2026-03-13):
    plays.csv does NOT exist in BDB 2026 format — data uses input_2023_w*.csv files.
    ball_land_x / ball_land_y are provided DIRECTLY per row in the input CSVs.
    After normalization (LOS-relative x, centered y), these values become ball_land_x_norm
    and ball_land_y_norm in the preprocessed DataFrame. ball_target_xy is read directly
    from the 'ball_land_x' and 'ball_land_y' columns (or normalized equivalents).
    See sample_builder.build_samples() for extraction logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Columns used as input features for each player frame
BASE_FEATURE_COLS = ["x", "y", "s", "a_computed", "dir_sin", "dir_cos", "o_sin", "o_cos"]

# Defensive positions to include (Phase 1 pre-filtering)
DEFENSIVE_POSITIONS = {"CB", "FS", "SS", "LB", "DE", "DT", "MLB", "ILB", "OLB", "SAF", "DB"}


def build_samples(
    df: pd.DataFrame,
    split: dict,
    sequence_length: int = 50,
) -> dict:
    """Construct fixed-length player-play tensor samples from preprocessed data.

    For each (gameId, playId, nflId) group belonging to a defensive player:
      - Sort frames by frameId
      - Take up to sequence_length frames
      - Post-pad short sequences with zeros
      - Record padding_mask (True = real frame, False = padded)
      - Set target_xy as the player's (x, y) at max(frameId)
      - For football rows in each play, record ball final (x, y) as ball_target_xy

    Args:
        df: Preprocessed tracking DataFrame with columns including BASE_FEATURE_COLS,
            gameId, playId, nflId, frameId, position.
        split: dict with keys train_game_ids, val_game_ids, test_game_ids (lists of int).
        sequence_length: Fixed output sequence length (default 50). Short sequences
            are zero-padded; longer sequences are truncated.

    Returns:
        dict with keys "train", "val", "test", each a list of sample dicts:
        {
            "gameId": int,
            "playId": int,
            "nflId": int,
            "frames": np.ndarray of shape (sequence_length, num_base_features),
            "padding_mask": np.ndarray of shape (sequence_length,) bool,
            "target_xy": np.ndarray of shape (2,),
            "ball_target_xy": np.ndarray of shape (2,) or None,
            "position": str,
        }
    """
    feature_cols = [c for c in BASE_FEATURE_COLS if c in df.columns]
    num_features = len(feature_cols)

    # Build ball_target_xy lookup: (gameId, playId) -> (ball_land_x, ball_land_y)
    # BDB 2026 format: ball_land_x / ball_land_y provided directly per row (already normalized).
    # Fallback: if ball_land_x not available, derive from football tracking rows at max(frameId).
    ball_targets: dict[tuple, np.ndarray | None] = {}
    if "ball_land_x" in df.columns and "ball_land_y" in df.columns:
        # Use the provided ball landing coordinates (one value per play — same for all rows)
        ball_df = df[["gameId", "playId", "ball_land_x", "ball_land_y"]].drop_duplicates(
            subset=["gameId", "playId"]
        )
        for _, row in ball_df.iterrows():
            key = (int(row["gameId"]), int(row["playId"]))
            ball_targets[key] = np.array([row["ball_land_x"], row["ball_land_y"]], dtype=np.float32)
    else:
        # Legacy fallback: derive from football tracking rows at max(frameId) per play
        # plays.csv does NOT contain targetX/targetY — ball_target_xy derived from
        # football tracking rows at max(frameId) per play. See sample_builder.build_samples().
        football_df = df[df["position"] == "football"]
        for (game_id, play_id), group in football_df.groupby(["gameId", "playId"]):
            last_frame = group.loc[group["frameId"].idxmax()]
            ball_targets[(int(game_id), int(play_id))] = np.array(
                [last_frame["x"], last_frame["y"]], dtype=np.float32
            )

    # Map each game to its split partition
    split_map: dict[int, str] = {}
    for partition in ("train", "val", "test"):
        for gid in split.get(f"{partition}_game_ids", []):
            split_map[int(gid)] = partition

    results: dict[str, list] = {"train": [], "val": [], "test": []}

    # Filter to defensive players (Phase 1 pre-filtering; Phase 2 enforces strictly)
    defensive_df = df[
        (df["position"].isin(DEFENSIVE_POSITIONS)) |
        (df["position"] == "football")  # keep football for reference only
    ]
    # Only build samples for actual defensive players (not football)
    player_df = df[df["position"].isin(DEFENSIVE_POSITIONS)]

    for (game_id, play_id, nfl_id), group in player_df.groupby(
        ["gameId", "playId", "nflId"], sort=False
    ):
        game_id = int(game_id)
        play_id = int(play_id)
        nfl_id  = int(nfl_id)

        partition = split_map.get(game_id)
        if partition is None:
            continue

        group_sorted = group.sort_values("frameId")
        position = str(group_sorted["position"].iloc[0])

        # Extract feature matrix
        raw = group_sorted[feature_cols].values.astype(np.float32)
        n_real = min(len(raw), sequence_length)

        frames = np.zeros((sequence_length, num_features), dtype=np.float32)
        frames[:n_real] = raw[:n_real]

        padding_mask = np.zeros(sequence_length, dtype=bool)
        padding_mask[:n_real] = True

        # Target: player's (x, y) at last real frame
        last_row = group_sorted.iloc[-1]
        target_xy = np.array([last_row["x"], last_row["y"]], dtype=np.float32)

        ball_target_xy = ball_targets.get((game_id, play_id))

        results[partition].append({
            "gameId": game_id,
            "playId": play_id,
            "nflId": nfl_id,
            "frames": frames,
            "padding_mask": padding_mask,
            "target_xy": target_xy,
            "ball_target_xy": ball_target_xy,
            "position": position,
        })

    return results
