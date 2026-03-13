"""Tests for DefensiveTrajectoryDataset — FEAT-01 through FEAT-06.

Each test starts RED (ImportError from missing dataset.py) and turns GREEN
after Plan 02 implements DefensiveTrajectoryDataset.
"""
import numpy as np
import torch
import pytest
from src.data.dataset import DefensiveTrajectoryDataset

# Strict positions that DefensiveTrajectoryDataset must accept.
STRICT_POSITIONS = {"CB", "FS", "SS", "LB"}


def test_position_filter(minimal_samples, minimal_context_df):
    """FEAT-01 — Dataset only accepts players with strict defensive positions.

    The minimal_samples fixture contains 4 CB, 1 FS, 1 SS — all strictly valid.
    A poisoned sample with position="ILB" must be excluded so that its nflId
    does not appear in any returned item.
    """
    import pandas as pd

    dataset = DefensiveTrajectoryDataset(
        samples=minimal_samples,
        context_df=minimal_context_df,
        sequence_length=25,
        include_ball_destination=False,
    )
    assert len(dataset) == 6, (
        f"Expected 6 items (all CB/FS/SS), got {len(dataset)}"
    )

    # Build a poisoned sample list: original 6 + 1 with position="ILB"
    poisoned_nfl_id = 999
    poisoned_sample = {
        "gameId": 2023010100,
        "playId": 1,
        "nflId": poisoned_nfl_id,
        "frames": np.zeros((25, 8), dtype=np.float32),
        "padding_mask": np.zeros(25, dtype=bool),
        "target_xy": np.array([2.5, -1.0], dtype=np.float32),
        "ball_target_xy": np.array([15.0, 3.0], dtype=np.float32),
        "position": "ILB",
    }
    poisoned_list = minimal_samples + [poisoned_sample]

    dataset_poisoned = DefensiveTrajectoryDataset(
        samples=poisoned_list,
        context_df=minimal_context_df,
        sequence_length=25,
        include_ball_destination=False,
    )
    positions_seen = [dataset_poisoned[i]["position"] for i in range(len(dataset_poisoned))]
    assert "ILB" not in positions_seen, (
        "Dataset must filter out position='ILB' (not in STRICT_POSITIONS)"
    )


def test_player_play_independence(minimal_samples, minimal_context_df):
    """FEAT-02 — Each (gameId, playId, nflId) triple is an independent item.

    No merging or aggregation across player-play pairs — dataset length equals
    the number of input samples (after position filtering).
    """
    dataset = DefensiveTrajectoryDataset(
        samples=minimal_samples,
        context_df=minimal_context_df,
        sequence_length=25,
        include_ball_destination=False,
    )
    assert len(dataset) == len(minimal_samples), (
        f"Dataset must have one item per input sample; "
        f"expected {len(minimal_samples)}, got {len(dataset)}"
    )


def test_sequence_padding_and_mask(minimal_samples, minimal_context_df):
    """FEAT-03 — T dimension is exactly 25; padding mask and padded frames match.

    For each item:
    - input.shape[0] == 25
    - padding_mask.dtype == torch.bool
    - padding_mask.sum() == 10 (10 real frames)
    - input[10:, 0:8] is all zeros (own-kinematic columns zeroed at padded steps)
    """
    dataset = DefensiveTrajectoryDataset(
        samples=minimal_samples,
        context_df=minimal_context_df,
        sequence_length=25,
        include_ball_destination=False,
    )
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert item["input"].shape[0] == 25, (
            f"Item {idx}: expected T=25, got {item['input'].shape[0]}"
        )
        assert item["padding_mask"].dtype == torch.bool, (
            f"Item {idx}: padding_mask must be torch.bool, got {item['padding_mask'].dtype}"
        )
        assert item["padding_mask"].sum().item() == 10, (
            f"Item {idx}: expected 10 real frames, got {item['padding_mask'].sum().item()}"
        )
        # Own-kinematic columns (0:8) must be zero at padded timesteps (10:25)
        padded_own_kin = item["input"][10:, 0:8]
        assert padded_own_kin.abs().sum().item() == 0.0, (
            f"Item {idx}: padded own-kinematic features must be zero"
        )


def test_social_context_shape(minimal_samples, minimal_context_df):
    """FEAT-04 — Model A input has feature dimension 50.

    50 features = 8 own-kinematic + 42 social context (21 other players × 2 coords).
    """
    dataset = DefensiveTrajectoryDataset(
        samples=minimal_samples,
        context_df=minimal_context_df,
        sequence_length=25,
        include_ball_destination=False,
    )
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert item["input"].shape == (25, 50), (
            f"Item {idx}: Model A input shape must be (25, 50), got {item['input'].shape}"
        )
        assert item["input"].shape[-1] == 50, (
            f"Item {idx}: Feature dimension must be 50 for Model A"
        )


def test_ball_destination_model_b(minimal_samples, minimal_context_df):
    """FEAT-05 — Model B input has feature dimension 52.

    52 features = 50 (Model A) + 2 ball landing coords broadcast across all T frames.
    The last 2 columns must equal ball_target_xy = [15.0, 3.0].
    """
    dataset = DefensiveTrajectoryDataset(
        samples=minimal_samples,
        context_df=minimal_context_df,
        sequence_length=25,
        include_ball_destination=True,
    )
    expected_ball = torch.tensor([15.0, 3.0])
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert item["input"].shape[-1] == 52, (
            f"Item {idx}: Model B input feature dim must be 52, got {item['input'].shape[-1]}"
        )
        ball_cols = item["input"][:, 50:]  # shape (25, 2)
        assert ball_cols.shape == (25, 2), (
            f"Item {idx}: ball destination columns must have shape (T, 2)"
        )
        # Ball destination is a play-level constant — every frame must match
        for t in range(25):
            assert torch.allclose(ball_cols[t], expected_ball), (
                f"Item {idx}, frame {t}: ball destination {ball_cols[t].tolist()} "
                f"!= expected {expected_ball.tolist()}"
            )


def test_no_ball_leakage_model_a(minimal_samples, minimal_context_df):
    """FEAT-06 — Model A input has exactly 50 features; no ball destination columns.

    Hard leakage prevention: columns 50+ must not exist in Model A output.
    """
    dataset = DefensiveTrajectoryDataset(
        samples=minimal_samples,
        context_df=minimal_context_df,
        sequence_length=25,
        include_ball_destination=False,
    )
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert item["input"].shape[-1] == 50, (
            f"Item {idx}: Model A must have exactly 50 features, "
            f"got {item['input'].shape[-1]}"
        )
        # Confirm no columns beyond index 49 exist
        assert item["input"][:, 50:].numel() == 0, (
            f"Item {idx}: Model A must have no features at index 50+ (ball leakage check)"
        )
