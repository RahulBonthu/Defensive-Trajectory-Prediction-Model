"""
RED stubs for Phase 4 evaluation requirements (EVAL-01 through EVAL-04).

All four tests import from scripts.evaluate_ablation, which does not yet exist.
Running this file will produce ImportError — the correct RED state.
Plan 04-03 will create scripts/evaluate_ablation.py to make these GREEN.

The import of TrajectoryTransformer succeeds — the model is already implemented.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# This import SUCCEEDS — model is already implemented in src/model/trajectory_model.py
from src.model.trajectory_model import TrajectoryTransformer

# These imports WILL fail until plan 04-03 creates scripts/evaluate_ablation.py
from scripts.evaluate_ablation import (
    collect_per_play_rmse,
    build_ablation_table,
    run_significance_tests,
    compute_per_position_rmse,
)


# ---------------------------------------------------------------------------
# Synthetic DataLoader helper
# ---------------------------------------------------------------------------

class _DictDataset(Dataset):
    """Wraps a list of dicts as a Dataset."""

    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def _make_eval_loader(input_dim: int = 50, n: int = 16, T: int = 25, batch_size: int = 4) -> DataLoader:
    """Build a DataLoader of n dict-based samples for evaluation."""
    items = [
        {
            "input": torch.randn(T, input_dim),
            "padding_mask": torch.ones(T, dtype=torch.bool),
            "target_xy": torch.randn(2),
            "position": "CB",
        }
        for _ in range(n)
    ]
    return DataLoader(_DictDataset(items), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# EVAL-01: collect_per_play_rmse returns list of dicts with "rmse" and "position"
# ---------------------------------------------------------------------------

def test_per_play_rmse_shape():
    """
    EVAL-01: collect_per_play_rmse must return a list of dicts, one per test sample,
    each containing keys 'rmse' (float) and 'position' (str).
    """
    model = TrajectoryTransformer(input_dim=50, dropout=0.0)
    test_loader = _make_eval_loader(input_dim=50, n=16)

    result = collect_per_play_rmse(model, test_loader)

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 16, f"Expected 16 dicts (one per sample), got {len(result)}"

    for i, item in enumerate(result):
        assert isinstance(item, dict), f"result[{i}] is {type(item)}, expected dict"
        assert "rmse" in item, f"result[{i}] missing 'rmse' key. Keys: {item.keys()}"
        assert "position" in item, f"result[{i}] missing 'position' key. Keys: {item.keys()}"
        assert isinstance(item["rmse"], float), (
            f"result[{i}]['rmse']={item['rmse']!r} is not a float"
        )
        assert isinstance(item["position"], str), (
            f"result[{i}]['position']={item['position']!r} is not a str"
        )


# ---------------------------------------------------------------------------
# EVAL-02: build_ablation_table returns DataFrame with the required 6 columns
# ---------------------------------------------------------------------------

def test_ablation_table_columns():
    """
    EVAL-02: build_ablation_table must return a pandas DataFrame with exactly the
    required columns for the ablation report.

    Required columns:
        ["Model", "Mean RMSE", "Std RMSE", "Delta (A-B)", "p (Wilcoxon)", "p (t-test)"]
    """
    import pandas as pd

    # Synthetic seed results: 3 seeds, 50 RMSE values each
    rng = np.random.default_rng(0)
    rmse_a_seeds = [rng.uniform(1.5, 2.5, size=50).astype(np.float32) for _ in range(3)]
    rmse_b_seeds = [rng.uniform(1.0, 2.0, size=50).astype(np.float32) for _ in range(3)]

    result = build_ablation_table(rmse_a_seeds, rmse_b_seeds)

    assert isinstance(result, pd.DataFrame), (
        f"Expected pd.DataFrame, got {type(result)}"
    )

    required_columns = ["Model", "Mean RMSE", "Std RMSE", "Delta (A-B)", "p (Wilcoxon)", "p (t-test)"]
    for col in required_columns:
        assert col in result.columns, (
            f"Column '{col}' missing from ablation table. "
            f"Columns present: {list(result.columns)}"
        )


# ---------------------------------------------------------------------------
# EVAL-03: run_significance_tests returns dict with p_wilcoxon and p_ttest floats
# ---------------------------------------------------------------------------

def test_significance_test_runs():
    """
    EVAL-03: run_significance_tests must return a dict with keys 'p_wilcoxon'
    and 'p_ttest', both being floats.

    Uses two aligned arrays of length 50 where A > B (Model A is slightly worse).
    """
    rng = np.random.default_rng(42)
    rmse_a = rng.uniform(1.5, 2.5, size=50).astype(np.float64)  # Model A (no ball dest)
    rmse_b = rng.uniform(1.0, 2.0, size=50).astype(np.float64)  # Model B (with ball dest)

    result = run_significance_tests(rmse_a, rmse_b)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "p_wilcoxon" in result, (
        f"'p_wilcoxon' missing from result. Keys: {list(result.keys())}"
    )
    assert "p_ttest" in result, (
        f"'p_ttest' missing from result. Keys: {list(result.keys())}"
    )
    assert isinstance(result["p_wilcoxon"], float), (
        f"p_wilcoxon={result['p_wilcoxon']!r} is not a float"
    )
    assert isinstance(result["p_ttest"], float), (
        f"p_ttest={result['p_ttest']!r} is not a float"
    )


# ---------------------------------------------------------------------------
# EVAL-04: compute_per_position_rmse returns dict with exactly the 4 position keys
# ---------------------------------------------------------------------------

def test_per_position_rmse_keys():
    """
    EVAL-04: compute_per_position_rmse must return a dict with exactly the keys
    {"CB", "FS", "SS", "LB"}, each mapping to a float mean RMSE.
    """
    rng = np.random.default_rng(7)
    # Build per-play dicts covering all 4 defensive positions
    positions = ["CB", "FS", "SS", "LB"]
    per_play = []
    for pos in positions:
        for _ in range(12):  # 12 samples per position
            per_play.append({"rmse": float(rng.uniform(0.5, 3.0)), "position": pos})

    result = compute_per_position_rmse(per_play)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    expected_keys = {"CB", "FS", "SS", "LB"}
    assert set(result.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(result.keys())}"
    )
    for pos in expected_keys:
        assert isinstance(result[pos], float), (
            f"result['{pos}']={result[pos]!r} is not a float"
        )
