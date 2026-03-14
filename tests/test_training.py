"""
RED stubs for Phase 4 training requirements (TRAIN-01 through TRAIN-04).

All four tests import from scripts.train_model, which does not yet exist.
Running this file will produce ImportError — the correct RED state.
Plans 04-02 will create scripts/train_model.py to make these GREEN.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, Dataset

# This import WILL fail until plan 04-02 creates scripts/train_model.py
from scripts.train_model import train_one_model


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


def _make_synthetic_loader(input_dim: int, n: int = 16, T: int = 25, batch_size: int = 4) -> DataLoader:
    """Build a DataLoader of n dict-based samples for a given input_dim."""
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
# TRAIN-01: Identical hyperparameters for Model A and Model B
# ---------------------------------------------------------------------------

def test_identical_hyperparameters():
    """
    TRAIN-01: Both model variants must use the same transformer hyperparameters.

    train_one_model must return a dict with key "config" containing:
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1
    """
    train_loader_a = _make_synthetic_loader(input_dim=50)
    val_loader_a = _make_synthetic_loader(input_dim=50, n=8)
    train_loader_b = _make_synthetic_loader(input_dim=52)
    val_loader_b = _make_synthetic_loader(input_dim=52, n=8)

    result_a = train_one_model(
        model_variant="A",
        train_loader=train_loader_a,
        val_loader=val_loader_a,
        num_epochs=1,
        seed=42,
    )
    result_b = train_one_model(
        model_variant="B",
        train_loader=train_loader_b,
        val_loader=val_loader_b,
        num_epochs=1,
        seed=42,
    )

    for result, variant in [(result_a, "A"), (result_b, "B")]:
        assert "config" in result, f"Model {variant} result missing 'config' key"
        cfg = result["config"]
        assert cfg["d_model"] == 128, f"Model {variant}: d_model={cfg['d_model']}, expected 128"
        assert cfg["nhead"] == 4, f"Model {variant}: nhead={cfg['nhead']}, expected 4"
        assert cfg["num_layers"] == 2, f"Model {variant}: num_layers={cfg['num_layers']}, expected 2"
        assert cfg["dim_feedforward"] == 256, f"Model {variant}: dim_feedforward={cfg['dim_feedforward']}, expected 256"
        assert abs(cfg["dropout"] - 0.1) < 1e-6, f"Model {variant}: dropout={cfg['dropout']}, expected 0.1"


# ---------------------------------------------------------------------------
# TRAIN-02: RMSE loss produces positive training loss values
# ---------------------------------------------------------------------------

def test_rmse_loss_used():
    """
    TRAIN-02: Training must use RMSE loss — all recorded loss values are positive floats.

    RMSE is always >= 0; with random data it will be > 0.
    """
    train_loader = _make_synthetic_loader(input_dim=50)
    val_loader = _make_synthetic_loader(input_dim=50, n=8)

    result = train_one_model(
        model_variant="A",
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        seed=42,
    )

    assert "train_losses" in result, "Result missing 'train_losses' key"
    losses = result["train_losses"]
    assert len(losses) > 0, "train_losses list is empty"
    for i, loss in enumerate(losses):
        assert isinstance(loss, float), f"train_losses[{i}]={loss!r} is not a float"
        assert loss > 0, f"train_losses[{i}]={loss} is not positive (expected RMSE > 0)"


# ---------------------------------------------------------------------------
# TRAIN-03: wandb.log called at least once per epoch with expected keys
# ---------------------------------------------------------------------------

def test_wandb_logging():
    """
    TRAIN-03: train_one_model must call wandb.log at least twice (once per epoch)
    with keys including 'train_loss' and 'val_loss'.
    """
    train_loader = _make_synthetic_loader(input_dim=50)
    val_loader = _make_synthetic_loader(input_dim=50, n=8)

    with patch("wandb.init") as mock_init, patch("wandb.log") as mock_log:
        mock_run = MagicMock()
        mock_init.return_value = mock_run

        train_one_model(
            model_variant="A",
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            seed=42,
        )

    assert mock_log.call_count >= 2, (
        f"wandb.log called {mock_log.call_count} times; expected >= 2 (once per epoch)"
    )

    # Collect all keys logged across all calls
    all_logged_keys = set()
    for call_args in mock_log.call_args_list:
        if call_args.args:
            all_logged_keys.update(call_args.args[0].keys())
        if call_args.kwargs:
            all_logged_keys.update(call_args.kwargs.keys())

    assert "train_loss" in all_logged_keys, (
        f"'train_loss' not found in wandb.log calls. Keys seen: {all_logged_keys}"
    )
    assert "val_loss" in all_logged_keys, (
        f"'val_loss' not found in wandb.log calls. Keys seen: {all_logged_keys}"
    )


# ---------------------------------------------------------------------------
# TRAIN-04: A .pt checkpoint file is saved in checkpoint_dir
# ---------------------------------------------------------------------------

def test_checkpoints_saved(tmp_path: Path):
    """
    TRAIN-04: train_one_model must save at least one .pt checkpoint in checkpoint_dir.
    """
    train_loader = _make_synthetic_loader(input_dim=50)
    val_loader = _make_synthetic_loader(input_dim=50, n=8)

    result = train_one_model(
        model_variant="A",
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        checkpoint_dir=tmp_path,
        seed=42,
    )

    pt_files = list(tmp_path.glob("*.pt"))
    assert len(pt_files) >= 1, (
        f"No .pt file found in {tmp_path}. "
        f"checkpoint_path in result: {result.get('checkpoint_path')}"
    )
