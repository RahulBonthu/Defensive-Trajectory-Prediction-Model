"""
train_model.py — Phase 04 Plan 02.

Exports:
    train_one_model   Standard PyTorch training loop for TrajectoryTransformer
                      with wandb logging, best-val checkpointing, and
                      ReduceLROnPlateau scheduling.

Usage example:
    from scripts.train_model import train_one_model
    result = train_one_model("A", train_loader, val_loader, num_epochs=50, seed=42)

Notes:
    - PYTORCH_ENABLE_MPS_FALLBACK must be set before any torch import.
    - wandb.log is called directly (module-level call) when wandb_run is None
      so that unittest.mock.patch("wandb.log") works in tests.
    - checkpoint_path follows the naming convention:
          checkpoint_dir / f"model_{variant.lower()}_seed{seed}_best.pt"
      Caller is responsible for copying to canonical model_a_best.pt / model_b_best.pt.
"""

import os

# Must appear before any torch import — required for MPS op fallback on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import wandb
from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.trajectory_model import TrajectoryTransformer, rmse_loss, get_device


def train_one_model(
    model_variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    checkpoint_dir: Union[str, Path] = "models",
    seed: int = 42,
    wandb_run=None,
) -> dict:
    """Train a TrajectoryTransformer for one (variant, seed) combination.

    Args:
        model_variant:   "A" (input_dim=50) or "B" (input_dim=52).
        train_loader:    DataLoader yielding dicts with keys
                         {"input", "padding_mask", "target_xy", "position"}.
        val_loader:      Same schema as train_loader.
        num_epochs:      Number of full passes over training data.
        lr:              Initial Adam learning rate.
        checkpoint_dir:  Directory in which to save the best-val checkpoint.
        seed:            Random seed for reproducibility.
        wandb_run:       Pre-initialised wandb.Run (for multi-run orchestration).
                         When None, wandb.log() is called directly at module level.

    Returns:
        dict with keys:
            "config":          dict of all hyperparameters used.
            "train_losses":    list[float] — one RMSE per epoch.
            "val_losses":      list[float] — one RMSE per epoch.
            "best_val_loss":   float — lowest validation RMSE achieved.
            "checkpoint_path": str  — absolute path of the saved .pt file.
    """
    # ------------------------------------------------------------------
    # Validate variant and resolve input_dim
    # ------------------------------------------------------------------
    variant_to_input_dim = {"A": 50, "B": 52}
    if model_variant not in variant_to_input_dim:
        raise ValueError(
            f"model_variant must be 'A' or 'B', got {model_variant!r}"
        )
    input_dim = variant_to_input_dim[model_variant]

    # ------------------------------------------------------------------
    # Build config dict (used in return value and wandb config logging)
    # ------------------------------------------------------------------
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1

    config = {
        "model_variant": model_variant,
        "input_dim": input_dim,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "lr": lr,
        "num_epochs": num_epochs,
        "seed": seed,
    }

    # ------------------------------------------------------------------
    # Reproducibility seeds
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Device and model
    # ------------------------------------------------------------------
    device = get_device()

    model = TrajectoryTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # ------------------------------------------------------------------
    # Checkpoint path
    # ------------------------------------------------------------------
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"model_{model_variant.lower()}_seed{seed}_best.pt"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in tqdm(range(num_epochs), desc=f"Training Model {model_variant} seed={seed}"):
        # ---- Train phase ----
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            x = batch["input"].to(device)               # (B, T, input_dim)
            padding_mask = batch["padding_mask"].to(device)  # (B, T)
            target = batch["target_xy"].to(device)      # (B, 2)

            optimizer.zero_grad()
            pred = model(x, padding_mask)               # (B, 2)
            loss = rmse_loss(pred, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        epoch_train_loss /= max(num_train_batches, 1)

        # ---- Validation phase ----
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                target = batch["target_xy"].to(device)

                pred = model(x, padding_mask)
                loss = rmse_loss(pred, target)

                epoch_val_loss += loss.item()
                num_val_batches += 1

        epoch_val_loss /= max(num_val_batches, 1)

        # ---- Record ----
        train_losses.append(float(epoch_train_loss))
        val_losses.append(float(epoch_val_loss))

        # ---- wandb logging ----
        log_payload = {
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "epoch": epoch,
        }
        if wandb_run is not None:
            wandb_run.log(log_payload)
        else:
            try:
                wandb.log(log_payload)
            except Exception:
                # wandb not initialised (e.g. offline unit tests without patch).
                # test_wandb_logging patches wandb.log so this branch is never hit
                # during that test; other tests don't need wandb to be active.
                pass

        # ---- LR scheduler ----
        scheduler.step(epoch_val_loss)

        # ---- Best-val checkpointing ----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), checkpoint_path)

    return {
        "config": config,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
