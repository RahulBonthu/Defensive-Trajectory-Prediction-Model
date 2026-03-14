"""Phase 2 smoke test — verifies DefensiveTrajectoryDataset shapes against real data.

Run from project root:
    python scripts/smoke_test_dataset.py

Exits 0 if all assertions pass, 1 if any fail.
"""
from __future__ import annotations

import json
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    STRICT_DEFENSIVE_POSITIONS,
    DefensiveTrajectoryDataset,
)
from src.data.sample_builder import build_samples


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data artifacts
    # ------------------------------------------------------------------
    print("Loading data/processed/cleaned.parquet ...")
    df = pd.read_parquet("data/processed/cleaned.parquet")

    print("Loading data/processed/splits.json ...")
    with open("data/processed/splits.json") as fh:
        split = json.load(fh)

    # ------------------------------------------------------------------
    # 2. Build samples (sequence_length=50 matches pipeline default)
    # ------------------------------------------------------------------
    print("Building samples from cleaned.parquet (this may take ~30-60s) ...")
    samples = build_samples(df, split, sequence_length=50)

    # ------------------------------------------------------------------
    # 3. Filter each split to strict defensive positions
    # ------------------------------------------------------------------
    train_samples = [s for s in samples["train"] if s["position"] in STRICT_DEFENSIVE_POSITIONS]
    val_samples   = [s for s in samples["val"]   if s["position"] in STRICT_DEFENSIVE_POSITIONS]

    # ------------------------------------------------------------------
    # 4. Build datasets
    # ------------------------------------------------------------------
    print("Building context index for train_a (this takes ~30-60s)...")
    train_a = DefensiveTrajectoryDataset(
        train_samples, df, sequence_length=25, include_ball_destination=False
    )

    print("Building context index for train_b (this takes ~30-60s)...")
    train_b = DefensiveTrajectoryDataset(
        train_samples, df, sequence_length=25, include_ball_destination=True
    )

    print("Building context index for val_a (this takes ~30-60s)...")
    val_a = DefensiveTrajectoryDataset(
        val_samples, df, sequence_length=25, include_ball_destination=False
    )

    # ------------------------------------------------------------------
    # 5. Build DataLoaders and fetch one batch each
    # ------------------------------------------------------------------
    loader_a = DataLoader(train_a, batch_size=64, shuffle=False, num_workers=0)
    loader_b = DataLoader(train_b, batch_size=64, shuffle=False, num_workers=0)

    print("Fetching one batch from loader_a ...")
    batch_a = next(iter(loader_a))

    print("Fetching one batch from loader_b ...")
    batch_b = next(iter(loader_b))

    # ------------------------------------------------------------------
    # 6. Evaluate assertions
    # ------------------------------------------------------------------
    checks = {
        "Model A input shape == (64, 25, 50)": batch_a["input"].shape == torch.Size([64, 25, 50]),
        "Model B input shape == (64, 25, 52)": batch_b["input"].shape == torch.Size([64, 25, 52]),
        "Model A no ball cols": batch_a["input"].shape[-1] == 50,
        "Model B last 2 cols non-zero": batch_b["input"][:, :, 50:].abs().sum().item() > 0,
        "train_a size == 52779": len(train_a) == 52779,
    }

    # ------------------------------------------------------------------
    # 7. Print formatted report
    # ------------------------------------------------------------------
    print()
    print("=== Phase 2 Smoke Test ===")
    print()
    print("Dataset sizes:")
    print(f"  train_a: {len(train_a)} samples  (expected: 52779)")
    print(f"  train_b: {len(train_b)} samples  (expected: 52779)")
    print(f"  val_a:   {len(val_a)} samples    (expected: 7497)")
    print()
    print("Batch shapes:")
    print(
        f"  Model A: input={tuple(batch_a['input'].shape)}"
        f"  padding_mask={tuple(batch_a['padding_mask'].shape)}"
        f"  target_xy={tuple(batch_a['target_xy'].shape)}"
    )
    print(
        f"  Model B: input={tuple(batch_b['input'].shape)}"
        f"  padding_mask={tuple(batch_b['padding_mask'].shape)}"
        f"  target_xy={tuple(batch_b['target_xy'].shape)}"
    )
    print()
    print("Assertions:")
    failures = []
    for label, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: {passed}")
        if not passed:
            failures.append(label)

    print()
    if failures:
        print("=== FAILURES DETECTED ===")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("=== ALL ASSERTIONS PASS ===")


if __name__ == "__main__":
    main()
