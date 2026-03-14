"""
run_training.py — Phase 04 Plan 04.

Orchestration script: trains TrajectoryTransformer variants A and B across
3 seeds (42, 123, 456), logs to wandb, copies best checkpoints to canonical
paths, and produces the ablation table + per-play RMSE CSVs.

Usage:
    python scripts/run_training.py

Artifacts produced:
    models/model_a_best.pt
    models/model_b_best.pt
    results/ablation_table.csv
    results/per_play_rmse_a_seed<N>.csv  (one per seed)
    results/per_play_rmse_b_seed<N>.csv  (one per seed)
    results/per_play_rmse_a.csv          (all seeds concatenated)
    results/per_play_rmse_b.csv          (all seeds concatenated)
"""

import os

# Must appear before any torch import — required for MPS op fallback on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

from scripts.train_model import train_one_model
from scripts.evaluate_ablation import (
    collect_per_play_rmse,
    build_ablation_table,
    compute_per_position_rmse,
    save_per_play_csv,
)
from src.data.dataset import DefensiveTrajectoryDataset
from src.data.sample_builder import build_samples
from src.model.trajectory_model import TrajectoryTransformer, get_device

SEEDS = [42, 123, 456]


def _build_loaders(samples, df, include_ball_destination: bool, device):
    """Build train / val / test DataLoaders for one model variant."""
    num_workers = 0 if str(device) == "mps" else 2

    train_ds = DefensiveTrajectoryDataset(
        samples["train"], df, include_ball_destination=include_ball_destination
    )
    val_ds = DefensiveTrajectoryDataset(
        samples["val"], df, include_ball_destination=include_ball_destination
    )
    test_ds = DefensiveTrajectoryDataset(
        samples["test"], df, include_ball_destination=include_ball_destination
    )

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=num_workers, drop_last=False
    )

    return train_loader, val_loader, test_loader


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    splits = json.loads(pathlib.Path("data/processed/splits.json").read_text())
    df = pd.read_parquet("data/processed/cleaned.parquet")
    samples = build_samples(df, splits)

    print(
        f"Samples — train: {len(samples['train'])}, "
        f"val: {len(samples['val'])}, "
        f"test: {len(samples['test'])}"
    )

    # ------------------------------------------------------------------
    # 2. Determine device
    # ------------------------------------------------------------------
    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 3. Training loop — variants A and B, 3 seeds each
    # ------------------------------------------------------------------
    # variant -> list of (result_dict, per_play_records, np_rmse_array)
    variant_results: dict[str, list] = {"A": [], "B": []}
    # variant -> list of np.ndarray — one per seed — for ablation table
    rmse_a_seeds: list[np.ndarray] = []
    rmse_b_seeds: list[np.ndarray] = []

    # Pre-build dataloaders outside the seed loop so we re-use datasets.
    # Note: DefensiveTrajectoryDataset sets self.context_df = None after __init__,
    # so building all datasets first then releasing df is safe.
    include_ball_dest = {"A": False, "B": True}

    print("Building datasets for both variants...")
    loaders: dict[str, tuple] = {}
    for variant, include_bd in include_ball_dest.items():
        loaders[variant] = _build_loaders(samples, df, include_bd, device)

    # df is no longer needed — release memory
    del df

    for variant in ["A", "B"]:
        train_loader, val_loader, test_loader = loaders[variant]

        seed_results_for_variant = []

        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Training Model {variant}, seed={seed}")
            print(f"{'='*60}")

            config_dict = {
                "model_variant": variant,
                "input_dim": 50 if variant == "A" else 52,
                "seed": seed,
                "num_epochs": 50,
                "lr": 1e-3,
                "batch_size": 256,
            }

            run = wandb.init(
                project="defensive-trajectory-prediction",
                name=f"model_{variant}_seed{seed}",
                config=config_dict,
                reinit=True,
            )

            result = train_one_model(
                model_variant=variant,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=50,
                lr=1e-3,
                checkpoint_dir="models",
                seed=seed,
                wandb_run=run,
            )

            run.finish()

            print(
                f"Model {variant} seed={seed}: "
                f"best_val_loss={result['best_val_loss']:.4f}, "
                f"checkpoint={result['checkpoint_path']}"
            )

            # ---- Evaluate on test set using this seed's checkpoint ----
            input_dim = 50 if variant == "A" else 52
            model = TrajectoryTransformer(input_dim=input_dim).to(device)
            model.load_state_dict(
                torch.load(result["checkpoint_path"], map_location=device)
            )
            model.eval()

            per_play = collect_per_play_rmse(model, test_loader, device=device)

            # Tag each record with seed so the merged CSV is traceable
            for rec in per_play:
                rec["seed"] = seed

            # Save per-seed CSV
            save_per_play_csv(per_play, variant, seed, output_dir="results")

            # Collect numpy RMSE array for ablation table
            rmse_array = np.array([rec["rmse"] for rec in per_play])

            seed_results_for_variant.append(
                {
                    "result": result,
                    "per_play": per_play,
                    "rmse_array": rmse_array,
                }
            )

        variant_results[variant] = seed_results_for_variant

        # Build list of rmse arrays for ablation table
        if variant == "A":
            rmse_a_seeds = [s["rmse_array"] for s in seed_results_for_variant]
        else:
            rmse_b_seeds = [s["rmse_array"] for s in seed_results_for_variant]

    # ------------------------------------------------------------------
    # 4. Determine and copy canonical best checkpoints
    # ------------------------------------------------------------------
    print("\nCopying canonical best checkpoints...")
    for variant in ["A", "B"]:
        best = min(variant_results[variant], key=lambda s: s["result"]["best_val_loss"])
        src = best["result"]["checkpoint_path"]
        dst = f"models/model_{variant.lower()}_best.pt"
        shutil.copy(src, dst)
        print(
            f"  model_{variant.lower()}_best.pt <- {src} "
            f"(val_loss={best['result']['best_val_loss']:.4f})"
        )

    # ------------------------------------------------------------------
    # 5. Build ablation table
    # ------------------------------------------------------------------
    print("\nBuilding ablation table...")
    ablation_df = build_ablation_table(rmse_a_seeds, rmse_b_seeds)
    print(ablation_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Write merged per-play CSVs (all seeds concatenated) for A and B
    # ------------------------------------------------------------------
    pathlib.Path("results").mkdir(exist_ok=True)
    for variant in ["A", "B"]:
        all_records = []
        for s in variant_results[variant]:
            all_records.extend(s["per_play"])
        merged_path = f"results/per_play_rmse_{variant.lower()}.csv"
        pd.DataFrame(all_records).to_csv(merged_path, index=False)
        print(f"Wrote {merged_path} ({len(all_records)} rows)")

    # ------------------------------------------------------------------
    # 7. Per-position RMSE (canonical best model for each variant)
    # ------------------------------------------------------------------
    print("\nPer-position RMSE:")
    for variant in ["A", "B"]:
        train_loader, val_loader, test_loader = loaders[variant]
        input_dim = 50 if variant == "A" else 52
        model = TrajectoryTransformer(input_dim=input_dim).to(device)
        model.load_state_dict(
            torch.load(f"models/model_{variant.lower()}_best.pt", map_location=device)
        )
        model.eval()

        per_play = collect_per_play_rmse(model, test_loader, device=device)
        pos_rmse = compute_per_position_rmse(per_play)

        print(f"  Model {variant}:")
        for pos, rmse_val in sorted(pos_rmse.items()):
            print(f"    {pos}: {rmse_val:.4f} yards")

        # Also save per-position summary CSV
        pos_path = f"results/per_position_rmse_{variant.lower()}.csv"
        pd.DataFrame([{"position": k, "mean_rmse": v} for k, v in pos_rmse.items()]).to_csv(
            pos_path, index=False
        )
        print(f"  Wrote {pos_path}")

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TRAINING COMPLETE — FINAL SUMMARY")
    print("="*60)
    for variant in ["A", "B"]:
        best = min(variant_results[variant], key=lambda s: s["result"]["best_val_loss"])
        all_vals = [s["result"]["best_val_loss"] for s in variant_results[variant]]
        print(
            f"  Model {variant}: best_val_loss={best['result']['best_val_loss']:.4f} "
            f"(seeds={all_vals})"
        )
    print("\nAblation table:")
    print(ablation_df.to_string(index=False))
    print("\nArtifacts:")
    print("  models/model_a_best.pt")
    print("  models/model_b_best.pt")
    print("  results/ablation_table.csv")
    print("  results/per_play_rmse_a.csv")
    print("  results/per_play_rmse_b.csv")


if __name__ == "__main__":
    main()
