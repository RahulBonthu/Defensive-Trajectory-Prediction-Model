"""
Normalization validation and schema inspection tool.

Modes:
    Default:          Load data/processed/cleaned.parquet, sample 50 plays,
                      plot each offensive player's (x, y) trajectory,
                      save PNG to outputs/validation/50_play_overlay.png.
    --show-schema:    Print plays.csv columns from data/raw/train/plays.csv.
    --show-positions: Print value counts for the "position" column.

Usage:
    python scripts/validate_normalization.py
    python scripts/validate_normalization.py --show-schema
    python scripts/validate_normalization.py --show-positions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _show_schema(data_dir: Path) -> None:
    plays_path = data_dir / "train" / "plays.csv"
    if not plays_path.exists():
        print(f"plays.csv not found at {plays_path}")
        return
    df = pd.read_csv(plays_path, nrows=0)
    print(f"plays.csv columns ({len(df.columns)} total):")
    for col in df.columns:
        print(f"  - {col}")


def _show_positions(cleaned_path: Path) -> None:
    if not cleaned_path.exists():
        print(f"cleaned.parquet not found at {cleaned_path}")
        return
    df = pd.read_parquet(cleaned_path, columns=["position"])
    print("Position value counts:")
    print(df["position"].value_counts().to_string())


def _plot_overlay(cleaned_path: Path, output_path: Path, n_plays: int = 50) -> None:
    """Plot x/y trajectories for offensive players over n_plays, save PNG."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    if not cleaned_path.exists():
        print(f"cleaned.parquet not found at {cleaned_path}")
        return

    df = pd.read_parquet(cleaned_path)

    # Sample up to n_plays unique play IDs
    play_ids = df["playId"].unique()
    rng = __import__("numpy").random.default_rng(seed=42)
    sampled_ids = rng.choice(play_ids, size=min(n_plays, len(play_ids)), replace=False)

    sample_df = df[df["playId"].isin(sampled_ids)]

    # Offensive positions (Phase 1 focus: any non-defensive, non-football)
    offensive_positions = {"QB", "WR", "TE", "RB", "FB", "C", "G", "T", "OL"}
    off_df = sample_df[sample_df["position"].isin(offensive_positions)]

    fig, ax = plt.subplots(figsize=(14, 6))

    for (play_id, nfl_id), group in off_df.groupby(["playId", "nflId"]):
        g = group.sort_values("frameId")
        ax.plot(g["x"], g["y"], alpha=0.15, linewidth=0.8, color="steelblue")

    # LOS reference line at x=0 (all plays are LOS-relative after normalization)
    ax.axvline(x=0, color="red", linewidth=1.5, linestyle="--", label="LOS (x=0)")

    ax.set_xlabel("x (yards, LOS-relative)")
    ax.set_ylabel("y (yards, centred laterally)")
    ax.set_title(f"Offensive player trajectories — {len(sampled_ids)} sampled plays")
    ax.legend()
    ax.set_xlim(-20, 40)
    ax.set_ylim(-30, 30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay plot -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate coordinate normalization and inspect dataset schema."
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Print plays.csv column names (for ball-landing coordinate verification).",
    )
    parser.add_argument(
        "--show-positions",
        action="store_true",
        help="Print position value counts from cleaned.parquet.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw data directory (default: data/raw).",
    )
    parser.add_argument(
        "--cleaned-parquet",
        type=Path,
        default=Path("data/processed/cleaned.parquet"),
        help="Path to cleaned.parquet (default: data/processed/cleaned.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Output directory for visualisations (default: outputs/validation).",
    )
    args = parser.parse_args()

    if args.show_schema:
        _show_schema(args.data_dir)
    elif args.show_positions:
        _show_positions(args.cleaned_parquet)
    else:
        output_png = args.output_dir / "50_play_overlay.png"
        _plot_overlay(args.cleaned_parquet, output_png)


if __name__ == "__main__":
    main()
