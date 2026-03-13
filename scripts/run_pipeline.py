"""
End-to-end NFL tracking data pipeline.

BDB 2026 competition format (updated 2026-03-13):
    Input: train/input_2023_w*.csv per week (tracking + metadata + ball_land_x/y all-in-one)
    Column names: snake_case in raw files, renamed to camelCase by loader for preprocessor
    ball_land_x / ball_land_y: directly provided per row (no derivation needed)
    week: encoded in filename, added as column by loader

Usage:
    python scripts/run_pipeline.py --zip-path nfl-big-data-bowl-2026-prediction.zip

Steps:
    1.  extract_dataset      — unzip to data/raw/
    2.  load_raw             — concatenate input_2023_w*.csv files with week column
    3.  Schema inspection    — print input CSV columns for documentation
    4.  normalize_coordinates — LOS-relative coords, direction flip
    5.  encode_angles        — dir/o -> sin/cos
    6.  interpolate_missing_frames
    7.  compute_acceleration
    8.  Save cleaned.parquet
    9.  make_temporal_split  — write splits.json (week-based from filename)
    10. build_samples        — construct player-play tensors
    11. Print sample counts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full NFL tracking data pipeline."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the competition ZIP file. "
             "If omitted, extraction is skipped and data/raw must already exist.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing (or to receive) the raw data (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for pipeline outputs (default: data/processed).",
    )
    parser.add_argument(
        "--val-weeks",
        type=int,
        nargs="*",
        default=[17, 18],
        help="Week numbers for the validation split (default: 17 18).",
    )
    parser.add_argument(
        "--test-weeks",
        type=int,
        nargs="*",
        default=[1, 2],
        help="Week numbers for the test split (default: 1 2).",
    )
    args = parser.parse_args()

    from src.data.loader import extract_dataset, load_raw
    from src.data.preprocessor import (
        normalize_coordinates,
        encode_angles,
        interpolate_missing_frames,
        compute_acceleration,
        make_temporal_split,
    )
    from src.data.sample_builder import build_samples

    # Step 1: extract
    if args.zip_path is not None:
        print(f"[1/9] Extracting {args.zip_path} -> {args.data_dir} ...")
        extract_dataset(args.zip_path, args.data_dir)
    else:
        print("[1/9] --zip-path not provided; assuming data/raw already populated.")

    # Step 2: load
    print("[2/9] Loading raw CSVs (BDB 2026 input_2023_w*.csv format) ...")
    df = load_raw(args.data_dir)
    print(f"      Loaded {len(df):,} tracking rows, {df['gameId'].nunique()} games.")

    # Step 3: schema check — document columns and ball landing coordinates
    print("[3/9] Input CSV columns (ball landing coordinate verification):")
    train_dir = args.data_dir / "train"
    sample_input = sorted(train_dir.glob("input_2023_w*.csv"))
    if sample_input:
        sample_cols = pd.read_csv(sample_input[0], nrows=0).columns.tolist()
        print("     ", sample_cols)
        has_ball_land = "ball_land_x" in sample_cols and "ball_land_y" in sample_cols
        print(f"      ball_land_x/ball_land_y present: {has_ball_land}")
        if has_ball_land:
            print("      Ball landing coordinates are directly provided per row — no derivation needed.")
    else:
        print("      No input CSVs found — skipping schema check.")

    # Step 4: normalize
    print("[4/9] Normalizing coordinates (LOS-relative, direction flip) ...")
    df = normalize_coordinates(df)

    # Step 5: encode angles
    print("[5/9] Encoding angles (sin/cos) ...")
    df = encode_angles(df)

    # Step 6: interpolate
    print("[6/9] Interpolating missing frames ...")
    df = interpolate_missing_frames(df)
    n_interp = df["is_interpolated"].sum()
    print(f"      Interpolated {n_interp:,} frames.")

    # Step 7: acceleration
    print("[7/9] Computing acceleration ...")
    df = compute_acceleration(df)

    # Step 8: save parquet
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = args.output_dir / "cleaned.parquet"
    print(f"[8/9] Saving cleaned data -> {cleaned_path} ...")
    df.to_parquet(cleaned_path, index=False)
    print(f"      Saved {len(df):,} rows, {df.shape[1]} columns.")

    # Step 9: temporal split — use week column added by loader from filename
    print("[9/9] Building temporal train/val/test split ...")
    # Build a synthetic games DataFrame from the loaded data (week already in df)
    games_df = (
        df[["gameId", "week"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    split_path = args.output_dir / "splits.json"
    split = make_temporal_split(
        games_df,
        val_weeks=args.val_weeks,
        test_weeks=args.test_weeks,
        output_path=split_path,
    )
    print(f"      Train games: {len(split['train_game_ids'])}")
    print(f"      Val   games: {len(split['val_game_ids'])}")
    print(f"      Test  games: {len(split['test_game_ids'])}")
    print(f"      Disjointness confirmed. Split written -> {split_path}")

    # Step 10: build samples
    print("[10/10] Building player-play tensor samples ...")
    samples = build_samples(df, split)
    for partition in ("train", "val", "test"):
        n = len(samples[partition])
        print(f"      {partition}: {n:,} samples")
        if n > 0:
            s = samples[partition][0]
            print(
                f"        Example — gameId={s['gameId']}, playId={s['playId']}, "
                f"nflId={s['nflId']}, position={s['position']}, "
                f"frames.shape={s['frames'].shape}, "
                f"target_xy={s['target_xy']}"
            )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
