"""
Preprocessing utilities for NFL tracking data.

Pipeline order (must be followed):
    1. normalize_coordinates  — LOS-relative x, centered y, direction flip for left plays
    2. encode_angles          — dir/o degrees -> sin/cos columns (call AFTER normalize)
    3. interpolate_missing_frames — fill gaps, flag interpolated rows (call AFTER encode_angles)
    4. compute_acceleration   — diff(s) * fps per group, NaN on interpolated rows
    5. make_temporal_split    — week-based train/val/test split, writes JSON

Note: interpolate_missing_frames operates on dir_sin/dir_cos (already encoded),
      NOT on raw degree columns.  This is the safer path because sin/cos values
      interpolate linearly and remain bounded, whereas raw degrees can wrap.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# PREP-01 / PREP-02: coordinate normalisation + direction flip
# ---------------------------------------------------------------------------

def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize tracking coordinates to LOS-relative, direction-agnostic frame.

    Steps (order is critical):
        1.  df = df.copy()  — pandas 3.0 CoW safety
        2.  Compute los_field_x = absoluteYardlineNumber + 10
        3.  Create left_mask = playDirection == "left"
        4.  Mirror x for left plays:  x -> 120 - x
        5.  Mirror y for left plays:  y -> 53.3 - y
        6.  Flip angles for left plays before encoding:
              dir -> (dir + 180) % 360
              o   -> (o   + 180) % 360
        7.  Mirror los_field_x for left plays: los_x -> 120 - los_x
        8.  Assign df["los_x"] = los_field_x
        9.  Subtract LOS:  x -> x - los_x
        10. Centre laterally: y -> y - 26.65

    Note: los_x is retained in the output for debugging.

    Args:
        df: Merged tracking DataFrame with columns:
            x, y, dir, o, absoluteYardlineNumber, playDirection.

    Returns:
        New DataFrame with LOS-relative, direction-normalised coordinates.
    """
    # Step 1 — CoW-safe copy
    df = df.copy()

    # Step 2 — line-of-scrimmage field position
    los_field_x = df["absoluteYardlineNumber"] + 10

    # Step 3 — mask for leftward plays
    left_mask = df["playDirection"] == "left"

    # Step 4 — mirror x
    df.loc[left_mask, "x"] = 120 - df.loc[left_mask, "x"]

    # Step 5 — mirror y
    df.loc[left_mask, "y"] = 53.3 - df.loc[left_mask, "y"]

    # Step 6 — flip angles BEFORE sin/cos encoding
    df.loc[left_mask, "dir"] = (df.loc[left_mask, "dir"] + 180) % 360
    df.loc[left_mask, "o"]   = (df.loc[left_mask, "o"]   + 180) % 360

    # Step 7 — mirror los_field_x for left plays
    los_field_x = los_field_x.copy()  # avoid modifying view
    los_field_x.loc[left_mask] = 120 - los_field_x.loc[left_mask]

    # Step 8 — store los_x (useful for debugging)
    df["los_x"] = los_field_x.values

    # Step 9 — subtract LOS to make x relative
    df["x"] = df["x"] - df["los_x"]

    # Step 10 — centre field laterally (field width = 53.3 yards)
    df["y"] = df["y"] - 26.65

    return df


# ---------------------------------------------------------------------------
# PREP-03: angle sin/cos encoding
# ---------------------------------------------------------------------------

def encode_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw dir and o degree columns with sin/cos components.

    Must be called AFTER normalize_coordinates (angles already flipped for
    left-direction plays).  Drops raw "dir" and "o" columns.

    Args:
        df: DataFrame with "dir" and "o" columns in degrees.

    Returns:
        New DataFrame with dir_sin, dir_cos, o_sin, o_cos; raw degrees removed.
    """
    df = df.copy()

    dir_rad = np.radians(df["dir"])
    o_rad   = np.radians(df["o"])

    df["dir_sin"] = np.sin(dir_rad)
    df["dir_cos"] = np.cos(dir_rad)
    df["o_sin"]   = np.sin(o_rad)
    df["o_cos"]   = np.cos(o_rad)

    df = df.drop(columns=["dir", "o"])

    return df


# ---------------------------------------------------------------------------
# PREP-04: missing frame interpolation
# ---------------------------------------------------------------------------

def _interpolate_group(
    group: pd.DataFrame,
    max_gap: int,
    interp_cols: list[str],
) -> pd.DataFrame:
    """Interpolate a single (gameId, playId, nflId) group.

    Args:
        group: Sub-DataFrame for one player-play.
        max_gap: Maximum allowable consecutive missing frames.
        interp_cols: Columns to interpolate.

    Returns:
        Group with missing frames filled (or flagged with too_many_missing).
    """
    group = group.sort_values("frameId").copy()

    min_f = int(group["frameId"].min())
    max_f = int(group["frameId"].max())
    all_frames  = set(range(min_f, max_f + 1))
    have_frames = set(group["frameId"].values)
    missing     = sorted(all_frames - have_frames)

    # Flag rows that existed from the start as not-interpolated
    group["is_interpolated"] = False
    group["too_many_missing"] = False

    if not missing:
        return group

    # Check longest consecutive run of missing frames
    if len(missing) > 1:
        diffs = np.diff(missing)
        max_run = int(np.max(np.diff(np.concatenate([[missing[0] - 1], missing, [missing[-1] + 1]])
                                     .tolist())) if False else
                   # Proper run-length encoding via diff approach
                   _max_consecutive_run(missing))
    else:
        max_run = 1

    if max_run > max_gap:
        group["too_many_missing"] = True
        return group

    # Build full-frame index
    full_index = pd.RangeIndex(min_f, max_f + 1)
    group = group.set_index("frameId").reindex(full_index)
    group.index.name = "frameId"
    group = group.reset_index()

    # Mark newly inserted rows
    group["is_interpolated"] = group["is_interpolated"].isna()  # NaN -> True (newly inserted)
    group["too_many_missing"] = group["too_many_missing"].fillna(False)

    # Interpolate numeric columns
    known_mask = ~group["is_interpolated"]
    x_known = group.loc[known_mask, "frameId"].values.astype(float)

    for col in interp_cols:
        if col not in group.columns:
            continue
        y_known = group.loc[known_mask, col].values.astype(float)
        if len(x_known) < 2:
            continue
        f = interp1d(x_known, y_known, kind="linear", bounds_error=False, fill_value="extrapolate")
        group.loc[group["is_interpolated"], col] = f(
            group.loc[group["is_interpolated"], "frameId"].values.astype(float)
        )

    # Forward-fill non-numeric metadata columns from adjacent rows
    meta_cols = [c for c in group.columns if c not in interp_cols + ["frameId", "is_interpolated", "too_many_missing"]]
    group[meta_cols] = group[meta_cols].ffill().bfill()

    return group


def _max_consecutive_run(sorted_missing: list[int]) -> int:
    """Return the length of the longest consecutive run in sorted_missing."""
    if not sorted_missing:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(sorted_missing)):
        if sorted_missing[i] == sorted_missing[i - 1] + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def interpolate_missing_frames(
    df: pd.DataFrame,
    max_gap: int = 3,
) -> pd.DataFrame:
    """Fill missing frames per (nflId, gameId, playId) group via linear interpolation.

    Operates on dir_sin/dir_cos (post-encode_angles) rather than raw degrees.
    Must be called AFTER encode_angles.

    Interpolated columns: x, y, s, dir_sin, dir_cos, o_sin, o_cos.
    Sequences with a consecutive missing run > max_gap are flagged
    too_many_missing=True and skipped.

    Args:
        df: Tracking DataFrame (post encode_angles).
        max_gap: Maximum consecutive missing frames before skipping interpolation.

    Returns:
        DataFrame with missing frames inserted and is_interpolated / too_many_missing flags.
    """
    interp_cols = [c for c in ["x", "y", "s", "dir_sin", "dir_cos", "o_sin", "o_cos"] if c in df.columns]

    parts = []
    for (game_id, play_id, nfl_id), group in df.groupby(
        ["gameId", "playId", "nflId"], sort=False
    ):
        processed = _interpolate_group(group, max_gap=max_gap, interp_cols=interp_cols)
        # groupby drops the key columns from the group; restore them
        processed["gameId"] = game_id
        processed["playId"] = play_id
        processed["nflId"] = nfl_id
        parts.append(processed)

    result = pd.concat(parts, ignore_index=True)
    result["is_interpolated"] = result["is_interpolated"].astype(bool)
    result["too_many_missing"] = result["too_many_missing"].astype(bool)

    return result


# ---------------------------------------------------------------------------
# PREP-05: acceleration computation
# ---------------------------------------------------------------------------

def compute_acceleration(df: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
    """Compute a_computed = diff(speed) * fps per (gameId, playId, nflId) group.

    For interpolated frames, a_computed is set to NaN then filled with the
    per-(play, player) mean acceleration.  The first frame of each group is
    always NaN before fill (no prior frame to diff against).

    Args:
        df: DataFrame with "s" (speed) and "is_interpolated" columns.
        fps: Frames per second of the tracking data (default 10).

    Returns:
        DataFrame with "a_computed" column added.
    """
    df = df.copy()

    def _accel_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("frameId")
        a = g["s"].diff() * fps
        # Nullify interpolated frames
        a[g["is_interpolated"].values] = np.nan
        return a

    df["a_computed"] = (
        df.groupby(["gameId", "playId", "nflId"], group_keys=False)
        .apply(_accel_group)
        .values
    )

    # Fill NaN (interpolated frames + first frame) with per-(play, player) mean
    def _fill_group(g: pd.DataFrame) -> pd.Series:
        mean_a = g["a_computed"].mean()
        return g["a_computed"].fillna(mean_a)

    df["a_computed"] = (
        df.groupby(["gameId", "playId", "nflId"], group_keys=False)
        .apply(_fill_group)
        .values
    )

    return df


# ---------------------------------------------------------------------------
# PREP-06: temporal train/val/test split
# ---------------------------------------------------------------------------

def make_temporal_split(
    games: pd.DataFrame,
    val_weeks: list[int],
    test_weeks: list[int],
    output_path: Path,
) -> dict:
    """Split games by week into disjoint train/val/test sets.

    Uses pandas 3.0 CoW-safe boolean indexing.  Asserts all three sets are
    pairwise disjoint.  Writes JSON with keys train_game_ids, val_game_ids,
    test_game_ids to output_path.

    Args:
        games: games.csv DataFrame with "gameId" and "week" columns.
        val_weeks: Week numbers assigned to the validation set.
        test_weeks: Week numbers assigned to the test set.
        output_path: Where to write splits.json.

    Returns:
        dict with keys train_game_ids, val_game_ids, test_game_ids (sorted lists).

    Raises:
        AssertionError: If any two splits share game IDs.
    """
    output_path = Path(output_path)

    val_mask  = games["week"].isin(val_weeks)
    test_mask = games["week"].isin(test_weeks)
    train_mask = ~val_mask & ~test_mask

    train_ids = sorted(games.loc[train_mask, "gameId"].tolist())
    val_ids   = sorted(games.loc[val_mask,   "gameId"].tolist())
    test_ids  = sorted(games.loc[test_mask,  "gameId"].tolist())

    # Disjointness assertions
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_set & test_set), (
        f"Train/test overlap: {train_set & test_set}"
    )
    assert not (train_set & val_set), (
        f"Train/val overlap: {train_set & val_set}"
    )
    assert not (val_set & test_set), (
        f"Val/test overlap: {val_set & test_set}"
    )

    split = {
        "train_game_ids": train_ids,
        "val_game_ids":   val_ids,
        "test_game_ids":  test_ids,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)

    return split
