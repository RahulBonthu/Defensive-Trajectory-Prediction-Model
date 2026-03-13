"""
Data loading utilities: ZIP extraction, CSV loading, and table merging.

BDB 2026 competition format (updated 2026-03-13):
    - ZIP contains train/input_2023_w*.csv and train/output_2023_w*.csv files
    - No separate plays/games/players CSVs — all metadata is embedded per row
    - Columns use snake_case: game_id, play_id, nfl_id, frame_id, player_position,
      play_direction, absolute_yardline_number
    - ball_land_x / ball_land_y are provided directly per row (no derivation needed)
    - Week is encoded in the filename (input_2023_w01.csv -> week=1)
    - player_to_predict=True marks the defensive player whose trajectory is the target

Column mapping to camelCase (for compatibility with preprocessor):
    game_id                  -> gameId
    play_id                  -> playId
    nfl_id                   -> nflId
    frame_id                 -> frameId
    player_position          -> position
    play_direction           -> playDirection
    absolute_yardline_number -> absoluteYardlineNumber
    ball_land_x              -> ball_land_x  (kept as-is, new column)
    ball_land_y              -> ball_land_y  (kept as-is, new column)
    player_to_predict        -> player_to_predict (kept as-is)
    player_side              -> player_side  (kept as-is)
    player_role              -> player_role  (kept as-is)
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd


# Column rename map: snake_case -> camelCase for preprocessor compatibility
_RENAME_MAP = {
    "game_id": "gameId",
    "play_id": "playId",
    "nfl_id": "nflId",
    "frame_id": "frameId",
    "player_position": "position",
    "play_direction": "playDirection",
    "absolute_yardline_number": "absoluteYardlineNumber",
}


def extract_dataset(zip_path: Path, dest_dir: Path) -> None:
    """Extract competition zip to dest_dir. Idempotent — skips if already extracted.

    BDB 2026 format: expects train/input_2023_w01.csv to exist after extraction.

    Args:
        zip_path: Path to the competition ZIP file.
        dest_dir: Destination directory (will be created if missing).

    Raises:
        AssertionError: If train/input_2023_w01.csv is not found after extraction.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)

    # Idempotency check — skip if already extracted
    if (dest_dir / "train").exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    assert (dest_dir / "train" / "input_2023_w01.csv").exists(), (
        f"Extraction completed but input_2023_w01.csv not found in {dest_dir / 'train'}. "
        "Check ZIP structure — expected BDB 2026 competition format."
    )


def _load_input_week(csv_path: Path, week: int) -> pd.DataFrame:
    """Load one input_2023_w*.csv file and add a week column.

    Args:
        csv_path: Path to the input CSV file.
        week: Week number to attach as a column.

    Returns:
        DataFrame with columns renamed to camelCase and week added.
    """
    df = pd.read_csv(csv_path)
    df["week"] = week

    # Rename snake_case columns to camelCase for preprocessor compatibility
    df = df.rename(columns=_RENAME_MAP)

    return df


def merge_tracking_tables(
    tracking: pd.DataFrame,
    plays: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Legacy function for test fixture compatibility.

    In BDB 2026 format, all auxiliary data is embedded in the input CSVs.
    This function supports the synthetic fixture-based unit tests by
    performing the same left-join logic as before.

    All merges are left joins to preserve every tracking row.

    Args:
        tracking: Raw tracking DataFrame (one row per player per frame).
        plays: plays DataFrame with absoluteYardlineNumber and playDirection.
        players: players DataFrame with position column.
        games: games DataFrame with week column.

    Returns:
        Merged DataFrame with the same row count as tracking.

    Raises:
        AssertionError: If row count changes after merging.
    """
    n_rows = len(tracking)

    # Columns we need from each table (avoid duplicate columns after merge)
    plays_cols = ["gameId", "playId", "absoluteYardlineNumber", "playDirection"]
    plays_cols = [c for c in plays_cols if c in plays.columns]

    players_cols = ["nflId", "position"]
    players_cols = [c for c in players_cols if c in players.columns]

    games_cols = ["gameId", "week"]
    games_cols = [c for c in games_cols if c in games.columns]

    # Drop columns from tracking that we will bring in from other tables,
    # to avoid pandas suffix collisions while allowing the tracking values
    # to be replaced with canonical per-play values.
    plays_extra = [c for c in plays_cols if c not in ("gameId", "playId") and c in tracking.columns]
    players_extra = [c for c in players_cols if c not in ("nflId",) and c in tracking.columns]
    games_extra = [c for c in games_cols if c not in ("gameId",) and c in tracking.columns]
    cols_to_drop = list(set(plays_extra + players_extra + games_extra))

    merged = tracking.copy()
    if cols_to_drop:
        merged = merged.drop(columns=cols_to_drop)

    # Step 1: merge with plays
    merged = merged.merge(
        plays[plays_cols],
        on=["gameId", "playId"],
        how="left",
    )

    # Step 2: merge with players
    if len(players_cols) > 1:  # need at least nflId + one extra
        merged = merged.merge(
            players[players_cols],
            on="nflId",
            how="left",
        )

    # Step 3: merge with games
    if len(games_cols) > 1:  # need at least gameId + one extra
        merged = merged.merge(
            games[games_cols],
            on="gameId",
            how="left",
        )

    assert len(merged) == n_rows, (
        f"Row count changed after merge: expected {n_rows}, got {len(merged)}. "
        "Check for duplicate keys in plays/players/games tables."
    )

    return merged


def load_raw(data_dir: Path) -> pd.DataFrame:
    """Load and merge all raw tracking CSVs from BDB 2026 competition format.

    Globs all input_2023_w*.csv files in data_dir/train/, loads each with its
    week number, and concatenates into a single DataFrame.

    Column names are renamed to camelCase for preprocessor compatibility.
    A 'week' column is derived from the filename (input_2023_w01.csv -> 1).

    Args:
        data_dir: Root data directory (should contain a 'train/' subdirectory).

    Returns:
        Combined tracking DataFrame with all weeks merged together.
    """
    from tqdm import tqdm  # lazy import — optional dependency at load time

    train_dir = Path(data_dir) / "train"
    input_files = sorted(train_dir.glob("input_2023_w*.csv"))

    if not input_files:
        raise FileNotFoundError(
            f"No input_2023_w*.csv files found in {train_dir}. "
            "Run extract_dataset() first."
        )

    dfs = []
    for path in tqdm(input_files, desc="Loading input CSVs"):
        # Extract week number from filename: input_2023_w03.csv -> 3
        match = re.search(r"_w(\d+)\.csv$", path.name)
        week = int(match.group(1)) if match else 0
        dfs.append(_load_input_week(path, week))

    return pd.concat(dfs, ignore_index=True)
