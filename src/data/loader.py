"""
Data loading utilities: ZIP extraction, CSV loading, and table merging.

Pattern 1: extract_dataset — idempotent ZIP extraction to data/raw/
Pattern 2: merge_tracking_tables — left-join tracking with plays/players/games
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


def extract_dataset(zip_path: Path, dest_dir: Path) -> None:
    """Extract competition zip to dest_dir. Idempotent — skips if already extracted.

    Args:
        zip_path: Path to the competition ZIP file.
        dest_dir: Destination directory (will be created if missing).

    Raises:
        AssertionError: If plays.csv is not found after extraction.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)

    # Idempotency check — skip if already extracted
    if (dest_dir / "train").exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    assert (dest_dir / "train" / "plays.csv").exists(), (
        f"Extraction completed but plays.csv not found in {dest_dir / 'train'}. "
        "Check ZIP structure."
    )


def merge_tracking_tables(
    tracking: pd.DataFrame,
    plays: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Join tracking with plays, players, and games tables.

    Merge order:
        1. tracking + plays on ["gameId", "playId"]  (adds absoluteYardlineNumber, playDirection)
        2. result + players on "nflId"               (adds position)
        3. result + games on "gameId"                (adds week)

    All merges are left joins to preserve every tracking row.

    Args:
        tracking: Raw tracking DataFrame (one row per player per frame).
        plays: plays.csv DataFrame.
        players: players.csv DataFrame.
        games: games.csv DataFrame.

    Returns:
        Merged DataFrame with the same row count as tracking.

    Raises:
        AssertionError: If row count changes after merging.
    """
    n_rows = len(tracking)

    # Columns we need from each table (avoid duplicate columns after merge)
    plays_cols = ["gameId", "playId", "absoluteYardlineNumber", "playDirection"]
    # Only keep plays columns that exist (handles minimal fixtures)
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
    """Load and merge all raw tracking CSVs with auxiliary tables.

    Globs all tracking_week_*.csv files in data_dir/train/, concatenates them,
    then calls merge_tracking_tables with plays, players, and games.

    Args:
        data_dir: Root data directory (should contain a 'train/' subdirectory).

    Returns:
        Merged tracking DataFrame with columns from all auxiliary tables.
    """
    from tqdm import tqdm  # lazy import — optional dependency at load time

    train_dir = Path(data_dir) / "train"
    tracking_files = sorted(train_dir.glob("tracking_week_*.csv"))

    if not tracking_files:
        raise FileNotFoundError(
            f"No tracking_week_*.csv files found in {train_dir}. "
            "Run extract_dataset() first."
        )

    dfs = []
    for path in tqdm(tracking_files, desc="Loading tracking CSVs"):
        dfs.append(pd.read_csv(path))
    tracking = pd.concat(dfs, ignore_index=True)

    plays = pd.read_csv(train_dir / "plays.csv")
    players = pd.read_csv(train_dir / "players.csv")
    games = pd.read_csv(train_dir / "games.csv")

    return merge_tracking_tables(tracking, plays, players, games)
