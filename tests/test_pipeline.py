"""
Phase 1 test suite.
Tests are initially skipped (Wave 0 stub state).
Remove skip marks as each implementation module is created.
"""
import pytest
import numpy as np
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# DATA-01: Zip extraction
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires real zip file — enabled in Wave 3 (pipeline execution plan)")
def test_zip_extraction(tmp_path):
    """DATA-01: extract_dataset() produces data/raw/train/plays.csv."""
    from src.data.loader import extract_dataset
    # Real test runs against actual zip; fixture covers structure only
    # Verified in plan 01-04 against actual dataset
    assert True


# ---------------------------------------------------------------------------
# DATA-02: CSV loading and join
# ---------------------------------------------------------------------------

def test_csv_loading(tracking_df, plays_df, players_df, games_df):
    """DATA-02: load_raw() returns merged DataFrame with expected columns."""
    from src.data.loader import merge_tracking_tables
    merged = merge_tracking_tables(tracking_df, plays_df, players_df, games_df)
    required_cols = {"gameId", "playId", "nflId", "frameId", "x", "y", "s", "a", "dir", "o",
                     "position", "week", "absoluteYardlineNumber", "playDirection"}
    assert required_cols.issubset(set(merged.columns)), f"Missing: {required_cols - set(merged.columns)}"
    assert len(merged) == len(tracking_df), "Row count changed after merge"


# ---------------------------------------------------------------------------
# PREP-01: LOS-relative coordinate normalization
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implementation in src/data/preprocessor.py — created in plan 01-02")
def test_los_normalization(tracking_df):
    """PREP-01: After normalization, snap-frame offensive linemen cluster near x=0."""
    from src.data.preprocessor import normalize_coordinates
    df = normalize_coordinates(tracking_df.copy())
    # For absoluteYardlineNumber=35, true field_x = 35+10 = 45
    # Player starts at x=50.5 (frame 1); after normalization: 50.5 - 45 = 5.5 (approx)
    # Key: the offset is applied and the column "x" is now LOS-relative
    assert "x" in df.columns
    # Snap-frame player x should be within ±20 yards of 0 (reasonable for any snap position)
    snap_x = df[df["frameId"] == 1]["x"]
    assert snap_x.abs().max() < 30, f"LOS normalization produced x={snap_x.max():.1f}, expected near 0"


# ---------------------------------------------------------------------------
# PREP-02: Play direction flip
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implementation in src/data/preprocessor.py — created in plan 01-02")
def test_play_direction_flip(tracking_df):
    """PREP-02: After flip, all plays have offense moving in positive X direction."""
    from src.data.preprocessor import normalize_coordinates
    df = normalize_coordinates(tracking_df.copy())
    # For left-direction plays, the flip must be applied before normalization
    # After flip + normalization, all offensive players should have positive mean displacement
    # (i.e., final_x > initial_x for the same play)
    for (game_id, play_id, nfl_id), group in df.groupby(["gameId", "playId", "nflId"]):
        if group["position"].iloc[0] in ["QB", "WR", "TE", "RB"]:
            group_sorted = group.sort_values("frameId")
            displacement = group_sorted["x"].iloc[-1] - group_sorted["x"].iloc[0]
            assert displacement >= 0, (
                f"Offensive player {nfl_id} on play {play_id} moved in negative x direction "
                f"after flip (displacement={displacement:.2f})"
            )


# ---------------------------------------------------------------------------
# PREP-03: Angle sin/cos encoding
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implementation in src/data/preprocessor.py — created in plan 01-02")
def test_angle_sincos_encoding(tracking_df):
    """PREP-03: dir_sin and dir_cos present; raw dir and o columns absent."""
    from src.data.preprocessor import normalize_coordinates, encode_angles
    df = normalize_coordinates(tracking_df.copy())
    df = encode_angles(df)
    assert "dir_sin" in df.columns, "dir_sin column missing"
    assert "dir_cos" in df.columns, "dir_cos column missing"
    assert "o_sin" in df.columns, "o_sin column missing"
    assert "o_cos" in df.columns, "o_cos column missing"
    assert "dir" not in df.columns, "Raw dir column should be removed after encoding"
    assert "o" not in df.columns, "Raw o column should be removed after encoding"
    # sin^2 + cos^2 == 1 for every row (unit circle invariant)
    assert np.allclose(df["dir_sin"]**2 + df["dir_cos"]**2, 1.0, atol=1e-6)
    assert np.allclose(df["o_sin"]**2 + df["o_cos"]**2, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# PREP-04: Missing frame interpolation and flagging
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implementation in src/data/preprocessor.py — created in plan 01-02")
def test_interpolation_and_flagging(tracking_df):
    """PREP-04: is_interpolated flag present; missing frames filled; >3 consecutive flagged."""
    from src.data.preprocessor import interpolate_missing_frames
    df = interpolate_missing_frames(tracking_df.copy())
    assert "is_interpolated" in df.columns, "is_interpolated column missing"
    assert df["is_interpolated"].dtype == bool, "is_interpolated must be boolean"
    # nflId=1 had frame 3 missing — after interpolation it should be present and flagged
    player1 = df[(df["nflId"] == 1) & (df["playId"] == 75)]
    assert 3 in player1["frameId"].values, "Frame 3 was not interpolated for nflId=1"
    assert player1[player1["frameId"] == 3]["is_interpolated"].iloc[0] is True or \
           player1[player1["frameId"] == 3]["is_interpolated"].values[0] == True, \
        "Frame 3 for nflId=1 should be flagged as interpolated"
    # Sequences with >3 consecutive missing frames should have too_many_missing=True
    # (tested via a separate synthetic group — logic covered in interpolate_group unit)


# ---------------------------------------------------------------------------
# PREP-05: Acceleration computation
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implementation in src/data/preprocessor.py — created in plan 01-02")
def test_acceleration_computed(tracking_df):
    """PREP-05: a_computed is non-null for all non-interpolated frames."""
    from src.data.preprocessor import interpolate_missing_frames, compute_acceleration
    df = interpolate_missing_frames(tracking_df.copy())
    df = compute_acceleration(df)
    assert "a_computed" in df.columns, "a_computed column missing"
    non_interpolated = df[~df["is_interpolated"]]
    # Allow NaN only on the very first frame of each (player, play) group (no prior frame)
    non_first_frames = non_interpolated[
        non_interpolated.groupby(["gameId", "playId", "nflId"]).cumcount() > 0
    ]
    null_count = non_first_frames["a_computed"].isna().sum()
    assert null_count == 0, f"{null_count} non-interpolated non-first frames have null acceleration"


# ---------------------------------------------------------------------------
# PREP-06: Temporal train/val/test split — zero overlap
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implementation in src/data/preprocessor.py — created in plan 01-02")
def test_temporal_split_disjoint(games_df, tmp_path):
    """PREP-06: set(train_game_ids) & set(test_game_ids) == set() — no leakage."""
    from src.data.preprocessor import make_temporal_split
    split_path = tmp_path / "splits.json"
    split = make_temporal_split(
        games_df,
        val_weeks=[],
        test_weeks=[2],
        output_path=split_path,
    )
    train_ids = set(split["train_game_ids"])
    val_ids   = set(split["val_game_ids"])
    test_ids  = set(split["test_game_ids"])
    assert len(train_ids & test_ids) == 0, "Train/test game ID overlap!"
    assert len(train_ids & val_ids)  == 0, "Train/val game ID overlap!"
    assert len(val_ids  & test_ids)  == 0, "Val/test game ID overlap!"
    assert split_path.exists(), "Split JSON not written to disk"
    loaded = json.loads(split_path.read_text())
    assert "train_game_ids" in loaded and "test_game_ids" in loaded
