import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def games_df():
    """Minimal games.csv analog. Two games across two weeks."""
    return pd.DataFrame({
        "gameId": [2022091100, 2022091800],
        "season": [2022, 2022],
        "week": [1, 2],
        "homeTeamAbbr": ["KC", "BUF"],
        "visitorTeamAbbr": ["ARI", "TEN"],
    })


@pytest.fixture
def plays_df():
    """Minimal plays.csv analog. One left-direction and one right-direction play."""
    return pd.DataFrame({
        "gameId": [2022091100, 2022091800],
        "playId": [75, 110],
        "possessionTeam": ["KC", "BUF"],
        "defensiveTeam": ["ARI", "TEN"],
        "absoluteYardlineNumber": [35, 25],   # true field x = 45, 35 (before +10 offset)
        "playDirection": ["right", "left"],
        "passResult": ["C", "C"],
        "offensePlayResult": [8, 12],
    })


@pytest.fixture
def tracking_df(plays_df, games_df):
    """
    Minimal tracking_week_N.csv analog.
    - 2 plays x 3 players x 5 frames = 30 rows
    - Includes one QB (offense), one CB (defense), one football row per play
    - One frame intentionally missing (frameId=3 absent for nflId=1 in play 75) to test interpolation
    - playDirection="left" play has x coords that should be mirrored
    """
    rows = []
    players_info = [
        # nflId, position, play_id, game_id, direction
        (1, "QB",       75, 2022091100, "right"),
        (2, "CB",       75, 2022091100, "right"),
        (99, "football", 75, 2022091100, "right"),
        (3, "QB",       110, 2022091800, "left"),
        (4, "CB",       110, 2022091800, "left"),
        (99, "football", 110, 2022091800, "left"),
    ]
    for nfl_id, pos, play_id, game_id, play_dir in players_info:
        direction = play_dir
        for frame_id in [1, 2, 3, 4, 5]:
            # Skip frame 3 for nflId=1 to simulate missing frame
            if nfl_id == 1 and frame_id == 3:
                continue
            # For left-directed plays, raw x should decrease (offense moving toward x=0).
            # Using 60.0 - frame_id * 0.5 ensures that after the 120-x mirror,
            # normalized x increases with frameId (positive displacement) — matching
            # the test_play_direction_flip assertion.  The mirrored snap frame x
            # is ~60.5, and los_x=85, giving norm x ~ -24.5 which satisfies |x|<30.
            raw_x = (50.0 + frame_id * 0.5) if direction == "right" else (60.0 - frame_id * 0.5)
            rows.append({
                "gameId": game_id,
                "playId": play_id,
                "nflId": nfl_id,
                "displayName": f"Player_{nfl_id}" if nfl_id != 99 else "football",
                "position": pos,
                "frameId": frame_id,
                "time": f"2022-09-11T00:00:0{frame_id}.000",
                "x": raw_x,
                "y": 26.65 + frame_id * 0.1,
                "s": 2.0 + frame_id * 0.1,
                "a": 0.5,
                "dis": 0.2,
                "o": 90.0 if direction == "right" else 270.0,
                "dir": 0.0 if direction == "right" else 180.0,
                "event": None,
                "playDirection": direction,
                "absoluteYardlineNumber": 35 if play_id == 75 else 25,
                "week": 1 if game_id == 2022091100 else 2,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def players_df():
    """Minimal players.csv analog."""
    return pd.DataFrame({
        "nflId": [1, 2, 3, 4],
        "displayName": ["Player_1", "Player_2", "Player_3", "Player_4"],
        "position": ["QB", "CB", "QB", "CB"],
        "height": ["6-2", "5-11", "6-1", "6-0"],
        "weight": [220, 195, 215, 200],
    })


@pytest.fixture
def minimal_samples():
    """6 synthetic sample dicts matching the build_samples() schema.

    Positions: 4 CB, 1 FS, 1 SS.
    T=25 frames; first 10 are real (non-zero), frames 10-24 are padded (zero).
    No disk I/O — all values are hardcoded constants.
    """
    T = 25
    real_frames = 10
    positions = ["CB", "CB", "CB", "CB", "FS", "SS"]
    game_ids   = [2023010100, 2023010100, 2023010100, 2023010100, 2023010200, 2023010200]
    play_ids   = [1, 2, 3, 4, 1, 2]
    nfl_ids    = [100, 101, 102, 103, 104, 105]
    samples = []
    for i in range(6):
        frames = np.zeros((T, 8), dtype=np.float32)
        frames[:real_frames] = np.ones((real_frames, 8), dtype=np.float32) * (i + 1)
        mask = np.zeros(T, dtype=bool)
        mask[:real_frames] = True
        samples.append({
            "gameId": game_ids[i],
            "playId": play_ids[i],
            "nflId": nfl_ids[i],
            "frames": frames,
            "padding_mask": mask,
            "target_xy": np.array([2.5, -1.0], dtype=np.float32),
            "ball_target_xy": np.array([15.0, 3.0], dtype=np.float32),
            "position": positions[i],
        })
    return samples


@pytest.fixture
def minimal_context_df():
    """Synthetic tracking-like DataFrame for social context assembly.

    Covers 4 game-play combos (2 games × 2 plays each).
    3 players × 10 frames per combo = 120 rows per combo → 480 rows total.
    Players: nflId 100 (CB), 200 (QB), 201 (WR).
    Includes ball_land_x / ball_land_y columns matching minimal_samples ball_target_xy.
    No disk I/O.
    """
    rows = []
    game_play_combos = [
        (2023010100, 1),
        (2023010100, 2),
        (2023010100, 3),
        (2023010100, 4),
        (2023010200, 1),
        (2023010200, 2),
    ]
    player_positions = [(100, "CB"), (200, "QB"), (201, "WR")]
    for game_id, play_id in game_play_combos:
        for frame_id in range(1, 11):
            for nfl_id, position in player_positions:
                rows.append({
                    "gameId": game_id,
                    "playId": play_id,
                    "frameId": frame_id,
                    "nflId": nfl_id,
                    "x": nfl_id * 0.01 + frame_id * 0.5,
                    "y": frame_id * 0.2,
                    "position": position,
                    "ball_land_x": 15.0,
                    "ball_land_y": 3.0,
                })
    return pd.DataFrame(rows)
