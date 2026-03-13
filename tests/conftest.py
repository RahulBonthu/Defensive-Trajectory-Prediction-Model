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
            rows.append({
                "gameId": game_id,
                "playId": play_id,
                "nflId": nfl_id,
                "displayName": f"Player_{nfl_id}" if nfl_id != 99 else "football",
                "position": pos,
                "frameId": frame_id,
                "time": f"2022-09-11T00:00:0{frame_id}.000",
                "x": 50.0 + frame_id * 0.5,
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
