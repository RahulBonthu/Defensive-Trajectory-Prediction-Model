# Phase 1: Data Pipeline and Validation - Research

**Researched:** 2026-03-13
**Domain:** NFL Big Data Bowl tracking data ingestion, coordinate normalization, interpolation, temporal splitting, visual validation
**Confidence:** MEDIUM-HIGH — stack HIGH, preprocessing patterns HIGH, NFL BDB column names MEDIUM (must be verified against actual zip contents)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | System accepts `nfl-big-data-bowl-2026-prediction.zip` and extracts `train/` and `test/` folders automatically | Python `zipfile` stdlib; extract to `data/raw/` before any loading begins |
| DATA-02 | System loads raw NFL Big Data Bowl CSV tracking files from extracted zip structure | `pandas.read_csv` on `tracking_week_N.csv`, `plays.csv`, `players.csv`, `games.csv`; concatenate weekly tracking files |
| DATA-03 | All changes are committed to GitHub before any step that prompts the user to provide the dataset | Git-first workflow: repo scaffolding, loader, preprocessor all committed before dataset upload step |
| PREP-01 | Player coordinates normalized relative to line of scrimmage (subtract LOS x-coordinate) | `absoluteYardlineNumber + 10` gives true field yard line; subtract from raw `x` column in tracking data |
| PREP-02 | All offensive plays flipped so offense always moves in positive X direction | `playDirection == "left"` → mirror x as `120 - x`, add 180° to `dir` and `o` columns (mod 360) |
| PREP-03 | Direction angle features (`dir`, `o`) encoded as sin/cos to handle 0°/360° discontinuity | `np.sin(np.radians(col))` and `np.cos(np.radians(col))`; replaces raw degree column with two float columns |
| PREP-04 | Missing frames interpolated (x, y, speed, direction); sequences with >3 consecutive missing flagged | `scipy.interpolate.interp1d` per (nflId, gameId, playId) group; boolean `is_interpolated` flag column; run-length encode gaps to detect >3 consecutive |
| PREP-05 | Acceleration computed from velocity when not provided in raw data | `a = (s_t - s_{t-1}) / dt`; set `a = NaN` on interpolated frames; fill with per-play mean acceleration |
| PREP-06 | Train/val/test split by game week before normalization stats computed | Group `gameId`s by `week` from `games.csv`; assign weeks 1–N_train to train, next to val, rest to test; assert disjoint game ID sets |
</phase_requirements>

---

## Summary

Phase 1 is the hard gate for everything else in this project. No model can be trained, no features can be built, and no ablation can be valid unless the data pipeline produces clean, canonically oriented, correctly split player-play samples. The preprocessing work here is non-trivial: NFL tracking data contains plays running in two directions, uses absolute field coordinates, has variable missing-frame rates, and ships in a zip archive that must be extracted before loading. All of this must be resolved and visually confirmed before Phase 2 begins.

The three highest-risk operations in this phase are (1) coordinate normalization and play-direction flip — if done wrong, the model sees a bimodal coordinate distribution and learns nothing meaningful; (2) the train/val/test split strategy — a random play split silently inflates test performance due to within-game correlation; and (3) the `absoluteYardlineNumber` end-zone offset — raw values are 1–50 on a 100-yard scale but the full field is 120 yards including end zones, so 10 must be added before using this as the LOS position. All three pitfalls have visual or assertion-based validation checks that must pass before the phase is considered done.

The data arrives as `nfl-big-data-bowl-2026-prediction.zip` with `train/` and `test/` folders inside. CODE AND REPO STRUCTURE must be committed to GitHub before the user is asked to upload the dataset — this is DATA-03, an explicit project requirement. The pipeline order is: unzip → load CSVs → join on (gameId, playId, nflId) → normalize coordinates → flip direction → encode angles → interpolate gaps → compute acceleration → assign temporal split → save cleaned parquet → build (player, play) sample tensors → run visual validation checks.

**Primary recommendation:** Build `loader.py` → `preprocessor.py` → `sample_builder.py` as three independently testable modules, save intermediate outputs to `data/processed/` as parquet after each stage, and run visual validation (50-play overlay) plus two assertions (split disjointness, acceleration non-null) before declaring the phase complete.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `zipfile` | built-in | Extract `nfl-big-data-bowl-2026-prediction.zip` | No dependency needed; handles nested folder structure cleanly |
| pandas | 3.0.1 | CSV loading, joins, tabular preprocessing | NFL tracking data is heterogeneous tabular; pandas is the standard; 3.0 Copy-on-Write prevents silent mutation bugs in preprocessing pipelines |
| NumPy | 2.4 | Coordinate math, array ops, sin/cos encoding | Coordinate flip and normalization are pure array ops; faster in NumPy than pandas for numerical transforms |
| scipy | 1.14.x | Linear interpolation of missing tracking frames | `scipy.interpolate.interp1d` is more flexible than pandas `.interpolate()` for variable-rate gaps and bounds control |
| pytest | 8.x | Unit tests for normalization, split assertion, leakage check | Silent failures are the primary risk; test coverage on preprocessing catches them before training |
| matplotlib | 3.10.8 | 50-play overlay visualization for coordinate validation | Required for the phase gate visualization |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyarrow | latest | Parquet read/write via pandas | Use `df.to_parquet()` and `pd.read_parquet()`; faster than CSV for repeated loads of processed data |
| tqdm | 4.x | Progress bar on CSV concatenation and per-play interpolation | Weekly tracking CSVs are large; progress visibility helps during development |
| jupyter / jupyterlab | 4.x | Interactive EDA and visualization during development | Use for sanity-checking coordinate plots interactively before writing tests |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `scipy.interpolate.interp1d` | `pandas.DataFrame.interpolate(method='linear')` | pandas interpolate is simpler for basic gaps but doesn't easily expose interpolated-frame flags or bounds control; scipy is more explicit |
| Parquet intermediate cache | CSV intermediate cache | Parquet is 3-5x faster to load on repeated runs and preserves dtypes; CSV is universal but slower for large tracking datasets |
| Temporal (week-based) split | Game-ID-level random split | Both are game-disjoint but temporal split is strictly more valid for research: earlier weeks train, later weeks test — mirrors real deployment |

**Installation:**
```bash
pip install "pandas==3.0.1" "numpy==2.4" "scipy>=1.14" "pytest>=8.0" "matplotlib==3.10.8" "tqdm>=4.0" pyarrow jupyterlab
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 1 scope)

```
data/
├── raw/                         # Extracted from zip (gitignored — not committed)
│   ├── train/
│   │   ├── tracking_week_1.csv
│   │   ├── tracking_week_2.csv  # ... through week N
│   │   ├── plays.csv
│   │   ├── players.csv
│   │   └── games.csv
│   └── test/
│       └── tracking_*.csv       # Competition test set (no labels)
└── processed/
    ├── cleaned.parquet          # Output of preprocessor.py
    └── splits.json              # Saved {train_game_ids, val_game_ids, test_game_ids}

src/
└── data/
    ├── loader.py                # CSV ingestion + join
    ├── preprocessor.py          # Normalization, flip, interpolation, acceleration
    └── sample_builder.py        # (player, play) → fixed-length tensor

scripts/
└── preprocess.py                # End-to-end pipeline runner

notebooks/
└── validate_pipeline.ipynb      # 50-play overlay + distribution checks

tests/
├── test_loader.py
├── test_preprocessor.py
└── test_splits.py
```

### Pattern 1: Zip Extraction Before Any Loading

**What:** Extract the zip to `data/raw/` as a one-time setup step, verify the folder structure, then proceed with CSV loading. Never load directly from the zip.

**When to use:** Always. Reading CSVs from inside a zip is possible but adds complexity and makes interactive debugging harder.

**Example:**
```python
# In loader.py or a setup script
import zipfile
from pathlib import Path

def extract_dataset(zip_path: Path, dest_dir: Path) -> None:
    """Extract competition zip to data/raw/. Idempotent."""
    if (dest_dir / "train").exists():
        print("Already extracted — skipping.")
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"Extracted to {dest_dir}")
    # Verify expected structure
    assert (dest_dir / "train" / "plays.csv").exists(), "plays.csv not found in train/"
```

### Pattern 2: CSV Load and Join

**What:** Load all weekly tracking files, concatenate, then join with plays/players/games on the correct composite keys.

**When to use:** Standard for NFL Big Data Bowl data.

**Example:**
```python
# In loader.py
import pandas as pd
from pathlib import Path

def load_raw(data_dir: Path) -> pd.DataFrame:
    train_dir = data_dir / "train"

    # Concatenate all weekly tracking files
    tracking_files = sorted(train_dir.glob("tracking_week_*.csv"))
    tracking = pd.concat([pd.read_csv(f) for f in tracking_files], ignore_index=True)

    plays    = pd.read_csv(train_dir / "plays.csv")
    players  = pd.read_csv(train_dir / "players.csv")
    games    = pd.read_csv(train_dir / "games.csv")

    # Join on composite keys — assert row count unchanged
    n_before = len(tracking)
    merged = (tracking
              .merge(plays,   on=["gameId", "playId"], how="left")
              .merge(players, on="nflId",              how="left")
              .merge(games,   on="gameId",             how="left"))
    assert len(merged) == n_before, f"Row count changed after merge: {n_before} → {len(merged)}"
    return merged
```

### Pattern 3: LOS-Relative Coordinate Normalization + Direction Flip

**What:** Normalize all (x, y) to be relative to the line of scrimmage, then flip plays going left so all offense moves in +x direction. Both operations must happen in the same pass.

**When to use:** Every play, without exception. This is the canonical orientation the model learns.

**Critical gotcha — `absoluteYardlineNumber` offset:** The `absoluteYardlineNumber` column in `plays.csv` is reported on a 1–50 scale that does NOT include the 10-yard end zones. The actual yard line on the 120-yard field is `absoluteYardlineNumber + 10`. Failing to add this offset shifts all player coordinates by 10 yards.

**Example:**
```python
# In preprocessor.py
import numpy as np

def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Compute true field position of LOS (0-120 yard scale)
    # absoluteYardlineNumber is 1-50 (field half); add 10 for end-zone offset
    # If yardlineSide == defensiveTeam, LOS is on defensive side → 100 - yardlineNumber + 10
    # Simplest safe approach: use the pre-computed absoluteYardlineNumber from the possession team's perspective
    df["los_x"] = df["absoluteYardlineNumber"] + 10  # end-zone offset

    # Flip plays going left so offense always moves in +x direction
    left_mask = df["playDirection"] == "left"
    df.loc[left_mask, "x"] = 120 - df.loc[left_mask, "x"]
    df.loc[left_mask, "y"] = 53.3 - df.loc[left_mask, "y"]  # mirror y if needed

    # For left-direction plays, the LOS is also mirrored
    df.loc[left_mask, "los_x"] = 120 - df.loc[left_mask, "los_x"]

    # Subtract LOS x to make coordinates relative to line of scrimmage
    df["x"] = df["x"] - df["los_x"]
    # y is already lateral; optionally center on field midpoint (26.65)
    df["y"] = df["y"] - 26.65  # center field laterally

    return df


def encode_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Replace dir and o degree columns with sin/cos pairs."""
    df = df.copy()
    for col in ["dir", "o"]:
        rad = np.radians(df[col])
        df[f"{col}_sin"] = np.sin(rad)
        df[f"{col}_cos"] = np.cos(rad)
    df = df.drop(columns=["dir", "o"])

    # Also flip angle encoding for left-direction plays BEFORE sin/cos encoding
    # Note: flip must happen BEFORE this call; preprocessor must flip raw dir first
    return df
```

**Important:** The direction flip on angles (`dir`, `o`) must add 180 degrees (mod 360) to the raw degree values BEFORE encoding as sin/cos. Do this in a single ordered step: flip coordinates → flip angles → encode angles as sin/cos.

### Pattern 4: Missing Frame Interpolation with Flag

**What:** For each (nflId, gameId, playId) group, detect missing frame indices, linearly interpolate x, y, speed, and sin/cos-encoded direction, and mark interpolated frames with a boolean flag.

**When to use:** Always before computing acceleration; interpolated frames must be excluded from acceleration calculation.

**Example:**
```python
# In preprocessor.py
from scipy.interpolate import interp1d
import numpy as np

def interpolate_group(group: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Interpolate missing frames for one (nflId, gameId, playId) group."""
    group = group.sort_values("frameId").copy()
    all_frames = np.arange(group["frameId"].min(), group["frameId"].max() + 1)
    present_frames = group["frameId"].values

    # Detect gaps
    missing = set(all_frames) - set(present_frames)
    if not missing:
        group["is_interpolated"] = False
        return group

    # Check for gaps > max_gap consecutive — flag entire group if found
    gaps = np.diff(np.sort(list(missing)))
    if any(g > max_gap for g in gaps):
        group["too_many_missing"] = True  # caller should flag/drop this group
        group["is_interpolated"] = False
        return group

    # Reindex to full frame range and interpolate
    full_index = pd.Index(all_frames, name="frameId")
    group = group.set_index("frameId").reindex(full_index).reset_index()
    group["is_interpolated"] = group["x"].isna()

    for col in ["x", "y", "s", "dir_sin", "dir_cos"]:
        known = group[col].notna()
        if known.sum() >= 2:
            interp_fn = interp1d(
                group.loc[known, "frameId"],
                group.loc[known, col],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )
            group.loc[~known, col] = interp_fn(group.loc[~known, "frameId"])

    return group
```

### Pattern 5: Temporal Train/Val/Test Split

**What:** Assign games to train/val/test splits based on the week they occurred. Use earlier weeks for training, later weeks for validation and test. Save the split to disk before any normalization statistics are computed.

**When to use:** Always. Never use random play-level splits.

**Example:**
```python
# In preprocessor.py or a dedicated split.py
import json
from pathlib import Path

def make_temporal_split(
    games: pd.DataFrame,
    val_weeks: list[int],
    test_weeks: list[int],
    output_path: Path
) -> dict:
    train_games = set(games.loc[~games["week"].isin(val_weeks + test_weeks), "gameId"])
    val_games   = set(games.loc[games["week"].isin(val_weeks),               "gameId"])
    test_games  = set(games.loc[games["week"].isin(test_weeks),              "gameId"])

    # Critical assertion: zero overlap
    assert len(train_games & test_games) == 0, "Train/test game ID overlap!"
    assert len(train_games & val_games) == 0,  "Train/val game ID overlap!"
    assert len(val_games & test_games) == 0,   "Val/test game ID overlap!"

    split = {
        "train_game_ids": sorted(train_games),
        "val_game_ids":   sorted(val_games),
        "test_game_ids":  sorted(test_games),
    }
    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)
    return split
```

### Pattern 6: Acceleration Computation (Excluding Interpolated Frames)

**What:** Compute acceleration as `(speed_t - speed_{t-1}) / dt` within each (player, play) group. Set acceleration to NaN on frames that were interpolated, then fill with per-play mean acceleration.

**When to use:** When the `a` column in raw tracking data is missing or unreliable. The 2026 competition data may or may not include it — check CSV headers first.

**Example:**
```python
def compute_acceleration(df: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
    df = df.sort_values(["gameId", "playId", "nflId", "frameId"]).copy()
    df["a_computed"] = (
        df.groupby(["gameId", "playId", "nflId"])["s"]
          .diff() * fps  # yards/frame → yards/s^2 approximation
    )
    # Null out acceleration on interpolated frames
    df.loc[df["is_interpolated"], "a_computed"] = np.nan
    # Fill with per-play mean
    df["a_computed"] = df.groupby(["gameId", "playId", "nflId"])["a_computed"].transform(
        lambda x: x.fillna(x.mean())
    )
    return df
```

### Anti-Patterns to Avoid

- **Normalizing to ball landing location:** Normalize to LOS (snap location), never to where the ball ends up. Normalizing to the ball's final position bakes the ablation variable into every player coordinate.
- **Direction flip after angle encoding:** Always flip raw degree values before calling sin/cos encoding. Flipping sin/cos values numerically requires different logic (negate sin, keep cos for 180° flip) and is error-prone.
- **Skipping the `absoluteYardlineNumber + 10` offset:** The 10-yard end-zone offset is a consistent gotcha across all NFL Big Data Bowl years. Build it in from the start.
- **Random `.sample()` split on plays DataFrame:** One line of code that silently destroys research validity. Always split on `gameId` grouped by week.
- **Computing acceleration on the full DataFrame before flagging interpolated frames:** The fill step must come after interpolation flags are set, not before.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ZIP extraction | Custom archive reader | `python zipfile.ZipFile` (stdlib) | Handles nested paths, unicode filenames, and partial extraction cleanly |
| Linear interpolation | Custom gap-filler loop | `scipy.interpolate.interp1d` | Handles non-uniform frame spacing, bounds control, and multiple fill modes |
| Parquet I/O | Custom binary format | `pandas.to_parquet` / `pd.read_parquet` via pyarrow | Preserves dtypes (including bool `is_interpolated`), 3-5x faster than CSV re-load |
| Temporal split | Manual week-to-set assignment | `games.csv` week column + `groupby` | Week assignments are authoritative in the dataset; don't derive them |
| Sin/cos angle encoding | Custom trigonometric encoding | `np.sin(np.radians(col))` | Numpy vectorized; handles NaN propagation correctly |

**Key insight:** The most dangerous thing to hand-roll in this phase is anything involving the ablation boundary — specifically, ball landing location extraction. This must be computed from the dataset's final frame (ball row at `max(frameId)` per play) and stored as a separate column, never mixed into the normalized coordinate base features.

---

## Common Pitfalls

### Pitfall 1: `absoluteYardlineNumber` End-Zone Offset

**What goes wrong:** `absoluteYardlineNumber` in `plays.csv` runs from 1 to 50 and does not include end zones. The actual field is 120 yards (100 yards of playing field + two 10-yard end zones). Subtracting the raw `absoluteYardlineNumber` from player `x` coordinates shifts all positions by 10 yards, producing a systematic bias in every normalized coordinate.

**Why it happens:** The column name sounds like an absolute yard line on the full field, but it is relative to the nearest end zone on the 100-yard playing field only.

**How to avoid:** Always compute `los_field_x = absoluteYardlineNumber + 10` before using it as the LOS reference. Validate: a snap at the 50-yard line should produce `los_field_x = 60`, and after normalization the offensive linemen should cluster near `x = 0`.

**Warning signs:** After normalization, the distribution of starting player x-positions shows a consistent 10-yard offset from zero for snap plays near midfield.

### Pitfall 2: Play Direction Flip Applied Only to Coordinates, Not Angles

**What goes wrong:** The `x` coordinate is mirrored (`120 - x`) for left-direction plays, but `dir` and `o` (orientation in degrees) are not adjusted. A player facing "north" on a left-to-right play and a player facing "south" on a right-to-left play are the same physical facing direction, but the raw degrees are different. After the flip, the spatial coordinates are canonical but the angles still encode the original field orientation.

**Why it happens:** Coordinate flipping is visually obvious when you plot the field. Angle flipping is not visible unless you specifically check compass rose distributions.

**How to avoid:** After mirroring x for left-direction plays, add 180 degrees to both `dir` and `o` (mod 360) before computing sin/cos. Validate: after encoding, the distribution of `dir_cos` should be symmetric around zero (not shifted), and the mean of `dir_sin` for all plays should be near zero.

**Warning signs:** `dir_cos` distribution is bimodal with peaks at +1 and -1 instead of a smooth distribution. Players appear to "face backward" in overlay visualizations.

### Pitfall 3: Missing Frame Gap Detection Before Interpolation

**What goes wrong:** `scipy.interp1d` or `pandas.interpolate` fills ALL gaps regardless of size. A sequence with 10 consecutive missing frames gets interpolated to appear smooth, but the interpolated values are unreliable. This passes silently — no error is raised.

**Why it happens:** The interpolation call does not know how large a gap is acceptable. The caller must detect gap sizes before interpolating.

**How to avoid:** Before interpolating, compute run-length encoding of the missing frames. For each run of consecutive missing frames, check if its length exceeds `max_gap` (3 consecutive frames). If it does, mark the entire sequence with `too_many_missing = True` and exclude it from the training set. Validate: log the count of flagged sequences per week.

**Warning signs:** Some plays have suspiciously smooth velocity or position transitions across many frames. The `is_interpolated` flag distribution shows sequences with long runs of `True`.

### Pitfall 4: Ball Landing Location Column Name Unknown

**What goes wrong:** The exact column name(s) in `plays.csv` that encode where the ball lands varies by NFL Big Data Bowl competition year. Common patterns: `targetX`/`targetY`, derived from ball tracking rows at the final frame, or implicit in `passResult` + `offensePlayResult`. Using the wrong column produces incorrect labels for Model B and invalidates the ablation.

**Why it happens:** The column names are not standardized across competition years. Research from earlier years (2019-2024) may use different names.

**How to avoid:** Immediately upon extracting the zip, print `plays.csv` column names and compare against the data dictionary provided with the 2026 competition. Do NOT hard-code a column name from a different year. If ball landing coordinates are not in `plays.csv`, derive them from ball tracking rows (`nflId` corresponding to "football" at `max(frameId)` per play).

**Warning signs (flag for user):** The `plays.csv` headers must be inspected before any preprocessing code is finalized. This is an explicit pre-implementation check.

### Pitfall 5: DATA-03 Violation — Code Not Committed Before Dataset Upload

**What goes wrong:** The user uploads the dataset before the codebase is committed to GitHub, meaning the pipeline code exists only locally. If anything goes wrong, there is no clean starting point.

**Why it happens:** The natural workflow is "get data, write code" — but DATA-03 explicitly reverses this.

**How to avoid:** The wave plan for this phase must place all code scaffolding, `loader.py`, `preprocessor.py`, `scripts/preprocess.py`, and test stubs in Wave 0 (committed to GitHub first). The data upload instruction must appear as a separate wave that runs only after the commit wave is complete.

---

## Code Examples

Verified patterns from official sources:

### Zip Extraction (stdlib)
```python
# Source: Python 3.11 stdlib zipfile docs
import zipfile
from pathlib import Path

zip_path = Path("nfl-big-data-bowl-2026-prediction.zip")
dest = Path("data/raw")
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(dest)
```

### Pandas 3.0 Safe Copy-on-Write Pattern
```python
# Source: pandas 3.0 migration guide — CoW enabled by default
# CORRECT: always use .copy() before mutation, or assign via loc
df = df.copy()
df.loc[mask, "x"] = 120 - df.loc[mask, "x"]

# WRONG in pandas 3.0 (raises ChainedAssignmentError):
# df[mask]["x"] = 120 - df[mask]["x"]
```

### Parquet Save and Load
```python
# Source: pandas 3.0 docs — requires pyarrow
df.to_parquet("data/processed/cleaned.parquet", index=False)
df = pd.read_parquet("data/processed/cleaned.parquet")
```

### 50-Play Overlay Validation Plot
```python
# Visualization for phase gate: all offense should move in +x direction
import matplotlib.pyplot as plt
import random

fig, ax = plt.subplots(figsize=(12, 5))
sample_play_ids = random.sample(list(df["playId"].unique()), 50)
for play_id in sample_play_ids:
    play = df[(df["playId"] == play_id) & (df["position"].isin(["QB", "WR", "TE", "RB"]))]
    for _, player in play.groupby("nflId"):
        ax.plot(player["x"], player["y"], alpha=0.2, linewidth=0.8, color="blue")
ax.axvline(0, color="red", linewidth=1, linestyle="--", label="LOS")
ax.set_xlabel("Yards from LOS (+x = offense direction)")
ax.set_ylabel("Lateral position (yards from field center)")
ax.set_title("50-Play Overlay: Offense should all move in +x direction")
plt.tight_layout()
plt.savefig("outputs/validation/50_play_overlay.png", dpi=150)
```

### Split Disjointness Assertion
```python
# This assertion must pass before phase is complete
train_ids = set(splits["train_game_ids"])
val_ids   = set(splits["val_game_ids"])
test_ids  = set(splits["test_game_ids"])
assert len(train_ids & test_ids) == 0, "Train/test overlap!"
assert len(train_ids & val_ids)  == 0, "Train/val overlap!"
assert len(val_ids  & test_ids)  == 0, "Val/test overlap!"
print(f"Split sizes — train: {len(train_ids)} games, val: {len(val_ids)}, test: {len(test_ids)}")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Load CSV per epoch | Pre-process to parquet once, load parquet in training | Standard practice for datasets > 1GB | 3-5x faster training startup; avoids CPU bottleneck |
| `df[mask]["col"] = val` in pandas | `df.loc[mask, "col"] = val` or `df.copy()` first | pandas 3.0 (Feb 2026) — CoW default | ChainedAssignmentError will raise in pandas 3.0 without CoW-safe patterns |
| `pandas.interpolate()` for tracking gaps | `scipy.interpolate.interp1d` with explicit frame index | No version change — scipy always preferred for variable-rate gaps | More control over bounds, gap detection, and frame index handling |
| Include `a` column as-is from CSV | Verify `a` column exists; compute from speed delta if absent or unreliable | Varies by competition year | Avoids silent zero-acceleration artifact on interpolated frames |

**Deprecated/outdated:**
- `df.interpolate(method='linear')` applied blindly: does not track which frames were interpolated, making downstream acceleration computation unreliable.
- Random `train_test_split(plays_df)`: This is the correct scikit-learn API but used on the wrong unit — must split on `gameId` groups, not individual rows.

---

## NFL Big Data Bowl 2026 Schema Notes

**MEDIUM confidence — column names must be verified against actual zip contents before implementing `preprocessor.py` and `sample_builder.py`.**

### Expected Files (based on 2024/2025 competition patterns)

| File | Expected Key Columns | Notes |
|------|---------------------|-------|
| `train/tracking_week_N.csv` | `gameId, playId, nflId, frameId, time, x, y, s, a, dis, o, dir, event` | Multiple files (one per week); concatenate all before processing |
| `train/plays.csv` | `gameId, playId, possessionTeam, defensiveTeam, absoluteYardlineNumber, playDirection, passResult, offensePlayResult` | Ball landing columns: **must be verified** — may be `targetX`/`targetY`, or must be derived from ball tracking rows |
| `train/players.csv` | `nflId, displayName, position, height, weight` | Filter by `position in {"CB", "FS", "SS", "LB"}` |
| `train/games.csv` | `gameId, season, week, homeTeamAbbr, visitorTeamAbbr` | Used for temporal split by week |

### Pre-Implementation Schema Check (required before writing preprocessor.py)

```python
import pandas as pd
from pathlib import Path

plays = pd.read_csv(Path("data/raw/train/plays.csv"))
print("plays.csv columns:")
print(plays.columns.tolist())
print("\nSample row:")
print(plays.iloc[0])
```

Run this check immediately after extracting the zip. Document the output in the implementation notes so column names are locked before any preprocessing code is written.

### Ball Landing Location Derivation

Two possible sources (check both):

1. **Direct columns in `plays.csv`:** Look for `targetX`, `targetY`, `passEndZone`, or similar.
2. **Derived from ball tracking:** Filter tracking rows where `displayName == "football"` (or `nflId` is the ball entity), get the row at `max(frameId)` per play, extract `x` and `y`. These will already be in the raw coordinate system; normalize them using the same LOS offset and direction flip as all other coordinates.

---

## Open Questions

1. **Ball landing location column name in `plays.csv`**
   - What we know: Competition years 2019-2024 have varied column naming; 2021 used `targetX`/`targetY`; other years may derive from ball tracking rows
   - What's unclear: Whether the 2026 competition includes a direct `targetX`/`targetY` column or requires deriving from the ball's final tracking position
   - Recommendation: User must run the schema check above immediately after extracting the zip, before any code that references ball landing coordinates is finalized

2. **Whether `a` (acceleration) column is present and reliable in 2026 data**
   - What we know: Older NFL BDB competitions included `a` column but it had quality issues (spikes, zeros during interpolated frames)
   - What's unclear: Whether the 2026 dataset includes a clean `a` column or requires computing from speed delta
   - Recommendation: Check for `a` column presence; if present, validate its distribution (should not have implausibly large spikes); compute from speed delta as a quality check regardless

3. **Total play count and per-position sample distribution**
   - What we know: Single-season NFL datasets typically contain 8,000–15,000 pass plays; filtered to CB/FS/SS/LB, each play produces 4-6 target player samples
   - What's unclear: Whether the 2026 competition uses a full season or a subset; whether the test set has labels
   - Recommendation: After loading, print position value counts and per-week play counts to confirm statistical power before committing to the experimental design

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pytest.ini` or `pyproject.toml [tool.pytest]` — see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v --tb=short` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Zip extracts to `data/raw/train/` and `data/raw/test/` with expected files | unit | `pytest tests/test_loader.py::test_extract_zip -x` | ❌ Wave 0 |
| DATA-02 | `load_raw()` returns merged DataFrame with correct shape and expected columns | unit | `pytest tests/test_loader.py::test_load_raw_columns -x` | ❌ Wave 0 |
| DATA-03 | Manual verification — code committed before dataset uploaded | manual | n/a | n/a |
| PREP-01 | After normalization, offensive linemen cluster near `x = 0` (LOS) | unit | `pytest tests/test_preprocessor.py::test_los_normalization -x` | ❌ Wave 0 |
| PREP-02 | After flip, all offensive players have positive mean x displacement | unit | `pytest tests/test_preprocessor.py::test_direction_flip -x` | ❌ Wave 0 |
| PREP-03 | `dir_sin` and `dir_cos` columns present; no raw `dir` or `o` columns | unit | `pytest tests/test_preprocessor.py::test_angle_encoding -x` | ❌ Wave 0 |
| PREP-04 | `is_interpolated` flag present; sequences with >3 consecutive missing frames flagged | unit | `pytest tests/test_preprocessor.py::test_interpolation_flag -x` | ❌ Wave 0 |
| PREP-05 | `a` column (or `a_computed`) is non-null for all non-interpolated frames | unit | `pytest tests/test_preprocessor.py::test_acceleration_nonnull -x` | ❌ Wave 0 |
| PREP-06 | `set(train_game_ids) & set(test_game_ids) == set()` assertion passes | unit | `pytest tests/test_splits.py::test_split_disjoint -x` | ❌ Wave 0 |

### Visual Validation (manual gate)

| Gate | Command | Pass Criteria |
|------|---------|---------------|
| 50-play overlay | `python scripts/validate_normalization.py` | All offensive motion in +x direction; no mirrored plays visible |
| Position distribution | `python scripts/validate_normalization.py --show-positions` | CB, FS, SS, LB counts printed; no position with 0 samples |

### Sampling Rate

- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v --tb=short`
- **Phase gate:** Full suite green + 50-play overlay visual confirmation before Phase 2

### Wave 0 Gaps

- [ ] `tests/test_loader.py` — covers DATA-01, DATA-02
- [ ] `tests/test_preprocessor.py` — covers PREP-01 through PREP-05
- [ ] `tests/test_splits.py` — covers PREP-06
- [ ] `pytest.ini` — framework config with `testpaths = tests`
- [ ] Framework install: `pip install pytest>=8.0` — if not in requirements.txt

---

## Sources

### Primary (HIGH confidence)
- `.planning/research/STACK.md` — verified library versions, NFL BDB CSV schema patterns, pandas 3.0 CoW behavior
- `.planning/research/PITFALLS.md` — coordinate normalization, split strategy, interpolation, and leakage pitfalls
- `.planning/research/ARCHITECTURE.md` — component responsibilities, data flow, project structure, key data shapes
- `.planning/research/SUMMARY.md` — executive summary, feature scope, phase rationale
- `.planning/REQUIREMENTS.md` — authoritative requirement IDs and descriptions for DATA-01 through PREP-06
- Python stdlib `zipfile` docs — ZIP extraction pattern
- pandas 3.0 documentation — Copy-on-Write behavior, `.loc` mutation pattern
- NumPy 2.4 documentation — `np.sin`, `np.radians` vectorized ops
- scipy 1.14 documentation — `scipy.interpolate.interp1d` API

### Secondary (MEDIUM confidence)
- NFL Big Data Bowl 2024/2025 competition Kaggle notebooks — CSV schema patterns, `absoluteYardlineNumber` end-zone offset, play direction flip conventions, ball position tracking as "football" row

### Tertiary (LOW confidence — needs validation)
- `plays.csv` ball landing column names for the 2026 competition specifically — must be verified against actual zip contents before implementation
- Whether 2026 `a` (acceleration) column is clean vs. requiring recomputation from speed delta — verify against actual data distribution

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified in STACK.md against official docs; pandas/NumPy/scipy/pytest are mature stable APIs
- Architecture: HIGH — data flow and component structure are well-established for this class of problem; based on both project research files and well-known NFL BDB preprocessing conventions
- Pitfalls: HIGH — all five major pitfalls documented with root causes, prevention patterns, and warning signs; the `absoluteYardlineNumber + 10` offset and direction-angle flip order are verified gotchas
- NFL BDB 2026 column names: MEDIUM — ball landing location column name is the only significant unknown; everything else is verifiable from CSV headers after extraction
- Validation tests: HIGH — test structure maps directly to requirements; all are simple unit-testable assertions on DataFrame properties

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (stable domain — column name check is the only time-sensitive item, and it is resolved at data extraction time)
