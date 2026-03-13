---
phase: 01-data-pipeline-and-validation
plan: 04
subsystem: data
tags: [pandas, numpy, parquet, json, zipfile, matplotlib, tqdm, pytest, bdb2026]

# Dependency graph
requires:
  - phase: 01-data-pipeline-and-validation/01-02
    provides: loader.py, preprocessor.py, sample_builder.py, run_pipeline.py, validate_normalization.py

provides:
  - data/processed/cleaned.parquet: 4,880,579 rows x 30 cols, normalized BDB 2026 tracking data
  - data/processed/splits.json: disjoint train(208)/val(32)/test(32) game ID sets by week
  - outputs/validation/50_play_overlay.png: visual proof of LOS-relative normalization (815KB)
  - loader.py (updated): BDB 2026 input_2023_w*.csv reader with week derivation from filename
  - preprocessor.py (updated): normalizes ball_land_x/ball_land_y alongside player coords
  - sample_builder.py (updated): reads ball_target_xy from ball_land_x/ball_land_y columns directly
  - run_pipeline.py (updated): handles new BDB 2026 format, builds games_df from loaded data
  - tests/test_pipeline.py: test_zip_extraction activated (was skipped), all 8 tests pass

affects:
  - 02 (model training — consumes cleaned.parquet, splits.json, and sample tensors)
  - Phase 2 feature engineering (ball_land_x/ball_land_y confirmed available per row)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - BDB 2026 format: input_2023_w*.csv per week (not tracking_week_*.csv + plays/games/players)
    - Week derived from filename regex: re.search(r'_w(\d+)\.csv$', path.name)
    - ball_land_x/ball_land_y normalized using same LOS-relative + center-y transform as player coords
    - Column rename map in loader: snake_case -> camelCase for preprocessor compatibility
    - games_df for temporal split built from (gameId, week) in loaded DataFrame (no games.csv)

key-files:
  created:
    - data/processed/cleaned.parquet
    - data/processed/splits.json
    - outputs/validation/50_play_overlay.png
  modified:
    - src/data/loader.py (BDB 2026 input format, column rename map, week from filename)
    - src/data/preprocessor.py (ball_land_x/y normalization in normalize_coordinates)
    - src/data/sample_builder.py (ball_target_xy from direct columns; documentation comment)
    - scripts/run_pipeline.py (BDB 2026 schema print, games_df from loaded data)
    - scripts/validate_normalization.py (schema mode reads input_2023_w*.csv not plays.csv)
    - tests/test_pipeline.py (test_zip_extraction activated with BDB 2026 structure assertions)
    - pyproject.toml (fixed build-backend: setuptools.build_meta)

key-decisions:
  - "BDB 2026 dataset uses input_2023_w*.csv format — no separate plays/games/players CSVs. Everything embedded per row including ball_land_x, ball_land_y, player_position, absolute_yardline_number."
  - "ball_land_x and ball_land_y are in raw NFL field coordinates and need the same LOS-relative normalization as player x/y — confirmed by range analysis (ball_land_y mean=26.2, raw field center=26.65)."
  - "Week number derived from filename (input_2023_w01.csv -> week=1) not from data — no week column in raw CSVs."
  - "pyproject.toml build-backend fixed from setuptools.backends.legacy:build to setuptools.build_meta — setuptools v82 removed the legacy module path."
  - "Column rename map (snake_case -> camelCase) applied in loader for preprocessor compatibility — avoids touching preprocessor/test column references."
  - "test_zip_extraction updated for BDB 2026: checks input_2023_w01.csv and output_2023_w01.csv instead of plays.csv."

patterns-established:
  - "BDB 2026 loader: load week files -> rename columns -> add week from filename -> concat"
  - "Ball landing coords: read ball_land_x/ball_land_y directly; normalize with same transform as player x/y"
  - "Temporal split: build games_df from (gameId, week).drop_duplicates() from loaded df"

requirements-completed: [DATA-01, DATA-02, PREP-01, PREP-02, PREP-03, PREP-04, PREP-05, PREP-06]

# Metrics
duration: 24min
completed: 2026-03-13
---

# Phase 1 Plan 04: Pipeline Execution Summary

**BDB 2026 pipeline adapted from synthetic-fixture design to actual competition format (input_2023_w*.csv with embedded ball_land_x/y), producing 4.88M-row cleaned.parquet, disjoint splits, and 50-play overlay; all 8 tests pass**

## Performance

- **Duration:** 24 min
- **Started:** 2026-03-13T23:03:33Z
- **Completed:** 2026-03-13T23:27:33Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Identified and adapted to BDB 2026 competition format (input_2023_w*.csv, 23 columns per row including ball_land_x/y)
- Pipeline produced 4,880,579 rows x 30 columns in cleaned.parquet with all required columns present
- Splits: train=208 games, val=32 games (weeks 17-18), test=32 games (weeks 1-2), disjoint confirmed
- 50-play overlay PNG generated (815KB), showing offensive trajectories in +x direction post-normalization
- All 8 pytest tests pass including newly activated test_zip_extraction
- Resolved open blocker: ball landing coordinates are `ball_land_x` / `ball_land_y`, provided directly per row

## Task Commits

Each task was committed atomically:

1. **Task 1: Run end-to-end pipeline and inspect schema** - `f733702` (feat)
2. **Task 2: Generate 50-play validation overlay, activate test_zip_extraction, confirm all tests pass** - `bd80223` (feat)

**Plan metadata:** (docs commit — follows this summary)

## Files Created/Modified

- `data/processed/cleaned.parquet` - 4,880,579 rows x 30 cols; normalized BDB 2026 tracking data
- `data/processed/splits.json` - Disjoint game ID sets: train=208, val=32, test=32
- `outputs/validation/50_play_overlay.png` - 815KB overlay of 50 sampled plays showing +x offensive motion
- `src/data/loader.py` - Rewritten for BDB 2026: reads input_2023_w*.csv, renames columns, derives week from filename
- `src/data/preprocessor.py` - Added ball_land_x/y normalization in normalize_coordinates (Step 11)
- `src/data/sample_builder.py` - ball_target_xy from direct ball_land_x/y columns; fallback to football rows; documentation comment added
- `scripts/run_pipeline.py` - Updated schema print, games_df built from loaded DataFrame
- `scripts/validate_normalization.py` - Schema mode reads input_2023_w*.csv not plays.csv
- `tests/test_pipeline.py` - Removed skip from test_zip_extraction; updated assertions for BDB 2026
- `pyproject.toml` - Fixed build-backend for setuptools v82 compatibility

## Decisions Made

- **BDB 2026 format discovery**: The actual dataset uses `input_2023_w*.csv` per week — no separate plays/games/players CSVs. All metadata (position, yardline, play direction, ball landing) is embedded per row. This was unknown during plan 01-02 implementation.
- **ball_land normalization**: Both `ball_land_x` and `ball_land_y` are in raw NFL field coordinates (x: 1-120, y: 0-53.3). Applied same LOS-relative + lateral centering transform. Confirmed by distribution analysis: ball_land_y mean=26.21 (field center=26.65), range -1.69 to 57.33.
- **No separate games.csv**: Temporal split now uses `df[["gameId", "week"]].drop_duplicates()` derived from the loaded DataFrame since week is added from filenames.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] pyproject.toml build-backend incompatible with setuptools v82**
- **Found during:** Task 1 (attempting `pip install -e .`)
- **Issue:** `setuptools.backends.legacy:build` was removed in setuptools v82.0.1 — `ModuleNotFoundError: No module named 'setuptools.backends'`
- **Fix:** Changed `build-backend` to `setuptools.build_meta` (correct modern path)
- **Files modified:** pyproject.toml
- **Verification:** `pip install -e .` succeeded; `from src.data.loader import ...` worked
- **Committed in:** f733702 (Task 1 commit)

**2. [Rule 1 - Bug] loader.py expected tracking_week_*.csv + plays/games/players CSVs — BDB 2026 uses input_2023_w*.csv**
- **Found during:** Task 1 (extraction assertion failed after ZIP extraction)
- **Issue:** `loader.py` asserted `dest_dir / "train" / "plays.csv"` exists after extraction. BDB 2026 has `input_2023_w*.csv` not `tracking_week_*.csv`. No separate plays/games/players files.
- **Fix:** Rewrote `loader.py` completely — new `_load_input_week()` reads input CSVs with column rename map; `load_raw()` globs `input_2023_w*.csv` and derives week from filename; `extract_dataset()` assertion updated.
- **Files modified:** src/data/loader.py
- **Verification:** `load_raw()` loaded 4.88M rows with all required columns; schema check passed
- **Committed in:** f733702 (Task 1 commit)

**3. [Rule 1 - Bug] preprocessor.normalize_coordinates didn't normalize ball_land_x/y**
- **Found during:** Task 1 (pipeline design analysis after discovering new format)
- **Issue:** ball_land_x and ball_land_y are in raw field coordinates (confirmed: x range 1.33-119.78, y range -1.69-57.33, mean=26.21). Without normalization they'd be inconsistent with the normalized player x/y coordinates, corrupting sample tensors.
- **Fix:** Added Step 11 to `normalize_coordinates` — applies same mirror-left, LOS-subtract, center-y transform to ball_land_x/y when columns are present. Guard ensures backward compatibility with fixtures that don't have these columns.
- **Files modified:** src/data/preprocessor.py
- **Verification:** Schema check passed; all 8 tests pass including synthetic-fixture tests
- **Committed in:** f733702 (Task 1 commit)

**4. [Rule 1 - Bug] sample_builder.build_samples() looked for "football" position rows for ball_target_xy — BDB 2026 has no football tracking rows**
- **Found during:** Task 1 (pipeline design — BDB 2026 embeds ball_land_x/y per row)
- **Issue:** Original code searched `df[df["position"] == "football"]` to derive ball landing. BDB 2026 doesn't have football tracking rows — all players are actual players.
- **Fix:** Updated ball_targets lookup to read `ball_land_x`/`ball_land_y` directly when columns present. Preserved football-row fallback for backward compatibility with legacy format.
- **Files modified:** src/data/sample_builder.py
- **Verification:** Sample builder produced 72,400 train / 10,276 val / 11,148 test samples with valid ball_target_xy values
- **Committed in:** f733702 (Task 1 commit)

**5. [Rule 1 - Bug] run_pipeline.py tried to read games.csv for temporal split — no games.csv in BDB 2026**
- **Found during:** Task 1 (pipeline design analysis)
- **Issue:** Step 9 of run_pipeline.py called `pd.read_csv(args.data_dir / "train" / "games.csv")` to get game/week mapping. BDB 2026 has no games.csv.
- **Fix:** Build `games_df` from `df[["gameId", "week"]].drop_duplicates()` — week is already in the loaded DataFrame (added by loader from filename).
- **Files modified:** scripts/run_pipeline.py
- **Verification:** Split built correctly — train=208, val=32, test=32 games; disjoint assertion passed
- **Committed in:** f733702 (Task 1 commit)

**6. [Rule 1 - Bug] test_zip_extraction expected plays.csv — updated for BDB 2026 structure**
- **Found during:** Task 2 (activating test_zip_extraction)
- **Issue:** The plan's proposed test body checked for `train/plays.csv` and `train/games.csv`. BDB 2026 has `input_2023_w01.csv` and `output_2023_w01.csv` instead.
- **Fix:** Updated test assertions to check for `input_2023_w01.csv` and `output_2023_w01.csv`.
- **Files modified:** tests/test_pipeline.py
- **Verification:** test_zip_extraction PASSED (copies zip to tmp_path, extracts, checks correct files)
- **Committed in:** bd80223 (Task 2 commit)

---

**Total deviations:** 6 auto-fixed (4 Rule 1 bugs, 1 Rule 1 schema bug, 1 Rule 3 blocker)
**Impact on plan:** All fixes required for correctness. Root cause: BDB 2026 competition format is fundamentally different from BDB 2024/2025 format that the plan was designed against. The pipeline was correctly abstracted (preprocessor logic unchanged) — only the loader and pipeline orchestration needed updating.

## Issues Encountered

- **BDB 2026 format mismatch**: The dataset ZIP uses `input_2023_w*.csv` / `output_2023_w*.csv` per week (competition prediction format) instead of separate tracking/plays/games/players CSVs. This was the major discovery of this plan — all downstream code had to be adapted. Resolved by rewriting the loader and updating the orchestration scripts.

## User Setup Required

None — pipeline runs fully locally. Dataset ZIP was already confirmed in project root from plan 01-03.

## Next Phase Readiness

- cleaned.parquet: 4,880,579 rows, all required columns present, 0% null acceleration
- splits.json: disjoint game ID sets, train=208, val=32, test=32 games
- ball_landing coordinates confirmed: `ball_land_x` / `ball_land_y` directly provided and normalized
- 50-play overlay shows correct +x offensive motion post-normalization
- All 8 tests pass; test_zip_extraction fully activated
- Phase gate assertions fully confirmed — ready for Phase 2 (model architecture)
- Position distribution: CB=1.06M, FS=476K, SS=392K, ILB=296K, OLB=207K, MLB=200K rows

## Self-Check: PASSED

- FOUND: data/processed/cleaned.parquet
- FOUND: data/processed/splits.json
- FOUND: outputs/validation/50_play_overlay.png
- FOUND: .planning/phases/01-data-pipeline-and-validation/01-04-SUMMARY.md
- FOUND: commit f733702 (Task 1)
- FOUND: commit bd80223 (Task 2)

---
*Phase: 01-data-pipeline-and-validation*
*Completed: 2026-03-13*
