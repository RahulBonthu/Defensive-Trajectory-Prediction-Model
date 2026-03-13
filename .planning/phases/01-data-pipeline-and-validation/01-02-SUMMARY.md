---
phase: 01-data-pipeline-and-validation
plan: 02
subsystem: data
tags: [pandas, numpy, scipy, parquet, json, zipfile, argparse, tqdm, matplotlib]

# Dependency graph
requires:
  - phase: 01-data-pipeline-and-validation/01-01
    provides: test scaffold with synthetic fixtures and skip-marked stubs

provides:
  - loader.py: ZIP extraction, CSV loading, left-join table merge with row-count assertion
  - preprocessor.py: LOS-relative normalization, angle sin/cos encoding, frame interpolation, acceleration, temporal split
  - sample_builder.py: (player, play) tensor construction with padding_mask and ball_target_xy
  - scripts/run_pipeline.py: end-to-end pipeline orchestration with argparse
  - scripts/validate_normalization.py: 50-play trajectory overlay PNG and schema inspection CLI

affects:
  - 01-03 (feature engineering — builds on cleaned.parquet and sample tensors)
  - 01-04 (pipeline execution — runs run_pipeline.py against real dataset)
  - 02 (model training — consumes sample tensors from build_samples)

# Tech tracking
tech-stack:
  added:
    - scipy.interpolate.interp1d (linear frame interpolation)
    - matplotlib (trajectory visualization, Agg backend)
    - zipfile stdlib (ZIP extraction)
  patterns:
    - pandas 3.0 CoW-safe: df = df.copy() as first line of all mutating functions
    - .loc for all in-place mutations (never chained indexing)
    - Pipeline order enforced: normalize -> encode_angles -> interpolate -> compute_acceleration
    - groupby loop (not .apply) for interpolation to preserve group key columns

key-files:
  created:
    - src/data/loader.py
    - src/data/preprocessor.py
    - src/data/sample_builder.py
    - scripts/run_pipeline.py
    - scripts/validate_normalization.py
  modified:
    - tests/test_pipeline.py (removed all 6 skip marks for preprocessor/sample tests)
    - tests/conftest.py (fixed left-play x values to decrease with frameId)

key-decisions:
  - "Pipeline execution order: normalize_coordinates -> encode_angles -> interpolate_missing_frames -> compute_acceleration. Interpolation operates on dir_sin/dir_cos (not raw degrees) to avoid wrap-around artifacts."
  - "Use explicit groupby loop (not .apply()) in interpolate_missing_frames — pandas groupby drops key columns from apply result; loop preserves gameId/playId/nflId in output."
  - "conftest.py left-play x values use 60.0 - frame_id * 0.5 (decreasing) to simulate realistic NFL motion. Right-play x remains 50.0 + frame_id * 0.5 (increasing). Needed for test_play_direction_flip assertion."
  - "los_x column retained in output of normalize_coordinates for debugging (not dropped after LOS subtraction)."

patterns-established:
  - "CoW-safe mutation: every function starts with df = df.copy()"
  - "Idempotent extraction: check (dest_dir/'train').exists() before unzipping"
  - "Row-count assertion after merge: assert len(merged) == len(tracking)"
  - "Interpolation guard: consecutive run > max_gap sets too_many_missing=True, skips interpolation"

requirements-completed: [DATA-01, DATA-02, PREP-01, PREP-02, PREP-03, PREP-04, PREP-05, PREP-06]

# Metrics
duration: 7min
completed: 2026-03-13
---

# Phase 1 Plan 02: Data Pipeline Implementation Summary

**LOS-relative coordinate normalization, sin/cos angle encoding, frame interpolation, and (player, play) tensor construction from raw NFL tracking CSVs using pandas 3.0 CoW-safe patterns and scipy interp1d**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-13T22:41:01Z
- **Completed:** 2026-03-13T22:48:01Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- All 5 source files implemented and importable: loader.py, preprocessor.py, sample_builder.py, run_pipeline.py, validate_normalization.py
- 7 of 8 tests PASS against synthetic fixtures (test_zip_extraction remains SKIPPED as designed)
- Full preprocessing pipeline: ZIP extract -> merge -> normalize -> encode angles -> interpolate -> compute acceleration -> save parquet -> temporal split -> tensor samples

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement loader.py** - `384422c` (feat)
2. **Task 2: Implement preprocessor.py, sample_builder.py, scripts** - `e663706` (feat)

**Plan metadata:** (docs commit — follows this summary)

## Files Created/Modified

- `src/data/loader.py` - ZIP extraction (idempotent), CSV concat via glob, left-join merge with row-count assertion
- `src/data/preprocessor.py` - Five preprocessing functions: normalize_coordinates, encode_angles, interpolate_missing_frames, compute_acceleration, make_temporal_split
- `src/data/sample_builder.py` - build_samples() producing {frames, padding_mask, target_xy, ball_target_xy, position} dicts for defensive players
- `scripts/run_pipeline.py` - End-to-end orchestration with argparse (--zip-path, --data-dir, --output-dir, --val-weeks, --test-weeks)
- `scripts/validate_normalization.py` - 50-play overlay PNG generator + --show-schema + --show-positions CLI modes
- `tests/test_pipeline.py` - Removed 6 skip marks (test_csv_loading through test_temporal_split_disjoint)
- `tests/conftest.py` - Fixed left-play x values to use decreasing formula for realistic motion

## Decisions Made

- **Pipeline order**: normalize_coordinates -> encode_angles -> interpolate -> compute_acceleration. Encoding before interpolation means we interpolate sin/cos (bounded, no wrap-around) rather than raw degrees.
- **Explicit groupby loop** for interpolation: pandas `.apply()` drops the groupby key columns (gameId, playId, nflId) from the result. Switching to an explicit `for (game_id, play_id, nfl_id), group in df.groupby(...)` loop and re-assigning the keys preserves them.
- **los_x retained**: The LOS field position column is kept in the output for debugging — useful when validating normalization visually.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] conftest.py left-play x values prevent test_play_direction_flip from passing**
- **Found during:** Task 2 (implementing preprocessor.py)
- **Issue:** The fixture used `x = 50.0 + frame_id * 0.5` for ALL players regardless of play direction. For left-directed plays, this means raw x increases with frameId. After the 120-x mirror normalization, x decreases — giving negative displacement. test_play_direction_flip asserts `displacement >= 0` for all offensive players.
- **Fix:** Changed conftest.py fixture to use `x = 60.0 - frame_id * 0.5` for left-directed players. This simulates realistic NFL motion (offense moves toward lower x in left-directed plays). After mirroring, x increases with frameId (displacement positive). Verified test_los_normalization still passes (mirrored snap frame x ~ -24.5, within |x| < 30 bound).
- **Files modified:** tests/conftest.py
- **Verification:** test_play_direction_flip PASSED, test_los_normalization PASSED
- **Committed in:** e663706 (Task 2 commit)

**2. [Rule 1 - Bug] groupby().apply() drops key columns in interpolate_missing_frames**
- **Found during:** Task 2 (testing interpolation)
- **Issue:** After `df.groupby(["gameId", "playId", "nflId"]).apply(...)`, the output DataFrame was missing the gameId, playId, nflId columns (they became the MultiIndex, not regular columns). Tests checking `df["nflId"]` raised KeyError.
- **Fix:** Replaced `.apply()` with an explicit `for (game_id, play_id, nfl_id), group in df.groupby(...)` loop, manually assigning the group keys back to each processed chunk before concatenation.
- **Files modified:** src/data/preprocessor.py
- **Verification:** test_interpolation_and_flagging and test_acceleration_computed both PASSED
- **Committed in:** e663706 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both fixes required for correctness. Fixture fix ensures tests are valid against realistic data. groupby fix ensures processed DataFrames remain usable by downstream pipeline steps.

## Issues Encountered

None beyond the two auto-fixed bugs documented above.

## User Setup Required

None — no external service configuration required. Dataset ZIP will be handled in plan 01-04.

## Next Phase Readiness

- All five source modules importable and tested against synthetic fixtures
- run_pipeline.py ready to execute once NFL BDB dataset ZIP is provided (plan 01-04)
- validate_normalization.py ready to generate overlay PNG once cleaned.parquet exists
- Blocker from STATE.md still applies: exact plays.csv column name for ball landing location (targetX/targetY) must be verified against the actual dataset in plan 01-04

---
*Phase: 01-data-pipeline-and-validation*
*Completed: 2026-03-13*
