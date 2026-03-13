---
phase: 02-feature-engineering-and-dataset-wrappers
plan: 01
subsystem: testing
tags: [pytest, numpy, pandas, fixtures, tdd, wave-0]

# Dependency graph
requires:
  - phase: 01-data-pipeline-and-validation
    provides: build_samples() schema (sample dicts with frames/padding_mask/target_xy/ball_target_xy/position)
provides:
  - 6 RED test stubs in tests/test_dataset.py covering FEAT-01 through FEAT-06
  - minimal_samples fixture (6 synthetic sample dicts, no disk I/O)
  - minimal_context_df fixture (480-row synthetic tracking DataFrame with ball_land_x/y)
affects:
  - 02-02 (DefensiveTrajectoryDataset implementation turns these tests GREEN)
  - all subsequent Phase 2 plans (test infrastructure established)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Wave-0 TDD scaffold — failing tests written before implementation exists
    - Synthetic fixtures with hardcoded constants eliminate disk I/O from unit tests
    - RED/GREEN state encoded via ImportError from missing src.data.dataset module

key-files:
  created:
    - tests/test_dataset.py
  modified:
    - tests/conftest.py

key-decisions:
  - "Test stubs use real assertions (not pytest.mark.skip) — skip masks the RED state"
  - "minimal_context_df includes ball_land_x/ball_land_y columns so DefensiveTrajectoryDataset can read them without disk access"
  - "STRICT_POSITIONS constant defined in test file as {'CB', 'FS', 'SS', 'LB'}"

patterns-established:
  - "Wave-0 pattern: write failing tests before any implementation; all 6 tests RED at plan close"
  - "Fixture isolation: new fixtures appended to conftest.py without modifying existing ones"

requirements-completed: [FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06]

# Metrics
duration: 2min
completed: 2026-03-13
---

# Phase 2 Plan 01: Dataset Test Scaffold Summary

**6 Wave-0 test stubs for DefensiveTrajectoryDataset (FEAT-01 through FEAT-06) with synthetic fixtures covering position filtering, padding, social context shape (50), and ball destination leakage (52 vs 50)**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-13T23:54:05Z
- **Completed:** 2026-03-13T23:56:13Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `minimal_samples` fixture (6 sample dicts: 4 CB, 1 FS, 1 SS; T=25 with 10 real frames, no disk I/O)
- Added `minimal_context_df` fixture (480-row synthetic DataFrame with `ball_land_x`/`ball_land_y`; no disk I/O)
- Created `tests/test_dataset.py` with 6 named test stubs, all currently RED (ImportError on `src.data.dataset`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add minimal_samples and minimal_context_df fixtures to conftest.py** - `b2f4e5a` (feat)
2. **Task 2: Write 6 failing test stubs in tests/test_dataset.py** - `f90b251` (test)

## Files Created/Modified

- `tests/test_dataset.py` - 6 test stubs (FEAT-01 through FEAT-06); RED until Plan 02-02 implements DefensiveTrajectoryDataset
- `tests/conftest.py` - `minimal_samples` and `minimal_context_df` fixtures appended; existing 4 fixtures untouched

## Decisions Made

- Test stubs use real assertions, not `pytest.mark.skip` — skip would hide the RED state and break the Nyquist enforcement model
- `minimal_context_df` includes `ball_land_x` / `ball_land_y` columns (15.0 / 3.0) so Plan 02-02 can read them from the DataFrame without needing additional fixture changes
- `STRICT_POSITIONS = {"CB", "FS", "SS", "LB"}` defined as a module-level constant in test file, documenting the filtering contract
- ImportError at collection counts as RED per plan spec — acceptable behavior until `src/data/dataset.py` exists

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Wave-0 scaffold complete; Plan 02-02 can begin implementing `DefensiveTrajectoryDataset` to turn these 6 tests GREEN
- `minimal_samples` and `minimal_context_df` fixtures are fully compatible with the `DefensiveTrajectoryDataset` signature from the plan interfaces
- All 8 Phase 1 pipeline tests remain green (no regression)

## Self-Check

### Files exist:
- tests/test_dataset.py: FOUND
- tests/conftest.py: FOUND (modified)

### Commits exist:
- b2f4e5a: feat(02-01): add minimal_samples and minimal_context_df fixtures to conftest.py
- f90b251: test(02-01): add 6 failing test stubs for FEAT-01 through FEAT-06

## Self-Check: PASSED

---
*Phase: 02-feature-engineering-and-dataset-wrappers*
*Completed: 2026-03-13*
