---
phase: 03-model-architecture-and-training-infrastructure
plan: "01"
subsystem: testing
tags: [pytorch, transformer, tdd, red-phase, model]

# Dependency graph
requires:
  - phase: 02-feature-engineering-and-dataset-wrappers
    provides: DefensiveTrajectoryDataset with input_dim=50/52, padding_mask convention (True=real frame)
provides:
  - 8 RED test stubs covering MODEL-01 through MODEL-06 and SC-4 (tests/test_model.py)
  - src/model/__init__.py enabling package import
  - scripts/overfit_test.py with full 100-sample overfit loop for both model variants
affects: [03-02-model-implementation, 03-03-training-loop, phase-4-training]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD RED-phase: deferred import inside each test function so pytest collects all 8 individually"
    - "Overfit script pattern: get_device() CUDA->MPS->CPU, dropout=0.0, assert final_loss < initial_loss * 0.5"
    - "AttentionCapturingEncoderLayer API: capture_attention=True kwarg on TrajectoryTransformer, last_attn_weights on encoder.layers[0]"

key-files:
  created:
    - tests/test_model.py
    - src/model/__init__.py
    - scripts/overfit_test.py
  modified: []

key-decisions:
  - "Deferred import inside each test function (not module-level) so pytest collects 8 tests individually and reports 8 FAILEDs rather than 1 collection ERROR"
  - "test_padding_mask_attention uses capture_attention=True kwarg — Plan 03-02 must add this parameter to TrajectoryTransformer.__init__"
  - "overfit_test.py written in full (not stub) so Plan 03-02 only needs to make the import work"

patterns-established:
  - "TDD RED pattern: import target inside test body so all stubs are individually collectable"

requirements-completed: [MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06]

# Metrics
duration: 2min
completed: 2026-03-14
---

# Phase 3 Plan 01: TrajectoryTransformer TDD RED Stubs Summary

**8 RED test stubs for TrajectoryTransformer (MODEL-01 to MODEL-06 + SC-4), empty src/model package, and full overfit verification script — all failing on import until Plan 03-02**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-14T00:27:03Z
- **Completed:** 2026-03-14T00:29:05Z
- **Tasks:** 1 (TDD RED phase — single atomic task)
- **Files modified:** 3

## Accomplishments

- Created `tests/test_model.py` with exactly 8 RED test stubs, all failing with `ModuleNotFoundError` (subclass of `ImportError`) until `trajectory_model.py` exists
- Created `src/model/__init__.py` as empty package init enabling `src.model` to be importable as a package
- Created `scripts/overfit_test.py` with the complete overfit loop: `get_device()`, 100-sample synthetic dataset for both `input_dim=50` and `input_dim=52`, 200-epoch Adam training, RMSE with eps guard, and `final_loss < initial_loss * 0.5` assertions

## Task Commits

1. **Task 1: RED stubs + model scaffold** - `2ec7dc6` (test)

## Files Created/Modified

- `tests/test_model.py` - 8 RED test stubs for TrajectoryTransformer covering MODEL-01 (×2), MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06, SC-4
- `src/model/__init__.py` - Empty package init for `src.model`
- `scripts/overfit_test.py` - Full 200-epoch overfit verification script for Model A (input_dim=50) and Model B (input_dim=52)

## Decisions Made

- **Deferred import inside test functions:** Module-level `from src.model.trajectory_model import TrajectoryTransformer` caused a single collection-level `ERROR` that prevented pytest from collecting any tests. Moving the import inside each test body (via a `_import_model()` helper) allows pytest to collect all 8 individually and report 8 FAILEDs — matching the expected RED state described in the plan.

- **`capture_attention=True` kwarg pattern:** `test_padding_mask_attention` requires `TrajectoryTransformer(input_dim=50, capture_attention=True)` to get an encoder with `AttentionCapturingEncoderLayer` in position 0. Plan 03-02 must implement this constructor kwarg.

- **Full overfit script written now:** The plan specified writing the complete loop (not a stub) so Plan 03-02 only needs to make `from src.model.trajectory_model import TrajectoryTransformer` succeed for the script to become runnable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Moved import inside test functions to get 8 individual FAILEDs**

- **Found during:** Task 1 (RED stub creation)
- **Issue:** Module-level `from src.model.trajectory_model import TrajectoryTransformer` caused pytest to stop at collection with a single `ERROR` instead of reporting 8 `FAILED` tests. The plan's success criteria require "8 errors (ImportError)" which implies 8 individual test results.
- **Fix:** Wrapped the import in a `_import_model()` helper called inside each test function body, so pytest fully collects all 8 tests before any import is attempted.
- **Files modified:** `tests/test_model.py`
- **Verification:** `pytest tests/test_model.py -v` shows `8 failed` (not 1 error); `pytest tests/test_pipeline.py tests/test_dataset.py -v` shows `14 passed`
- **Committed in:** `2ec7dc6`

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in test structure)
**Impact on plan:** Required for correct RED state as specified. No scope creep.

## Issues Encountered

None beyond the import deferral fix noted above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- RED contract established: 8 tests define the exact API surface for `TrajectoryTransformer`
- Plan 03-02 must create `src/model/trajectory_model.py` with: `TrajectoryTransformer(input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, capture_attention=False)` and `AttentionCapturingEncoderLayer`
- `scripts/overfit_test.py` is ready to run once the import succeeds

---
*Phase: 03-model-architecture-and-training-infrastructure*
*Completed: 2026-03-14*
