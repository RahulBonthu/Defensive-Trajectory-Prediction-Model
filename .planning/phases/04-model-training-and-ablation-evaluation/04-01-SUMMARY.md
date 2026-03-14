---
phase: 04-model-training-and-ablation-evaluation
plan: "01"
subsystem: testing

tags: [pytest, tdd, transformer, training, evaluation, ablation]

# Dependency graph
requires:
  - phase: 03-model-architecture-and-training-infrastructure
    provides: TrajectoryTransformer, rmse_loss, get_device — used in test_evaluation.py stubs
provides:
  - 4 RED test stubs defining train_one_model behavioral contract (TRAIN-01..04)
  - 4 RED test stubs defining evaluate_ablation behavioral contract (EVAL-01..04)
affects:
  - 04-02-train-model (must satisfy test_training.py GREEN criteria)
  - 04-03-evaluate-ablation (must satisfy test_evaluation.py GREEN criteria)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_DictDataset inline Dataset class wrapping list-of-dicts for DataLoader compatibility"
    - "Top-level import of unimplemented module to guarantee collection-time ImportError RED state (no pytest.mark.skip)"
    - "unittest.mock.patch for wandb.init / wandb.log in training tests"

key-files:
  created:
    - tests/test_training.py
    - tests/test_evaluation.py
  modified: []

key-decisions:
  - "Top-level imports used (not deferred inside functions) so ImportError fires at collection time — gives clear 4-error RED signal rather than 4 PASSED with skips"
  - "_DictDataset helper class defined in both test files to bridge list-of-dicts API to PyTorch DataLoader"
  - "test_wandb_logging patches wandb.init and wandb.log at module level (not scripts.train_model.wandb) to avoid requiring wandb to be installed in train_model.py's namespace before it exists"

patterns-established:
  - "TDD RED stubs: import fails at collection time with ImportError — plans 04-02 and 04-03 create the modules to turn RED GREEN"
  - "Synthetic _DictDataset pattern: used identically in both test files for consistent DataLoader construction"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, EVAL-01, EVAL-02, EVAL-03, EVAL-04]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 04 Plan 01: RED TDD Stubs for Training and Evaluation Summary

**8 pytest stubs establishing exact function signatures and return-shape contracts for train_one_model and 4 evaluate_ablation functions — all fail with ImportError until plans 04-02 and 04-03 create those modules**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T00:52:39Z
- **Completed:** 2026-03-14T00:54:58Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `tests/test_training.py` with 4 stubs (TRAIN-01 through TRAIN-04) — all fail with ImportError from `scripts.train_model`
- Created `tests/test_evaluation.py` with 4 stubs (EVAL-01 through EVAL-04) — all fail with ImportError from `scripts.evaluate_ablation`
- All 22 existing tests remain GREEN (regression guard confirmed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write RED stubs for test_training.py** - `a56f5a8` (test)
2. **Task 2: Write RED stubs for test_evaluation.py** - `ba8795a` (test)

**Plan metadata:** (docs commit follows)

_Note: Both tasks are TDD RED phase — no GREEN commits in this plan._

## Files Created/Modified

- `tests/test_training.py` — 4 stubs for TRAIN-01..04: identical hyperparameters, positive RMSE loss, wandb logging, checkpoint saved
- `tests/test_evaluation.py` — 4 stubs for EVAL-01..04: per-play RMSE shape, ablation table columns, significance test keys, per-position RMSE keys

## Decisions Made

- Top-level imports used (not deferred inside test functions) so `ImportError` fires at pytest collection time — gives an unambiguous 4-error RED signal that is impossible to accidentally pass
- `_DictDataset` helper class defined in both test files (rather than conftest.py) to keep each file self-contained
- `test_wandb_logging` patches `wandb.init` and `wandb.log` at the top-level namespace rather than at `scripts.train_model.wandb` — the latter doesn't exist yet, so top-level patching is the only approach that works before plan 04-02 exists

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Plan 04-02 (`train_model.py`) can be implemented — test contracts are fully specified
- Plan 04-03 (`evaluate_ablation.py`) can be implemented — all 4 function signatures and return types are locked in
- Exact return shapes: `train_one_model` returns dict with keys `config`, `train_losses`, `val_losses`, `best_val_loss`, `checkpoint_path`; evaluation functions return typed primitives matching the assertions

---
*Phase: 04-model-training-and-ablation-evaluation*
*Completed: 2026-03-14*
