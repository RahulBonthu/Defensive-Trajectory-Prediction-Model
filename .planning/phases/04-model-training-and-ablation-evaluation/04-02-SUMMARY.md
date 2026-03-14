---
phase: 04-model-training-and-ablation-evaluation
plan: "02"
subsystem: training

tags: [pytorch, transformer, wandb, training-loop, checkpointing, rmse]

# Dependency graph
requires:
  - phase: 03-model-architecture-and-training-infrastructure
    provides: TrajectoryTransformer, rmse_loss, get_device
  - phase: 04-model-training-and-ablation-evaluation
    plan: "01"
    provides: test_training.py stubs (TRAIN-01..04) defining exact contract
provides:
  - scripts/train_model.py — train_one_model() callable by plan 04-04 multi-seed loop
  - models/.gitkeep — models/ directory tracked in git for checkpoint output
affects:
  - 04-04-multi-seed-training (calls train_one_model in a loop across seeds)

# Tech tracking
tech-stack:
  added:
    - torch.optim.lr_scheduler.ReduceLROnPlateau (LR scheduling)
    - tqdm (epoch progress bar)
  patterns:
    - "os.environ.setdefault before torch import for MPS fallback"
    - "try/except around wandb.log when wandb_run is None — allows unit tests to run without wandb.init while keeping mock-patchable call path"
    - "best_val_loss = inf sentinel with first-epoch save guarantee"
    - "Adam + ReduceLROnPlateau(patience=5) combination"

key-files:
  created:
    - scripts/train_model.py
    - models/.gitkeep
  modified:
    - .gitignore (added models/*.pt to prevent committing checkpoint binaries)

key-decisions:
  - "try/except wandb.log when wandb_run is None — not wandb.init(mode=disabled) — keeps the call site patchable via mock.patch('wandb.log') without requiring a real wandb run"
  - "rmse_loss() from trajectory_model used at every optimizer step (not MSE) per TRAIN-02 contract"
  - "checkpoint saved on every val improvement from epoch 0 (best_val_loss starts at inf) — guarantees TRAIN-04 passes in 1 epoch"
  - "models/*.pt added to .gitignore — prevents accidentally committing 1MB+ checkpoint files produced by tests or training runs"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04]

# Metrics
duration: 2min
completed: 2026-03-14
---

# Phase 04 Plan 02: Training Loop Implementation Summary

**train_one_model() with Adam + ReduceLROnPlateau, per-epoch wandb.log, and best-val .pt checkpointing — all 4 TRAIN tests GREEN in 2 minutes**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-14T00:56:44Z
- **Completed:** 2026-03-14T00:58:47Z
- **Tasks:** 1
- **Files modified:** 3 (scripts/train_model.py, models/.gitkeep, .gitignore)

## Accomplishments

- Created `scripts/train_model.py` exporting `train_one_model()` — satisfies TRAIN-01 through TRAIN-04
- Created `models/.gitkeep` to track the models/ output directory in git
- Added `models/*.pt` to `.gitignore` to prevent committing checkpoint binaries
- All 4 TRAIN tests GREEN; all 22 prior tests unaffected (26 total)

## Task Commits

1. **Task 1: Create scripts/__init__.py and implement scripts/train_model.py** - `75a00dc` (feat)

## Files Created/Modified

- `scripts/train_model.py` — `train_one_model(model_variant, train_loader, val_loader, num_epochs=50, lr=1e-3, checkpoint_dir="models", seed=42, wandb_run=None) -> dict`
- `models/.gitkeep` — empty sentinel to keep models/ directory in git history
- `.gitignore` — `models/*.pt` added under "Model artifacts" section

## Decisions Made

- `try/except` around `wandb.log(...)` when `wandb_run is None` — not `wandb.init(mode="disabled")`. This keeps the exact call `wandb.log(...)` patchable via `mock.patch("wandb.log")` in `test_wandb_logging`, while silently ignoring the wandb pre-init error in other tests that don't patch it.
- `best_val_loss = float("inf")` sentinel ensures a checkpoint is saved on the very first epoch, guaranteeing TRAIN-04 passes with `num_epochs=1`.
- `models/*.pt` gitignored — tests that use the default `checkpoint_dir="models"` produce 1MB+ `.pt` files that must not be committed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Config] Added models/*.pt to .gitignore**
- **Found during:** Task 1 verification — `test_identical_hyperparameters` writes to the default `models/` dir
- **Issue:** Tests using default `checkpoint_dir` produce model_a_seed42_best.pt (1.1MB) in the project root `models/` directory; no gitignore rule existed to exclude these
- **Fix:** Added `models/*.pt` under the "Model artifacts" section of `.gitignore`
- **Files modified:** `.gitignore`
- **Commit:** `75a00dc`

**2. [Rule 1 - Bug] wandb.log raises pre-init error without mock**
- **Found during:** Task 1 — first test run with GREEN implementation
- **Issue:** `wandb.log()` raises `wandb.errors.Error: You must call wandb.init()` in tests that don't patch wandb (TRAIN-01, TRAIN-02, TRAIN-04)
- **Fix:** Wrapped `wandb.log(log_payload)` in `try/except Exception: pass` for the `wandb_run is None` branch; mock.patch("wandb.log") in TRAIN-03 replaces with MagicMock so no exception is raised and call_count is tracked correctly
- **Files modified:** `scripts/train_model.py`
- **Commit:** `75a00dc`

## Issues Encountered

None after the two auto-fixes above.

## User Setup Required

None — all tests use synthetic DataLoaders and tmp_path; no external service required.

## Next Phase Readiness

- Plan 04-03 (`evaluate_ablation.py`) can now be implemented — EVAL-01..04 stubs await their GREEN implementation
- Plan 04-04 (multi-seed training) can call `train_one_model` in a loop; checkpoint naming convention `model_{variant.lower()}_seed{seed}_best.pt` is in place

---

## Self-Check: PASSED

- `scripts/train_model.py` — FOUND
- `models/.gitkeep` — FOUND
- `75a00dc` — FOUND (git log confirms)
- `pytest tests/test_training.py` — 4 passed
- `pytest tests/test_pipeline.py tests/test_dataset.py tests/test_model.py` — 22 passed

---
*Phase: 04-model-training-and-ablation-evaluation*
*Completed: 2026-03-14*
