---
phase: 04-model-training-and-ablation-evaluation
plan: "03"
subsystem: evaluation

tags: [scipy, pandas, numpy, pytest, tdd, transformer, ablation, statistics]

# Dependency graph
requires:
  - phase: 04-model-training-and-ablation-evaluation
    plan: "01"
    provides: "4 RED EVAL test stubs in tests/test_evaluation.py"
  - phase: 04-model-training-and-ablation-evaluation
    plan: "02"
    provides: "TrajectoryTransformer, get_device — used in collect_per_play_rmse"
provides:
  - "scripts/evaluate_ablation.py with 4 evaluation functions: collect_per_play_rmse, build_ablation_table, run_significance_tests, compute_per_position_rmse"
  - "results/.gitkeep — tracked results/ directory for CSV outputs"
  - "EVAL-01 through EVAL-04 all GREEN (30 total tests passing)"
affects:
  - 04-04-ablation-orchestration (imports all 4 functions; calls save_per_play_csv helper)

# Tech tracking
tech-stack:
  added:
    - "scipy.stats (wilcoxon, ttest_rel) — statistical significance testing"
  patterns:
    - "Per-sample RMSE via ((pred-target)**2).mean(dim=1).sqrt() — NOT rmse_loss() which returns batch mean scalar"
    - "Median-seed selection: np.argsort(seed_means)[len // 2] — representative seed for paired significance tests"
    - "Wilcoxon ValueError guard: all-zero differences caught, returns stat=0.0, p=1.0"
    - "STRICT_POSITIONS = {CB, FS, SS, LB} as module-level constant — guarantees all 4 keys in compute_per_position_rmse output"
    - "results/ directory created on demand via Path('results').mkdir(exist_ok=True) — no manual setup needed"

key-files:
  created:
    - scripts/evaluate_ablation.py
    - results/.gitkeep
  modified: []

key-decisions:
  - "Per-sample RMSE uses mean(dim=1).sqrt() over (batch, 2) pred/target — ensures one float per play, not one per batch"
  - "build_ablation_table edge case: single seed sets std=0.0 and uses index 0 directly (avoids np.argsort on 1-element list)"
  - "compute_per_position_rmse always returns all 4 STRICT_POSITIONS keys — missing positions yield float('nan') so callers never get KeyError"
  - "save_per_play_csv included as helper for plan 04-04 orchestration; not unit-tested in 04-03 scope"
  - "Wilcoxon alternative='greater': H1 = Model A RMSE > Model B RMSE (Model B is better) — matches research hypothesis direction"

patterns-established:
  - "Evaluation functions are pure (no training state) — model.eval() + torch.no_grad() pattern for inference-only passes"
  - "CSV side effects via pathlib.Path('results').mkdir(exist_ok=True) before to_csv() — safe on first run"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04]

# Metrics
duration: 5min
completed: 2026-03-14
---

# Phase 04 Plan 03: Evaluation Functions Summary

**Four pure evaluation utilities (collect_per_play_rmse, build_ablation_table, run_significance_tests, compute_per_position_rmse) with Wilcoxon + paired t-test significance testing via scipy.stats — all 4 EVAL tests GREEN, 30 total tests passing**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-14T01:00:25Z
- **Completed:** 2026-03-14T01:05:00Z
- **Tasks:** 1 (TDD GREEN phase)
- **Files modified:** 2

## Accomplishments

- Created `scripts/evaluate_ablation.py` implementing all 4 evaluation functions exactly matching the behavioral contracts from `tests/test_evaluation.py`
- `collect_per_play_rmse`: uses per-sample `((pred-target)**2).mean(dim=1).sqrt()` formula — not the batch-mean `rmse_loss()` scalar
- `build_ablation_table`: selects median-performing seed for paired significance tests, writes `results/ablation_table.csv` as side effect
- `run_significance_tests`: Wilcoxon (alternative="greater") + paired t-test, guarded against all-zero differences edge case
- `compute_per_position_rmse`: always returns all 4 position keys (CB, FS, SS, LB) — missing positions yield nan rather than KeyError
- Added `save_per_play_csv` helper for the plan 04-04 orchestration script
- Created `results/.gitkeep` to track the results directory in git
- 30 tests pass (22 existing + 4 training + 4 evaluation) — zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement scripts/evaluate_ablation.py** - `c395cb5` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `scripts/evaluate_ablation.py` — 4 evaluation functions + save_per_play_csv helper
- `results/.gitkeep` — tracks results/ directory in git

## Decisions Made

- Per-sample RMSE uses `mean(dim=1).sqrt()` over `(batch, 2)` pred/target tensors — the `rmse_loss()` function returns a batch-mean scalar which would collapse 16 samples to 1 value; the per-sample formula preserves all N values needed for Wilcoxon test
- `build_ablation_table` single-seed edge case: std set to 0.0 and `rep_idx=0` directly (rather than argsort on 1-element list) — unit tests pass 3 seeds but edge case guard ensures robustness
- `compute_per_position_rmse` guarantees all 4 keys always present — callers (plan 04-04) can index `result["LB"]` without conditional checks even if a rare position has no test samples
- Wilcoxon `alternative="greater"`: testing H1 that Model A RMSE exceeds Model B RMSE — aligns with research hypothesis that ball destination knowledge improves predictions

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Plan 04-04 (ablation orchestration) can now import all 4 functions from `scripts.evaluate_ablation`
- `save_per_play_csv` helper is ready for the multi-seed training loop
- `results/` directory exists and is tracked; `ablation_table.csv` will be overwritten on each full run
- All 30 tests GREEN — clean baseline for plan 04-04

---
*Phase: 04-model-training-and-ablation-evaluation*
*Completed: 2026-03-14*
