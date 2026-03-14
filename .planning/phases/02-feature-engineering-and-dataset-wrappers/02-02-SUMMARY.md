---
phase: 02-feature-engineering-and-dataset-wrappers
plan: "02"
subsystem: dataset

tags: [pytorch, numpy, pandas, trajectory-prediction, social-context, ablation]

requires:
  - phase: 02-01
    provides: "6 RED test stubs in test_dataset.py + minimal_samples / minimal_context_df fixtures"
  - phase: 01-data-pipeline-and-validation
    provides: "build_samples() schema (sample dicts with frames/padding_mask/ball_target_xy)"

provides:
  - "DefensiveTrajectoryDataset: PyTorch Dataset with position filtering, social context assembly, ablation boundary"
  - "STRICT_DEFENSIVE_POSITIONS, CONTEXT_PLAYERS, CONTEXT_FEATURES_PER_PLAYER, SEQUENCE_LENGTH module constants"
  - "_build_context_index: pre-computed nested dict {(gameId,playId): {frameId: {nflId: (x,y)}}}"
  - "_assemble_social_context: 21-slot zero-padded (x,y) matrix sorted by nflId ascending"
  - "include_ball_destination flag: Model A (T,50) vs Model B (T,52) ablation boundary"

affects:
  - 02-03-model-a-transformer
  - 02-04-model-b-transformer
  - 03-training-loop

tech-stack:
  added: []
  patterns:
    - "Pre-build context index in __init__, set self.context_df=None to prevent DataLoader OOM"
    - "Ablation boundary via include_ball_destination bool — columns 50:52 appended for Model B only"
    - "Social context determinism: sort context players by nflId ascending before filling slots"
    - "Sentinel frame_id=-1 in frame_ids list produces all-zero social context row safely"

key-files:
  created:
    - src/data/dataset.py
  modified: []

key-decisions:
  - "Pre-build context index in __init__ (not live DataFrame scan in __getitem__) for DataLoader worker safety"
  - "Social context slots sorted by nflId ascending — deterministic order regardless of DataFrame scan order"
  - "Sentinel value -1 for padded frame slots; _assemble_social_context returns zero row for -1 without branching complexity"
  - "Ball destination fallback chain: _ball_index -> sample['ball_target_xy'] -> ValueError (explicit failure over silent wrong value)"

patterns-established:
  - "TDD RED->GREEN: write failing tests first (Plan 02-01), implement to pass (Plan 02-02)"
  - "context_df=None after index build: standard pattern for all Dataset classes in this project"

requirements-completed: [FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06]

duration: 1min
completed: 2026-03-13
---

# Phase 02 Plan 02: DefensiveTrajectoryDataset Summary

**PyTorch Dataset with pre-built context index, 21-slot social context assembly, and provably correct Model A/B ablation boundary enforced by 6 TDD contract tests**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-13T23:58:11Z
- **Completed:** 2026-03-13T23:59:21Z
- **Tasks:** 1 (single implementation task; TDD stubs were Plan 02-01)
- **Files modified:** 1

## Accomplishments

- Implemented `DefensiveTrajectoryDataset` in `src/data/dataset.py` (247 lines) making all 6 RED test stubs turn GREEN in one shot
- Strict position filter (CB/FS/SS/LB only) enforced at construction time with silent exclusion of all other positions
- Pre-built nested context index eliminates live DataFrame scans in `__getitem__`, making the class DataLoader worker-safe
- Model A/B ablation boundary is compile-time correct: `include_ball_destination=False` produces exactly 50 features; `True` produces 52

## Task Commits

1. **Task 1: Implement DefensiveTrajectoryDataset** - `6e3b72f` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/data/dataset.py` — DefensiveTrajectoryDataset class with context index, social context assembly, and ablation boundary

## Decisions Made

- Pre-build context index in `__init__`, set `self.context_df = None` afterwards — prevents DataLoader worker OOM when the Dataset is pickled for multi-process loading
- Social context slots are sorted by `nflId` ascending — deterministic ordering regardless of DataFrame row order
- Sentinel value `-1` used for padded frame IDs; `_assemble_social_context` returns all-zero row for -1 safely via `dict.get`
- Ball destination lookup chain: `_ball_index` first, then `sample["ball_target_xy"]` fallback, then `ValueError` — explicit failure over silently wrong values

## Deviations from Plan

None — plan executed exactly as written. All 6 tests passed on the first implementation attempt; no iteration required.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `DefensiveTrajectoryDataset` is ready to be instantiated with real `build_samples()` output and the cleaned parquet as `context_df`
- All 14 tests pass (6 Phase 2 + 8 Phase 1) — test suite is stable
- Phase 2 plan 02-03 (Model A transformer) can begin immediately

---
*Phase: 02-feature-engineering-and-dataset-wrappers*
*Completed: 2026-03-13*
