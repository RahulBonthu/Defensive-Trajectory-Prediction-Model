---
phase: 02-feature-engineering-and-dataset-wrappers
plan: "03"
subsystem: testing
tags: [pytorch, dataset, dataloader, smoke-test, parquet, integration]

# Dependency graph
requires:
  - phase: 02-feature-engineering-and-dataset-wrappers
    provides: DefensiveTrajectoryDataset (src/data/dataset.py) and cleaned.parquet
provides:
  - End-to-end integration smoke test confirming DataLoader shapes on real 4.88M-row dataset
  - Human-verified confirmation that Model A outputs (batch, 25, 50) and Model B outputs (batch, 25, 52)
  - Verified sample counts: train=52779, val=7497 after strict position filter
affects: [03-model-a-transformer, 04-model-b-transformer, phase-3]

# Tech tracking
tech-stack:
  added: []
  patterns: [Integration smoke test script at scripts/ level; explicit shape assertions with PASS/FAIL report; human-verify gate before phase close]

key-files:
  created: [scripts/smoke_test_dataset.py]
  modified: []

key-decisions:
  - "Smoke test runs from project root so relative parquet/splits.json paths resolve correctly"
  - "num_workers=0 avoids multiprocessing overhead for a one-shot integration script"
  - "Human visual gate required before Phase 2 close — synthetic unit tests do not cover real-data shapes"

patterns-established:
  - "Integration smoke test pattern: load real artifact, run assertions, print PASS/FAIL report, exit 0 or 1"
  - "Progress message before expensive context index build (~30-60s) so user knows script is working"

requirements-completed: [FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06]

# Metrics
duration: ~5min
completed: 2026-03-13
---

# Phase 02-03: Smoke Test Dataset Summary

**Integration smoke test against real 4.88M-row parquet confirms Model A (batch,25,50) and Model B (batch,25,52) shapes with train=52779 and val=7497 samples; all 5 assertions PASS and human-verified.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-13T17:00:00Z
- **Completed:** 2026-03-13T17:05:00Z
- **Tasks:** 2 (1 auto + 1 human-verify checkpoint)
- **Files modified:** 1

## Accomplishments

- Wrote `scripts/smoke_test_dataset.py` (130 lines) that loads `cleaned.parquet` and `splits.json`, builds four datasets across train/val splits, and fires two DataLoaders
- Verified Model A input tensor shape `(64, 25, 50)` and Model B input tensor shape `(64, 25, 52)` against real data
- Confirmed strict position filter (CB/FS/SS/LB) yields exactly 52,779 train samples and 7,497 val samples
- Human reviewed printed shape report and confirmed all 5 assertions True — Phase 2 integration gate passed

## Task Commits

Each task was committed atomically:

1. **Task 1: Write and run scripts/smoke_test_dataset.py** - `5f9bc5b` (feat)
2. **Task 2: Human visual check of smoke test output** - checkpoint approved (no code commit)

**Plan metadata:** (this summary commit)

## Files Created/Modified

- `scripts/smoke_test_dataset.py` - End-to-end DataLoader shape verification against real `data/processed/cleaned.parquet`; exits 0 if all 5 assertions pass

## Decisions Made

- Smoke test runs from project root so relative paths to `data/processed/cleaned.parquet` and `data/processed/splits.json` resolve correctly without path manipulation
- `num_workers=0` used in DataLoaders — avoids multiprocessing overhead for a one-shot integration script
- Human-verify gate kept blocking — unit tests use synthetic fixtures and cannot substitute for a real-data shape check

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 success criteria fully verified:
  - SC1: DataLoader Model A shape `(batch, 25, 50)` confirmed on real data
  - SC1: DataLoader Model B shape `(batch, 25, 52)` confirmed on real data
  - SC2: Unit test asserting Model A width == 50 passes (14 tests green)
  - SC3: Only CB/FS/SS/LB in dataset (strict filter, count verified at 52,779 train)
  - SC4: padding_mask bool tensor propagates through DataLoader
- Phase 3 (Model A transformer) can begin; dataset interface is locked and verified against production data
- Concern carried forward: confirm training hardware (MPS vs CUDA) before Phase 3 training runs

---
*Phase: 02-feature-engineering-and-dataset-wrappers*
*Completed: 2026-03-13*
