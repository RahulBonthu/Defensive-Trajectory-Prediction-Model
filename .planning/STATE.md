---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 03-02 — TrajectoryTransformer implemented; 22/22 tests GREEN; overfit test 95%+ loss reduction for both model variants
last_updated: "2026-03-14T00:34:38.523Z"
last_activity: "2026-03-13 — Plan 02-02 complete (DefensiveTrajectoryDataset: position filter, context index, social context, ablation boundary)"
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 10
  completed_plans: 10
  percent: 100
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Completed 02-02 — DefensiveTrajectoryDataset implemented; all 14 tests GREEN"
last_updated: "2026-03-14T00:00:19Z"
last_activity: "2026-03-13 — Plan 02-02 complete (DefensiveTrajectoryDataset; 6 Phase 2 tests GREEN; 14 total tests passing)"
progress:
  [██████████] 100%
  completed_phases: 1
  total_plans: 8
  completed_plans: 7
  percent: 88
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-13)

**Core value:** Quantifiably demonstrate that knowing the ball's destination improves defensive player trajectory prediction accuracy via a clean transformer ablation study
**Current focus:** Phase 2 — Feature Engineering and Dataset Wrappers

## Current Position

Phase: 2 of 5 (Feature Engineering and Dataset Wrappers) — in progress
Plan: 02-02 complete (2/N) — next: plan 02-03 (Model A transformer)
Status: Phase 2 in progress — DefensiveTrajectoryDataset implemented; all 14 tests GREEN
Last activity: 2026-03-13 — Plan 02-02 complete (DefensiveTrajectoryDataset: position filter, context index, social context, ablation boundary)

Progress: [█████████░] 88%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 8 min
- Total execution time: ~0.75 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-pipeline-and-validation | 4 | 43 min | 11 min |
| 02-feature-engineering-and-dataset-wrappers | 2 | 2 min | 1 min |

**Recent Trend:**
- Last 5 plans: 4 min, 7 min, 8 min, 24 min, 1 min
- Trend: stable (02-02 fast due to complete TDD scaffold from 02-01)

*Updated after each plan completion*
| Phase 02-feature-engineering-and-dataset-wrappers P03 | 5 | 2 tasks | 1 files |
| Phase 03-model-architecture-and-training-infrastructure P01 | 2 | 1 tasks | 3 files |
| Phase 03-model-architecture-and-training-infrastructure P02 | 8 | 2 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Project init]: Two separate models (not one with masking) — cleaner ablation, eliminates masking confounds
- [Project init]: Predict ending location only, not full trajectory — simplifies problem, matches poster scope
- [Project init]: Include all 22 players as social context features — richer context, aligns with future work
- [Project init]: RMSE in yards as primary metric — standard for continuous motion prediction, interpretable
- [01-01]: PEP 621 pyproject.toml (not setup.py) for modern Python packaging
- [01-01]: Test stubs use pytest.mark.skip — CI always exits 0 until implementations are added
- [01-01]: Synthetic fixtures build DataFrames from constants — zero dependency on real data for unit testing
- [01-01]: Missing frame intentionally omitted from tracking fixture to test interpolation in plan 01-02
- [Phase 01-data-pipeline-and-validation]: Pipeline order: normalize_coordinates -> encode_angles -> interpolate_missing_frames -> compute_acceleration (interpolate sin/cos, not raw degrees)
- [Phase 01-data-pipeline-and-validation]: Use explicit groupby loop (not .apply()) in interpolate_missing_frames to preserve gameId/playId/nflId key columns
- [01-03]: DATA-03 enforcement — code-first gate: all Wave 1 source pushed to GitHub before dataset zip is ever placed locally
- [01-03]: Dataset filename nfl-big-data-bowl-2026-prediction.zip must match exactly — loader.py references this filename
- [01-04]: BDB 2026 dataset uses input_2023_w*.csv format — no separate plays/games/players CSVs; ball_land_x/ball_land_y provided directly per row; week derived from filename
- [01-04]: ball_land_x and ball_land_y are in raw NFL field coords; apply same LOS-relative + center-y normalization as player x/y
- [01-04]: pyproject.toml build-backend: setuptools.build_meta (not setuptools.backends.legacy:build — removed in v82)

- [02-01]: Test stubs use real assertions (not pytest.mark.skip) to keep RED state visible — skip masks the RED/GREEN signal
- [02-01]: minimal_context_df fixture includes ball_land_x/ball_land_y columns so dataset implementation can read them without disk access
- [02-01]: STRICT_POSITIONS = {"CB", "FS", "SS", "LB"} documented in test_dataset.py as the position filtering contract
- [Phase 02-02]: Pre-build context index in __init__, set self.context_df=None afterwards — prevents DataLoader worker OOM
- [Phase 02-02]: Social context slots sorted by nflId ascending — deterministic order regardless of DataFrame scan order
- [Phase 02-02]: Ball destination fallback chain: _ball_index -> sample['ball_target_xy'] -> ValueError (explicit failure over silent wrong value)
- [Phase 02-03]: Smoke test runs from project root so relative parquet/splits.json paths resolve correctly
- [Phase 02-03]: Human-verify gate kept blocking — unit tests use synthetic fixtures and cannot substitute for real-data shape check
- [Phase 03-model-architecture-and-training-infrastructure]: Deferred import inside test functions so pytest collects 8 tests individually (not 1 collection ERROR)
- [Phase 03-model-architecture-and-training-infrastructure]: capture_attention=True kwarg pattern for TrajectoryTransformer — Plan 03-02 must implement this to enable SC-4 attention weight inspection
- [Phase 03-model-architecture-and-training-infrastructure]: overfit_test.py written in full during RED phase so Plan 03-02 only needs the import to succeed
- [Phase 03-model-architecture-and-training-infrastructure]: Override forward() in AttentionCapturingEncoderLayer not _sa_block() — PyTorch 2.10 C++ fast-path bypasses _sa_block in eval mode
- [Phase 03-model-architecture-and-training-infrastructure]: capture_attention=True constructor kwarg selects AttentionCapturingEncoderLayer — single kwarg cleaner than passing layer class

### Pending Todos

None.

### Blockers/Concerns

- [RESOLVED 01-04]: NFL BDB exact column name for ball landing location — RESOLVED: BDB 2026 provides ball_land_x / ball_land_y directly per row; no derivation needed
- [Phase 1 pre-work]: Confirm training environment (Apple Silicon MPS vs CUDA) before Phase 3 to avoid mid-training surprises
- [Phase 1 pre-work]: Dataset has 4.88M rows / 93,824 total samples (72,400 train + 10,276 val + 11,148 test) — sufficient for Phase 2 training

## Session Continuity

Last session: 2026-03-14T00:34:38.520Z
Stopped at: Completed 03-02 — TrajectoryTransformer implemented; 22/22 tests GREEN; overfit test 95%+ loss reduction for both model variants
Resume file: None
