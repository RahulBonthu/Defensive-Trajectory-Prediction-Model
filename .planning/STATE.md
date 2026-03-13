# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-13)

**Core value:** Quantifiably demonstrate that knowing the ball's destination improves defensive player trajectory prediction accuracy via a clean transformer ablation study
**Current focus:** Phase 1 — Data Pipeline and Validation

## Current Position

Phase: 1 of 5 (Data Pipeline and Validation)
Plan: 1 of 5 in current phase
Status: In progress
Last activity: 2026-03-13 — Plan 01-01 complete

Progress: [█░░░░░░░░░] 4%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 4 min
- Total execution time: 0.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-pipeline-and-validation | 1 | 4 min | 4 min |

**Recent Trend:**
- Last 5 plans: 4 min
- Trend: -

*Updated after each plan completion*

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1 pre-work]: NFL BDB exact column name for ball landing location in plays.csv (e.g., `targetX`/`targetY`) must be verified against the specific competition year's data dictionary before implementation begins
- [Phase 1 pre-work]: Confirm training environment (Apple Silicon MPS vs CUDA) before Phase 3 to avoid mid-training surprises
- [Phase 1 pre-work]: Confirm actual dataset year and total play count — if sample count is smaller than expected, per-position subgroup analysis may lack statistical power

## Session Continuity

Last session: 2026-03-13
Stopped at: Completed 01-01-PLAN.md — project scaffold, fixtures, and test stubs
Resume file: None
