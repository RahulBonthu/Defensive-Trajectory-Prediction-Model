# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-13)

**Core value:** Quantifiably demonstrate that knowing the ball's destination improves defensive player trajectory prediction accuracy via a clean transformer ablation study
**Current focus:** Phase 1 — Data Pipeline and Validation

## Current Position

Phase: 1 of 5 (Data Pipeline and Validation)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-03-13 — Roadmap created

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
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

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1 pre-work]: NFL BDB exact column name for ball landing location in plays.csv (e.g., `targetX`/`targetY`) must be verified against the specific competition year's data dictionary before implementation begins
- [Phase 1 pre-work]: Confirm training environment (Apple Silicon MPS vs CUDA) before Phase 3 to avoid mid-training surprises
- [Phase 1 pre-work]: Confirm actual dataset year and total play count — if sample count is smaller than expected, per-position subgroup analysis may lack statistical power

## Session Continuity

Last session: 2026-03-13
Stopped at: Roadmap created — ready to plan Phase 1
Resume file: None
