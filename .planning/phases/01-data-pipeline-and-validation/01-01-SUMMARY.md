---
phase: 01-data-pipeline-and-validation
plan: 01
subsystem: testing
tags: [pytest, pandas, numpy, pyproject, scaffold]

# Dependency graph
requires: []
provides:
  - pyproject.toml with pinned Phase 1 dependencies (pandas==3.0.1, numpy==2.4, scikit-learn, torch, etc.)
  - pytest.ini with testpaths=tests for consistent test discovery
  - .gitignore that excludes data/raw/, data/processed/, *.zip, *.parquet
  - src/ and tests/ directory structure with __init__.py files
  - tests/conftest.py with synthetic tracking_df, plays_df, games_df, players_df fixtures (no real data needed)
  - tests/test_pipeline.py with 8 skipped stubs covering DATA-01, DATA-02, PREP-01 through PREP-06
  - data/ and outputs/validation/ directory stubs (.gitkeep)
affects: [01-02, 01-03, 01-04, 01-05]

# Tech tracking
tech-stack:
  added: [pytest>=8.0, pandas==3.0.1, numpy==2.4, scipy>=1.14, scikit-learn>=1.8.0, torch>=2.10.0, matplotlib==3.10.8, seaborn>=0.13, wandb, tqdm>=4.0, jupyterlab>=4.0, pyarrow]
  patterns: [PEP-621 pyproject.toml, pytest fixture injection, skip-until-implemented test stubs]

key-files:
  created:
    - pyproject.toml
    - pytest.ini
    - .gitignore
    - src/__init__.py
    - src/data/__init__.py
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_pipeline.py
    - data/.gitkeep
    - outputs/validation/.gitkeep
  modified: []

key-decisions:
  - "PEP 621 pyproject.toml used (not setup.py) for modern Python packaging"
  - "Test stubs use pytest.mark.skip so pytest exits 0 — CI will never fail on unimplemented tests"
  - "Synthetic fixtures build DataFrames from constants — zero dependency on real data for testing"
  - "Missing frame intentionally omitted from tracking_df fixture (nflId=1, frameId=3) to test interpolation in plan 01-02"

patterns-established:
  - "Skip pattern: all Phase 1 tests start skipped, skip reason specifies which plan implements them"
  - "Fixture pattern: conftest.py provides NFL BDB schema-accurate DataFrames without real files"
  - "Directory pattern: data/ tracked via .gitkeep but excluded from git via .gitignore patterns"

requirements-completed: [DATA-03]

# Metrics
duration: 4min
completed: 2026-03-13
---

# Phase 1 Plan 01: Project Scaffold Summary

**pip-installable Python project skeleton with pinned dependencies, synthetic pytest fixtures, and 8 skipped test stubs covering all Phase 1 preprocessing requirements**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-03-13T22:35:07Z
- **Completed:** 2026-03-13T22:38:34Z
- **Tasks:** 2 of 2
- **Files modified:** 10

## Accomplishments

- pyproject.toml with PEP 621 format, all Phase 1 dependencies pinned, pip-installable
- conftest.py providing tracking_df/plays_df/games_df/players_df fixtures from synthetic data — no NFL dataset required to run tests
- 8 skipped test stubs that define the full Phase 1 test surface (DATA-01, DATA-02, PREP-01 through PREP-06)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create pyproject.toml, .gitignore, pytest.ini, and directory skeleton** - `ce63b93` (chore)
2. **Task 2: Create conftest.py with synthetic fixtures and test_pipeline.py with stubs** - `51a8c52` (feat)

**Plan metadata:** _(to be added in final commit)_

## Files Created/Modified

- `pyproject.toml` - PEP 621 dependency manifest with all Phase 1 packages pinned
- `.gitignore` - Excludes data/raw/, data/processed/, *.zip, *.parquet, model checkpoints
- `pytest.ini` - testpaths = tests, belt-and-suspenders alongside pyproject.toml
- `src/__init__.py` - Package marker for src/
- `src/data/__init__.py` - Package marker for src/data/
- `tests/__init__.py` - Package marker for tests/
- `tests/conftest.py` - Synthetic NFL BDB fixtures: tracking_df (29 rows with one intentional missing frame), plays_df, games_df, players_df
- `tests/test_pipeline.py` - 8 skipped stubs: test_zip_extraction, test_csv_loading, test_los_normalization, test_play_direction_flip, test_angle_sincos_encoding, test_interpolation_and_flagging, test_acceleration_computed, test_temporal_split_disjoint
- `data/.gitkeep` - Directory stub (data/ itself excluded from git)
- `outputs/validation/.gitkeep` - Directory stub for validation outputs

## Decisions Made

- Used PEP 621 `[project]` table in pyproject.toml (not setup.py) for forward-compatible packaging
- Both pytest.ini and pyproject.toml `[tool.pytest.ini_options]` configured — pytest.ini takes precedence, pyproject config is belt-and-suspenders
- Test stubs use `pytest.mark.skip` rather than empty bodies so they're clearly intentional skips visible in CI output
- Tracking fixture omits frameId=3 for nflId=1 intentionally to test interpolation in plan 01-02

## Deviations from Plan

None — plan executed exactly as written.

One minor deviation handled automatically: pandas was not installed in the environment. Installed via `pip install pandas numpy` (Rule 3 — blocking issue) before test verification. This is a local environment setup step, not a code change.

## Issues Encountered

- pytest not installed initially in environment — installed via pip, tests then ran correctly
- pandas not installed initially — installed via pip, conftest.py then loaded correctly

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Full test surface defined: plan 01-02 can implement src/data/loader.py and src/data/preprocessor.py and make tests go green
- Synthetic fixtures ready for immediate use in plan 01-02
- .gitignore ensures the NFL dataset (once downloaded in plan 01-04) will never be committed

## Self-Check: PASSED

All created files verified to exist on disk. All task commits verified in git log.

---
*Phase: 01-data-pipeline-and-validation*
*Completed: 2026-03-13*
