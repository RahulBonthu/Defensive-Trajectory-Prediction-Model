---
phase: 1
slug: data-pipeline-and-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pytest.ini` or `pyproject.toml` — Wave 0 creates |
| **Quick run command** | `pytest tests/test_pipeline.py -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_pipeline.py -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | DATA-03 | unit | `pytest tests/test_pipeline.py::test_repo_committed -x -q` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | DATA-01 | unit | `pytest tests/test_pipeline.py::test_zip_extraction -x -q` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | DATA-02 | unit | `pytest tests/test_pipeline.py::test_csv_loading -x -q` | ❌ W0 | ⬜ pending |
| 1-01-04 | 01 | 2 | PREP-01 | unit | `pytest tests/test_pipeline.py::test_los_normalization -x -q` | ❌ W0 | ⬜ pending |
| 1-01-05 | 01 | 2 | PREP-02 | unit | `pytest tests/test_pipeline.py::test_play_direction_flip -x -q` | ❌ W0 | ⬜ pending |
| 1-01-06 | 01 | 2 | PREP-03 | unit | `pytest tests/test_pipeline.py::test_angle_sincos_encoding -x -q` | ❌ W0 | ⬜ pending |
| 1-01-07 | 01 | 2 | PREP-04 | unit | `pytest tests/test_pipeline.py::test_interpolation_and_flagging -x -q` | ❌ W0 | ⬜ pending |
| 1-01-08 | 01 | 2 | PREP-05 | unit | `pytest tests/test_pipeline.py::test_acceleration_computed -x -q` | ❌ W0 | ⬜ pending |
| 1-01-09 | 01 | 3 | PREP-06 | unit | `pytest tests/test_pipeline.py::test_temporal_split_disjoint -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_pipeline.py` — stubs for DATA-01, DATA-02, DATA-03, PREP-01 through PREP-06
- [ ] `tests/conftest.py` — shared fixtures (tiny synthetic tracking DataFrame, sample plays.csv rows)
- [ ] `pytest` install in `requirements.txt`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 50-play overlay shows all offense in +x direction | PREP-02 | Requires visual inspection of matplotlib plot | Run `python scripts/validate_pipeline.py --plot-overlay` and confirm all play traces move left-to-right |
| Ball landing column name verified in plays.csv | DATA-02 | Column name varies by competition year; must inspect actual CSV headers | After extraction, run `python -c "import pandas as pd; df=pd.read_csv('data/raw/train/plays.csv'); print(df.columns.tolist())"` and confirm ball landing coordinate column(s) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
