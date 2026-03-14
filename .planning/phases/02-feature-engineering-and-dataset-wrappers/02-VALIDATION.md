---
phase: 2
slug: feature-engineering-and-dataset-wrappers
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` testpaths = ["tests"] |
| **Quick run command** | `pytest tests/test_dataset.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_dataset.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-W0-01 | W0 | 0 | FEAT-01 | unit | `pytest tests/test_dataset.py::test_position_filter -x` | ❌ W0 | ⬜ pending |
| 2-W0-02 | W0 | 0 | FEAT-02 | unit | `pytest tests/test_dataset.py::test_player_play_independence -x` | ❌ W0 | ⬜ pending |
| 2-W0-03 | W0 | 0 | FEAT-03 | unit | `pytest tests/test_dataset.py::test_sequence_padding_and_mask -x` | ❌ W0 | ⬜ pending |
| 2-W0-04 | W0 | 0 | FEAT-04 | unit | `pytest tests/test_dataset.py::test_social_context_shape -x` | ❌ W0 | ⬜ pending |
| 2-W0-05 | W0 | 0 | FEAT-05 | unit | `pytest tests/test_dataset.py::test_ball_destination_model_b -x` | ❌ W0 | ⬜ pending |
| 2-W0-06 | W0 | 0 | FEAT-06 | unit | `pytest tests/test_dataset.py::test_no_ball_leakage_model_a -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_dataset.py` — stubs for FEAT-01 through FEAT-06 (6 tests)
- [ ] `tests/conftest.py` — add `minimal_samples_fixture` and `minimal_context_df_fixture` for dataset unit tests

*Existing infrastructure: `tests/conftest.py` (tracking_df fixture), `tests/test_pipeline.py` (8 passing tests). New fixtures needed for dataset-level tests.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| DataLoader yields correct tensor shapes end-to-end | FEAT-01/FEAT-05 | Requires real cleaned.parquet on disk | `python -c "from src.data.dataset import NFLDataset; ..."` smoke test |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
