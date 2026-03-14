---
phase: 3
slug: model-architecture-and-training-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/test_model.py -x` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_model.py -x`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-W0-01 | W0 | 0 | MODEL-01 | unit | `pytest tests/test_model.py::test_forward_pass_model_a -x` | ❌ W0 | ⬜ pending |
| 3-W0-02 | W0 | 0 | MODEL-01 | unit | `pytest tests/test_model.py::test_forward_pass_model_b -x` | ❌ W0 | ⬜ pending |
| 3-W0-03 | W0 | 0 | MODEL-02 | unit | `pytest tests/test_model.py::test_conv_output_shape -x` | ❌ W0 | ⬜ pending |
| 3-W0-04 | W0 | 0 | MODEL-03 | unit | `pytest tests/test_model.py::test_encoder_with_padding_mask -x` | ❌ W0 | ⬜ pending |
| 3-W0-05 | W0 | 0 | MODEL-04 | unit | `pytest tests/test_model.py::test_output_shape -x` | ❌ W0 | ⬜ pending |
| 3-W0-06 | W0 | 0 | MODEL-05 | unit | `pytest tests/test_model.py::test_model_a_config -x` | ❌ W0 | ⬜ pending |
| 3-W0-07 | W0 | 0 | MODEL-06 | unit | `pytest tests/test_model.py::test_model_b_config -x` | ❌ W0 | ⬜ pending |
| 3-W0-08 | W0 | 0 | SC-4 | unit | `pytest tests/test_model.py::test_padding_mask_attention -x` | ❌ W0 | ⬜ pending |
| 3-W1-01 | 03-02 | 1 | SC-2 | integration | `python scripts/overfit_test.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_model.py` — 8 failing test stubs (MODEL-01 × 2, MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06, SC-4)
- [ ] `src/model/__init__.py` — empty module init (enables `from src.model.trajectory_model import ...`)
- [ ] `scripts/overfit_test.py` — 100-sample overfit verification script stub

*Existing infrastructure: pytest config in pyproject.toml, 14 passing tests (6 dataset + 8 pipeline). No new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | — | All behaviors have automated verification | — |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
