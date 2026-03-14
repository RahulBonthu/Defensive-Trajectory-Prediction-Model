---
phase: 4
slug: model-training-and-ablation-evaluation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~20 seconds (unit/smoke tests only; full training is a long-running script) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-W0-01 | W0 | 0 | TRAIN-01 | smoke | `pytest tests/test_training.py::test_identical_hyperparameters -x` | ❌ W0 | ⬜ pending |
| 4-W0-02 | W0 | 0 | TRAIN-02 | unit | `pytest tests/test_training.py::test_rmse_loss_used -x` | ❌ W0 | ⬜ pending |
| 4-W0-03 | W0 | 0 | TRAIN-03 | smoke | `pytest tests/test_training.py::test_wandb_logging -x` | ❌ W0 | ⬜ pending |
| 4-W0-04 | W0 | 0 | TRAIN-04 | integration | `pytest tests/test_training.py::test_checkpoints_saved -x` | ❌ W0 | ⬜ pending |
| 4-W0-05 | W0 | 0 | EVAL-01 | unit | `pytest tests/test_evaluation.py::test_per_play_rmse_shape -x` | ❌ W0 | ⬜ pending |
| 4-W0-06 | W0 | 0 | EVAL-02 | unit | `pytest tests/test_evaluation.py::test_ablation_table_columns -x` | ❌ W0 | ⬜ pending |
| 4-W0-07 | W0 | 0 | EVAL-03 | unit | `pytest tests/test_evaluation.py::test_significance_test_runs -x` | ❌ W0 | ⬜ pending |
| 4-W0-08 | W0 | 0 | EVAL-04 | unit | `pytest tests/test_evaluation.py::test_per_position_rmse_keys -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_training.py` — 4 stubs (TRAIN-01 through TRAIN-04), mock wandb, synthetic DataLoader
- [ ] `tests/test_evaluation.py` — 4 stubs (EVAL-01 through EVAL-04), synthetic per-play RMSE vectors

*Existing infrastructure: 22 passing tests across test_pipeline.py, test_dataset.py, test_model.py — regression guard.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Validation loss decreases then plateaus | TRAIN-04/SC-2 | Requires real training run + wandb chart inspection | Run `python scripts/train.py` and review wandb val_loss curve |
| model_a_best.pt and model_b_best.pt exist with valid weights | TRAIN-04 | Requires real training run on cleaned.parquet | `python -c "import torch; torch.load('checkpoints/model_a_best.pt')"` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 20s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
