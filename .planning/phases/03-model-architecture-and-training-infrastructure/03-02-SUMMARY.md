---
phase: 03-model-architecture-and-training-infrastructure
plan: "02"
subsystem: model
tags: [pytorch, transformer, conv1d, attention, tdd, green-phase]

# Dependency graph
requires:
  - phase: 03-model-architecture-and-training-infrastructure/03-01
    provides: 8 RED test stubs in tests/test_model.py, scripts/overfit_test.py with 200-epoch overfit loop
  - phase: 02-feature-engineering-and-dataset-wrappers
    provides: DefensiveTrajectoryDataset with input_dim=50/52, padding_mask convention (True=real frame)
provides:
  - src/model/trajectory_model.py — TrajectoryTransformer, AttentionCapturingEncoderLayer, rmse_loss, get_device
  - Model A (input_dim=50) and Model B (input_dim=52) share one class, only input_dim differs
  - Verified 22/22 tests GREEN and overfit convergence for both model variants
affects: [03-03-training-loop, phase-4-training]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Conv1d permute pattern: x.permute(0,2,1) before conv, x.permute(0,2,1) after to restore batch-first"
    - "Padding mask inversion: src_key_padding_mask = ~padding_mask (dataset True=real, transformer True=ignore)"
    - "Masked mean pooling: (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)"
    - "AttentionCapturingEncoderLayer overrides forward() not _sa_block() to bypass PyTorch 2.x C++ fast-path in eval mode"
    - "enable_nested_tensor=False on TransformerEncoder for subclass compatibility"

key-files:
  created:
    - src/model/trajectory_model.py
  modified: []

key-decisions:
  - "Override forward() in AttentionCapturingEncoderLayer instead of _sa_block() — PyTorch 2.10 eval mode takes a C++ fast-path that completely bypasses _sa_block, making _sa_block override useless for attention capture"
  - "capture_attention=True constructor kwarg selects AttentionCapturingEncoderLayer for all encoder layers — consistent with test_model.py API contract from Plan 03-01"
  - "dropout=0.1 default; overfit test uses dropout=0.0 to allow clean memorization — both tested, both pass"

patterns-established:
  - "Attention capture: override forward() not _sa_block() when using PyTorch 2.x — fast-path bypasses _sa_block in eval mode"
  - "Model variant pattern: single TrajectoryTransformer class, input_dim is the only differing argument between Model A and Model B"

requirements-completed: [MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06]

# Metrics
duration: 8min
completed: 2026-03-14
---

# Phase 3 Plan 02: TrajectoryTransformer Implementation Summary

**Conv1d -> TransformerEncoder -> masked mean-pool transformer with 95%+ overfit convergence, verified GREEN across all 22 tests**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-14T00:31:08Z
- **Completed:** 2026-03-14T00:39:00Z
- **Tasks:** 2 (Task 1: implement + GREEN; Task 2: overfit verification)
- **Files modified:** 1

## Accomplishments

- Created `src/model/trajectory_model.py` with all four required exports: `TrajectoryTransformer`, `AttentionCapturingEncoderLayer`, `rmse_loss`, `get_device`
- All 8 RED test stubs from Plan 03-01 turned GREEN; 22/22 total tests pass (0 regressions)
- Overfit test confirmed: Model A (input_dim=50) 95.9% loss reduction; Model B (input_dim=52) 98.4% — both exceed 50% threshold

## Task Commits

1. **Task 1: Implement TrajectoryTransformer and helpers** - `6f51a0a` (feat)

Task 2 (overfit test) required no file changes — `scripts/overfit_test.py` ran unchanged and exited 0.

## Files Created/Modified

- `src/model/trajectory_model.py` — TrajectoryTransformer (Conv1d->Encoder->MeanPool->Linear), AttentionCapturingEncoderLayer (attention capture in eval mode), rmse_loss, get_device

## Decisions Made

- **Override `forward()` in `AttentionCapturingEncoderLayer`, not `_sa_block()`:** Plan 03-01 specified `_sa_block` override. During testing, `last_attn_weights` remained `None` even with `enable_nested_tensor=False`. Investigation showed PyTorch 2.10 `TransformerEncoderLayer.forward()` takes a C++ fast-path (`torch._transformer_encoder_layer_fwd`) in eval mode when all fast-path conditions pass, completely skipping `_sa_block`. Overriding `forward()` directly guarantees the attention weights are always captured regardless of fast-path state.

- **`capture_attention=True` kwarg (not `encoder_layer_cls`):** The 03-01 SUMMARY noted `capture_attention=True` as the test contract; the 03-02 PLAN spec listed `encoder_layer_cls` as an alternative. The test file unambiguously uses `capture_attention=True`, so that is what was implemented.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Overrode `forward()` instead of `_sa_block()` in AttentionCapturingEncoderLayer**

- **Found during:** Task 1 (GREEN verification — `test_padding_mask_attention` failing)
- **Issue:** Plan specified overriding `_sa_block` with `**kwargs`. With PyTorch 2.10 in eval mode, the C++ fast-path in `TransformerEncoderLayer.forward()` is taken when no grad or hook conditions prevent it, making `_sa_block` unreachable. `last_attn_weights` remained `None` after forward pass.
- **Fix:** Replaced `_sa_block` override with a full `forward()` override that calls `self.self_attn` directly with `need_weights=True`, then applies the standard post-attention residual + norm + FFN path.
- **Files modified:** `src/model/trajectory_model.py`
- **Verification:** `test_padding_mask_attention` passes GREEN; `padded_col_mean < 0.05` assertion satisfied
- **Committed in:** `6f51a0a` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in spec caused by PyTorch version behavior)
**Impact on plan:** Required for SC-4 correctness. No scope creep. All must_haves satisfied.

## Issues Encountered

None beyond the `_sa_block` fast-path issue documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `TrajectoryTransformer` is fully tested and verified for both Model A (input_dim=50) and Model B (input_dim=52)
- `rmse_loss` and `get_device` are available for the training loop in Phase 3 Plan 03 and Phase 4
- Overfit test passed for both variants — architecture and optimizer wiring confirmed correct
- Phase 4 training can proceed with confidence in the model architecture

---
*Phase: 03-model-architecture-and-training-infrastructure*
*Completed: 2026-03-14*
