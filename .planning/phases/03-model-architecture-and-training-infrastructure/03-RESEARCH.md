# Phase 3: Model Architecture and Training Infrastructure - Research

**Researched:** 2026-03-13
**Domain:** PyTorch — Conv1d, TransformerEncoder, RMSE loss, device selection, attention weight inspection
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODEL-01 | Shared model class: 1D Conv → Transformer Encoder → Linear output head | Architecture patterns section; Conv1d + TransformerEncoder + Linear wiring |
| MODEL-02 | 1D Conv processes own kinematic time-series for local trajectory pattern extraction | Conv1d shape/permute patterns; kernel_size choice for T=25 |
| MODEL-03 | Transformer Encoder cross-references trajectory tokens against social context | Self-attention over the full (batch, T, d_model) sequence; src_key_padding_mask application |
| MODEL-04 | Linear output head predicts (x, y) ending position — 2 scalar outputs | Mean-pool or CLS token → Linear(d_model, 2); output shape (batch, 2) |
| MODEL-05 | Model A instantiated with ball destination disabled (input_dim=50) | Single class with input_dim constructor arg; no structural difference |
| MODEL-06 | Model B instantiated with ball destination injected as directed feature token (input_dim=52) | Same class; input_dim=52 routes ball_land_x/y through conv and attention |
</phase_requirements>

---

## Summary

Phase 3 builds one shared PyTorch model class that accepts variable input dimension (50 for Model A, 52 for Model B) and produces a (batch, 2) endpoint prediction. The architecture is a standard Conv1d → TransformerEncoder → mean-pool → Linear head — a well-validated pattern for multivariate time-series regression. The training harness is a minimal TDD-verified loop: an overfit test on 100 synthetic samples proves the architecture and optimizer are wired correctly before full training begins in Phase 4.

The primary architectural decisions that must be locked in Phase 3 are: (a) how Conv1d output dimension maps to TransformerEncoder d_model, (b) the correct boolean convention for src_key_padding_mask (PyTorch uses `True` = ignore/pad, which is the OPPOSITE of the dataset's padding_mask convention where `True` = real frame), and (c) how attention weights are extracted at inference time — critical for Phase 3 success criterion 4 (padded positions must show near-zero attention weight). The attention extraction path requires subclassing or monkey-patching because `batch_first=True` bypasses the MHA forward path when using the fast-path implementation.

The training environment concern logged in Phase 1 (Apple Silicon MPS vs CUDA) is resolvable with a standard three-way device selector: CUDA → MPS → CPU. No training code change is needed beyond this selector; PyTorch MPS support is stable as of 2.10.0 for all operations used here.

**Primary recommendation:** Use `nn.Conv1d` with `kernel_size=3, padding=1` (preserves T=25 sequence length), project to `d_model=128`, feed to `nn.TransformerEncoder` with `batch_first=True`, mean-pool over T, then `nn.Linear(128, 2)`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.nn.Conv1d | torch>=2.10.0 (pinned) | Local temporal feature extraction from own-kinematic input | Standard pattern for time-series feature extraction before attention |
| torch.nn.TransformerEncoder + TransformerEncoderLayer | torch>=2.10.0 | Self-attention over time steps to model interactions | PyTorch's complete stacked encoder; handles src_key_padding_mask natively |
| torch.nn.MultiheadAttention | torch>=2.10.0 | Attention weight inspection (subclassed) | Required for SUCCESS CRITERION 4: attention weight inspection on padded sequences |
| torch.nn.MSELoss | torch>=2.10.0 | RMSE computation via `torch.sqrt(mse + eps)` | No built-in RMSELoss in PyTorch; MSELoss + sqrt is idiomatic |
| torch.optim.Adam | torch>=2.10.0 | Optimization | Dominant default for transformer regression; lr=1e-3 for overfit test |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | >=4.0 (pinned) | Progress bar for training loop | Phase 4 full training; optional for overfit test |
| wandb | pinned | Experiment tracking | Phase 4 — not needed in Phase 3 overfit test |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nn.TransformerEncoder | Custom attention | TransformerEncoder handles padding mask, layer norm, and fast-path automatically; hand-rolling adds ~200 lines with no benefit |
| Mean pooling over T | CLS token prepend | Mean pooling is simpler, no extra token dimension, and equivalent for regression; CLS token adds implementation surface for no accuracy gain here |
| Conv1d kernel_size=3 | kernel_size=5 or 7 | k=3 with padding=1 preserves sequence length; wider kernels force padding=2 or more, increasing receptive field but no evidence this improves T=25 sequences |

**Installation:** No new packages needed. All dependencies are already in pyproject.toml.

---

## Architecture Patterns

### Recommended Project Structure
```
src/
├── data/          # (exists) DefensiveTrajectoryDataset
└── model/         # NEW in Phase 3
    ├── __init__.py
    └── trajectory_model.py  # TrajectoryTransformer class
scripts/
└── overfit_test.py          # 100-sample overfit verification
tests/
└── test_model.py            # MODEL-01 through MODEL-06 TDD stubs
```

### Pattern 1: Conv1d Shape Handling (the permute pattern)

**What:** `nn.Conv1d` expects `(batch, channels, seq_len)` but the dataset produces `(batch, T, features)`. Two permutes are needed: one before Conv1d, one after to restore `(batch, T, d_model)` for the Transformer.

**When to use:** Always — this is not optional.

**Example:**
```python
# Source: PyTorch Conv1d docs + PyTorch Forums (confirmed)
# Input x: (batch, T, input_dim)  — e.g., (64, 25, 50) or (64, 25, 52)
x = x.permute(0, 2, 1)            # → (batch, input_dim, T)
x = self.conv(x)                   # Conv1d(input_dim, d_model, kernel_size=3, padding=1)
                                   # → (batch, d_model, T)
x = x.permute(0, 2, 1)            # → (batch, T, d_model)
x = self.transformer_encoder(x, src_key_padding_mask=padding_mask_for_transformer)
```

### Pattern 2: src_key_padding_mask Boolean Convention (CRITICAL INVERSION)

**What:** PyTorch's `TransformerEncoder.forward()` takes `src_key_padding_mask` where `True` = **ignore this position** (it is a padding token). The dataset's `padding_mask` has `True` = **real frame**. These are OPPOSITE conventions and must be inverted before passing to the encoder.

**When to use:** Always when wiring dataset output to TransformerEncoder.

**Example:**
```python
# Source: PyTorch TransformerEncoder docs (confirmed)
# dataset["padding_mask"]: True = real frame, False = padded frame
# TransformerEncoder wants:  True = IGNORE (padded), False = attend to (real)

# In forward():
src_key_padding_mask = ~padding_mask   # invert: True→False, False→True
out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
```

### Pattern 3: Aggregating Sequence Output for Regression

**What:** TransformerEncoder outputs `(batch, T, d_model)`. For a scalar regression target the sequence must be reduced to `(batch, d_model)`. Mean pooling over the real (non-padded) timesteps is standard.

**When to use:** Before the final Linear head.

**Example:**
```python
# Source: standard PyTorch time-series regression pattern
# x: (batch, T, d_model), padding_mask: (batch, T) True=real
# Mean pool only over real (non-padded) frames
mask_f = padding_mask.unsqueeze(-1).float()          # (batch, T, 1)
x_pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # (batch, d_model)
out = self.head(x_pooled)                             # (batch, 2)
```

### Pattern 4: Attention Weight Extraction for Padded Sequence Inspection

**What:** `TransformerEncoder` with `batch_first=True` uses a fast-path implementation that bypasses the `nn.MultiheadAttention` forward call entirely, so `register_forward_hook` does not fire. The correct approach is to subclass `TransformerEncoderLayer` and override `_sa_block` to pass `need_weights=True`.

**When to use:** Phase 3 success criterion 4 — "padded sequence positions receive near-zero attention weight." This is inspection-only, not training code.

**Example:**
```python
# Source: PyTorch Forums + GitHub Issue #99304 (confirmed pattern)
class AttentionCapturingEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that captures attention weights on demand."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_attn_weights: torch.Tensor | None = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self._last_attn_weights = attn_weights.detach()
        return self.dropout1(x)
```

### Anti-Patterns to Avoid

- **Using `batch_first=False` (default):** The dataset yields `(batch, T, features)`. Using the default `seq_first` convention requires two extra transposes. Always set `batch_first=True`.
- **Passing dataset padding_mask directly as src_key_padding_mask without inverting:** Results in the model attending ONLY to padded frames and ignoring real frames — silent correctness bug, loss will diverge rather than decrease.
- **Using `torch.sqrt(loss)` without eps:** When loss approaches zero during overfitting, gradient of sqrt at zero is undefined. Always use `torch.sqrt(loss + 1e-6)`.
- **Applying Conv1d with no padding:** For `kernel_size=3, padding=0`, sequence length drops from 25 to 23 after each conv. If you stack convs without padding, T shrinks and the Transformer sees fewer positions than the padding mask expects.
- **Calling `torch.nn.TransformerEncoder` with `enable_nested_tensor=True` (default) during attention inspection:** The nested tensor path may interfere with custom `_sa_block` overrides in some PyTorch versions. Disable it on the inspection-only copy: `TransformerEncoder(..., enable_nested_tensor=False)`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Self-attention with masking | Custom dot-product attention + mask logic | nn.MultiheadAttention / nn.TransformerEncoderLayer | Handles numerical stability (masked_fill -inf), causal vs padding mask separation, and flash-attention fast path automatically |
| RMSE loss | Custom RMSE class | `torch.sqrt(nn.MSELoss()(pred, tgt) + 1e-6)` | One-liner; eps guard prevents NaN gradient; no class needed until Phase 4 if you want a reusable module |
| Positional encoding | Sinusoidal PE from scratch | None needed — Conv1d implicitly encodes local position via receptive field | With T=25 and Conv1d providing local structure, sinusoidal PE is not required for this scale. The Transformer will learn relative positions from the Conv features. |
| Device selection | If/else chains | Three-line selector (CUDA → MPS → CPU) | Covers training environment uncertainty flagged in Phase 1 |
| DataLoader iteration | Custom iterator | torch.utils.data.DataLoader (already implemented) | Phase 2 DataLoaders are ready; just instantiate and iterate |

**Key insight:** The entire architecture is 5 PyTorch primitives assembled in sequence. Custom implementations add bugs without adding value.

---

## Common Pitfalls

### Pitfall 1: padding_mask Boolean Inversion
**What goes wrong:** Model attends only to padded timesteps; loss diverges; overfit test never converges.
**Why it happens:** Dataset uses `True = real frame` (intuitive); TransformerEncoder uses `True = ignore frame` (PyTorch convention). Developers assume same polarity.
**How to avoid:** Always invert: `src_key_padding_mask = ~batch["padding_mask"]` before passing to encoder.
**Warning signs:** Training loss increases or is constant; attention weights show equal or higher weight on frame indices 10-24 than 0-9 in a padded-sequence test.

### Pitfall 2: Conv1d Shape Error
**What goes wrong:** `RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv1d...` or silent wrong-dimension operations.
**Why it happens:** Conv1d requires `(batch, channels, length)` but dataset yields `(batch, length, channels)`.
**How to avoid:** Explicitly permute `x.permute(0, 2, 1)` before Conv1d and `x.permute(0, 2, 1)` after.
**Warning signs:** Runtime error on first forward pass, or model receives zero gradient through conv.

### Pitfall 3: Attention Inspection Bypassed by Fast Path
**What goes wrong:** Forward hook never fires; `_last_attn_weights` stays None; success criterion 4 cannot be verified.
**Why it happens:** With `batch_first=True`, PyTorch routes to `torch._transformer_encoder_layer_fwd` C++ kernel — this never calls `nn.MultiheadAttention.forward()`.
**How to avoid:** Subclass `TransformerEncoderLayer` and override `_sa_block`. Do NOT use `register_forward_hook` on the MHA module.
**Warning signs:** Attention weight tensor is None after forward pass; hook output list remains empty.

### Pitfall 4: d_model Must Be Divisible by nhead
**What goes wrong:** `ValueError: embed_dim must be divisible by num_heads` at layer construction.
**Why it happens:** Multi-head attention splits d_model across heads.
**How to avoid:** Use `d_model=128, nhead=4` or `d_model=64, nhead=4`. Both are divisible.
**Warning signs:** Instantiation error, not training error — caught immediately.

### Pitfall 5: Overfit Test Stall Due to Too-High Dropout
**What goes wrong:** Loss does not decrease monotonically on the 100-sample overfit test.
**Why it happens:** Dropout randomly zeroes tokens during training on a tiny dataset, preventing memorization.
**How to avoid:** Set `dropout=0.0` in both `TransformerEncoderLayer` and `Conv1d` (no BatchNorm) for the overfit test. Dropout is a generalization tool, not needed for overfit verification.
**Warning signs:** Loss oscillates or decreases non-monotonically across epochs on the small sample.

### Pitfall 6: MPS Fallback Operations Slow Training
**What goes wrong:** Training is unexpectedly slow on Apple Silicon.
**Why it happens:** Some operations fall back to CPU silently unless `PYTORCH_ENABLE_MPS_FALLBACK=1` is set.
**How to avoid:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the training script environment. For the overfit test, CPU is fast enough; MPS matters in Phase 4.
**Warning signs:** `NotImplementedError: operator not implemented for MPS` or unexpectedly slow iteration.

---

## Code Examples

### Full Model Skeleton
```python
# Source: PyTorch Conv1d docs + TransformerEncoder docs (both HIGH confidence)
import torch
import torch.nn as nn


class TrajectoryTransformer(nn.Module):
    """Shared model for Model A (input_dim=50) and Model B (input_dim=52).

    Architecture:
        Conv1d(input_dim, d_model, kernel_size=3, padding=1)  # local trajectory features
        → TransformerEncoder(d_model, nhead, num_layers)        # social context attention
        → masked mean pool over T
        → Linear(d_model, 2)                                    # (x, y) endpoint prediction
    """

    def __init__(
        self,
        input_dim: int,          # 50 for Model A, 52 for Model B
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=3,
            padding=1,        # preserves T=25 exactly
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # expects (batch, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(
        self,
        x: torch.Tensor,           # (batch, T, input_dim)
        padding_mask: torch.Tensor, # (batch, T) True=real, False=padded
    ) -> torch.Tensor:             # (batch, 2)
        # Conv1d requires (batch, channels, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)    # back to (batch, T, d_model)

        # CRITICAL: invert mask convention for PyTorch Transformer
        # Dataset: True=real  →  Transformer: True=ignore(padded)
        src_key_padding_mask = ~padding_mask   # (batch, T)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pool: average only real (non-padded) frames
        mask_f = padding_mask.unsqueeze(-1).float()   # (batch, T, 1)
        x_pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.head(x_pooled)  # (batch, 2)
```

### RMSE Loss
```python
# Source: PyTorch MSELoss docs + PyTorch Forums (confirmed pattern)
def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSE loss in same units as target (yards)."""
    return torch.sqrt(torch.nn.functional.mse_loss(pred, target) + eps)
```

### Device Selection
```python
# Source: PyTorch MPS docs (confirmed)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### Overfit Test Loop Pattern
```python
# Source: standard PyTorch overfitting debug pattern
def overfit_test(model, device, n_samples=100, n_epochs=200):
    """Verify loss decreases monotonically on a tiny fixed dataset."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic data matching real shapes
    T, input_dim = 25, 50  # or 52 for Model B
    x = torch.randn(n_samples, T, input_dim, device=device)
    mask = torch.ones(n_samples, T, dtype=torch.bool, device=device)
    target = torch.randn(n_samples, 2, device=device)

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(x, mask)
        loss = rmse_loss(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Monotone check: loss at end should be lower than at start
    assert losses[-1] < losses[0], (
        f"Overfit test failed: final loss {losses[-1]:.4f} >= initial {losses[0]:.4f}"
    )
    return losses
```

### Attention Weight Inspection (Success Criterion 4)
```python
# Source: PyTorch Forums / GitHub Issue #99304 — confirmed workaround
class AttentionCapturingEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights: torch.Tensor | None = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x_out, weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_attn_weights = weights.detach()  # (batch, T, T)
        return self.dropout1(x_out)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `batch_first=False` (default) | `batch_first=True` (explicit) | PyTorch >=1.9 | Eliminates two extra transposes in every forward call |
| `enable_nested_tensor=False` | `enable_nested_tensor=True` (default in 2.x) | PyTorch 2.0 | Fast-path for padded sequences — but breaks MHA forward hooks |
| Sinusoidal positional encoding | Conv1d handles local position implicitly | tAPE paper 2024 | For T=25 short sequences, conv local features suffice; sinusoidal PE degrades at low d_model |
| `nn.ReLU()` between conv and transformer | `torch.relu()` inline | N/A | Minor; either works — inline avoids extra module |

**Deprecated/outdated:**
- `seq_first` (default) TransformerEncoder: Not deprecated, but universally replaced by `batch_first=True` in new code. The project uses `batch_first=True` throughout.
- Monkey-patching `nn.MultiheadAttention.forward` for attention inspection: Works with `batch_first=False`; does NOT work with `batch_first=True` fast path. Use `_sa_block` override instead (see Pattern 4).

---

## Open Questions

1. **Should `_sa_block` signature include `is_causal` parameter?**
   - What we know: `is_causal` was added to `_sa_block` in PyTorch 2.x; older versions do not have it.
   - What's unclear: The exact PyTorch 2.10.0 signature for `_sa_block` — whether `is_causal` is positional or keyword-only.
   - Recommendation: Override with `**kwargs` to absorb extra arguments safely: `def _sa_block(self, x, attn_mask, key_padding_mask, **kwargs)`.

2. **Monotone overfit test: strict monotone or trend?**
   - What we know: With `dropout=0.0` and lr=1e-3, loss should trend downward on 100 samples over 200 epochs.
   - What's unclear: RMSE loss on synthetic random targets may have noise spikes at epoch boundaries.
   - Recommendation: Check that `final_loss < initial_loss * 0.5` (50% reduction) rather than strict per-epoch monotonicity. Document this tolerance in the test.

3. **Apple Silicon MPS availability in the actual training environment**
   - What we know: The training environment concern was logged in Phase 1. MPS is supported in PyTorch 2.10.0. The device selector pattern handles all three cases.
   - What's unclear: Whether the actual machine running Phase 4 training is macOS with Apple Silicon, Linux with CUDA, or something else.
   - Recommendation: Implement `get_device()` as a utility in `src/model/trajectory_model.py` so Phase 4 inherits it automatically. The overfit test should log which device it ran on.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `pytest tests/test_model.py -x` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| MODEL-01 | Shared class instantiates; forward pass (batch, T, 50) → (batch, 2) | unit | `pytest tests/test_model.py::test_forward_pass_model_a -x` | Wave 0 |
| MODEL-01 | Shared class instantiates; forward pass (batch, T, 52) → (batch, 2) | unit | `pytest tests/test_model.py::test_forward_pass_model_b -x` | Wave 0 |
| MODEL-02 | Conv1d output shape (batch, T, d_model) — sequence length preserved | unit | `pytest tests/test_model.py::test_conv_output_shape -x` | Wave 0 |
| MODEL-03 | Transformer encoder runs without error on masked input | unit | `pytest tests/test_model.py::test_encoder_with_padding_mask -x` | Wave 0 |
| MODEL-04 | Output tensor shape is (batch, 2) — no extra dimensions | unit | `pytest tests/test_model.py::test_output_shape -x` | Wave 0 |
| MODEL-05 | Model A config has input_dim=50 | unit | `pytest tests/test_model.py::test_model_a_config -x` | Wave 0 |
| MODEL-06 | Model B config has input_dim=52 | unit | `pytest tests/test_model.py::test_model_b_config -x` | Wave 0 |
| SC-2 | Training loss decreases on 100-sample overfit | integration | `python scripts/overfit_test.py` | Wave 0 |
| SC-4 | Padded positions receive near-zero attention weight | unit | `pytest tests/test_model.py::test_padding_mask_attention -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_model.py -x`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_model.py` — 9 test stubs for MODEL-01 through MODEL-06 + SC-4
- [ ] `src/model/__init__.py` — empty init for module import
- [ ] `src/model/trajectory_model.py` — TrajectoryTransformer class (created in Wave 1)
- [ ] `scripts/overfit_test.py` — 100-sample overfit verification script

*(No framework install needed — pytest >=8.0 already in pyproject.toml)*

---

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10 TransformerEncoder docs — `src_key_padding_mask` boolean convention (`True` = ignore)
- PyTorch 2.10 Conv1d docs — input shape `(batch, channels, length)` confirmed
- PyTorch 2.10 MultiheadAttention docs — `need_weights`, `average_attn_weights` parameters confirmed
- PyTorch 2.10 MPS backend docs — device selection pattern confirmed; MPS stable in 2.10.0
- PyTorch 2.10 MSELoss docs — no built-in RMSE; `sqrt(MSE + eps)` pattern confirmed

### Secondary (MEDIUM confidence)
- [GitHub Issue #99304](https://github.com/pytorch/pytorch/issues/99304) — TransformerEncoder cannot output attention map; `_sa_block` override approach documented
- [GitHub Issue #100469](https://github.com/pytorch/pytorch/issues/100469) — `batch_first=True` fast-path bypasses MHA; confirmed `_sa_block` is the correct override point
- PyTorch Forums: `src_key_padding_mask` boolean convention (`True` = padded/ignored) — cross-verified with official docs

### Tertiary (LOW confidence)
- General trajectory prediction architecture survey (Conv1d → Transformer → regression head) from multiple 2024 papers — consistent with PRIMARY sources but not PyTorch-version specific

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries from existing pyproject.toml; API details verified against PyTorch 2.10 docs
- Architecture: HIGH — Conv1d + TransformerEncoder + mean-pool + Linear is a well-documented pattern; specific gotchas (mask inversion, permute, attention extraction) verified via official docs and confirmed issues
- Pitfalls: HIGH — mask inversion bug and fast-path attention bypass are documented in official PyTorch GitHub issues; not speculation

**Research date:** 2026-03-13
**Valid until:** 2026-06-13 (PyTorch 2.x API stable; 90 days reasonable for torch.nn primitives)
