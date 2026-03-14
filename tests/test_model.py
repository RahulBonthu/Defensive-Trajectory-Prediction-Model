"""
RED stubs for TrajectoryTransformer — Phase 3 Plan 01.

All 8 tests MUST fail with ImportError until trajectory_model.py is
implemented (Plan 03-02). This file defines the TDD contract for
MODEL-01 through MODEL-06 and SC-4.

The import of TrajectoryTransformer is deferred inside each test so that
pytest can collect all 8 tests individually and report 8 ERRORs rather
than a single collection-level failure.

Input shapes:
  Model A: (batch, T=25, input_dim=50)
  Model B: (batch, T=25, input_dim=52)
  Output both: (batch, 2)
  padding_mask: (batch, T) bool, True=real frame, False=padded frame
"""

import torch
import pytest  # noqa: F401


# ---------------------------------------------------------------------------
# Helper — import inside tests so pytest can collect all 8 individually
# ---------------------------------------------------------------------------

def _import_model(capture_attention: bool = False):
    """Import and return TrajectoryTransformer.  Fails with ImportError until
    Plan 03-02 creates src/model/trajectory_model.py.
    """
    from src.model.trajectory_model import TrajectoryTransformer  # noqa
    return TrajectoryTransformer


# ---------------------------------------------------------------------------
# MODEL-01 (×2): Shared class forward pass — Model A and Model B
# ---------------------------------------------------------------------------

def test_forward_pass_model_a():
    """MODEL-01: Forward pass with input_dim=50 produces (batch, 2) output."""
    TrajectoryTransformer = _import_model()
    model = TrajectoryTransformer(input_dim=50)
    x = torch.randn(2, 25, 50)
    mask = torch.ones(2, 25, dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"


def test_forward_pass_model_b():
    """MODEL-01: Forward pass with input_dim=52 produces (batch, 2) output."""
    TrajectoryTransformer = _import_model()
    model = TrajectoryTransformer(input_dim=52)
    x = torch.randn(2, 25, 52)
    mask = torch.ones(2, 25, dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"


# ---------------------------------------------------------------------------
# MODEL-02: Conv1d preserves sequence length T=25
# ---------------------------------------------------------------------------

def test_conv_output_shape():
    """MODEL-02: Conv1d with kernel_size=3, padding=1 preserves T=25."""
    TrajectoryTransformer = _import_model()
    model = TrajectoryTransformer(input_dim=50)
    x = torch.randn(1, 25, 50)
    mask = torch.ones(1, 25, dtype=torch.bool)
    out = model(x, mask)
    # If T were not preserved the downstream attention/pooling would fail or
    # produce a different shape.  The final output must be (1, 2).
    assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"


# ---------------------------------------------------------------------------
# MODEL-03: Transformer encoder runs without error on a masked (padded) input
# ---------------------------------------------------------------------------

def test_encoder_with_padding_mask():
    """MODEL-03: Encoder accepts padding mask; last 5 frames are padding."""
    TrajectoryTransformer = _import_model()
    model = TrajectoryTransformer(input_dim=50)
    x = torch.randn(2, 25, 50)
    mask = torch.ones(2, 25, dtype=torch.bool)
    mask[0, 20:] = False  # first sample: frames 20-24 are padded
    out = model(x, mask)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"


# ---------------------------------------------------------------------------
# MODEL-04: Output has exactly 2 scalars, float dtype
# ---------------------------------------------------------------------------

def test_output_shape():
    """MODEL-04: Output is (batch=4, 2) — no extra dims, dtype float."""
    TrajectoryTransformer = _import_model()
    model = TrajectoryTransformer(input_dim=50)
    x = torch.randn(4, 25, 50)
    mask = torch.ones(4, 25, dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"
    assert out.dtype in (torch.float32, torch.float64), (
        f"Expected float dtype, got {out.dtype}"
    )
    assert out.ndim == 2, f"Expected 2 dims, got {out.ndim}"


# ---------------------------------------------------------------------------
# MODEL-05: Model A config inspection — conv.in_channels == 50
# ---------------------------------------------------------------------------

def test_model_a_config():
    """MODEL-05: TrajectoryTransformer(input_dim=50) has conv.in_channels == 50."""
    TrajectoryTransformer = _import_model()
    model = TrajectoryTransformer(input_dim=50)
    assert model.conv.in_channels == 50, (
        f"Expected conv.in_channels=50, got {model.conv.in_channels}"
    )


# ---------------------------------------------------------------------------
# MODEL-06: Model B config inspection — conv.in_channels == 52
# ---------------------------------------------------------------------------

def test_model_b_config():
    """MODEL-06: TrajectoryTransformer(input_dim=52) has conv.in_channels == 52."""
    TrajectoryTransformer = _import_model()
    model = TrajectoryTransformer(input_dim=52)
    assert model.conv.in_channels == 52, (
        f"Expected conv.in_channels=52, got {model.conv.in_channels}"
    )


# ---------------------------------------------------------------------------
# SC-4: Padded positions receive near-zero attention weight
# ---------------------------------------------------------------------------

def test_padding_mask_attention():
    """SC-4: Padded token columns (frames 10-24) get mean attention < 0.05.

    Uses AttentionCapturingEncoderLayer (subclass of TransformerEncoderLayer
    that overrides _sa_block to capture weights).  The model must expose this
    layer on its encoder so tests can read last_attn_weights.
    """
    from src.model.trajectory_model import TrajectoryTransformer, AttentionCapturingEncoderLayer  # noqa

    # Build model with attention-capturing layer in the first encoder position.
    model = TrajectoryTransformer(input_dim=50, capture_attention=True)

    x = torch.randn(1, 25, 50)
    mask = torch.ones(1, 25, dtype=torch.bool)
    mask[0, 10:] = False  # frames 10-24 are padded

    model.eval()
    with torch.no_grad():
        _ = model(x, mask)

    # Retrieve attention weights from the first encoder layer.
    first_layer = model.encoder.layers[0]
    assert hasattr(first_layer, "last_attn_weights"), (
        "First encoder layer must expose last_attn_weights after forward pass"
    )
    attn = first_layer.last_attn_weights  # (1, T, T) — averaged over heads
    assert attn is not None, "last_attn_weights should not be None after forward"

    # Columns 10-24 correspond to padded key positions.
    # Mean attention weight on those columns must be near-zero.
    padded_col_mean = attn[:, :, 10:].mean().item()
    assert padded_col_mean < 0.05, (
        f"Expected near-zero attention on padded columns, got {padded_col_mean:.4f}"
    )
