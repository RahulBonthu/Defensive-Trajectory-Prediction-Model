"""
TrajectoryTransformer — Phase 3 Plan 02.

Exports:
  - TrajectoryTransformer   Shared model class for Model A (input_dim=50) and
                            Model B (input_dim=52).
  - AttentionCapturingEncoderLayer  TransformerEncoderLayer subclass that stores
                                    per-batch attention weights after each forward
                                    pass for inspection / test verification (SC-4).
  - rmse_loss               RMSE loss with eps guard to prevent NaN gradients.
  - get_device              Three-way device selector: CUDA -> MPS -> CPU.

Architecture overview:
  Input (batch, T, input_dim)
    -> Conv1d(input_dim, d_model, kernel_size=3, padding=1)   [T preserved]
    -> ReLU
    -> TransformerEncoder(d_model, nhead, num_layers)          [with padding mask]
    -> masked mean-pool over T dimension
    -> Linear(d_model, 2)
  Output (batch, 2)

Padding mask convention (matches src/data/dataset.py):
  True  = real frame   (attend)
  False = padded frame (ignore)

  The PyTorch TransformerEncoder uses the OPPOSITE convention for
  src_key_padding_mask: True = IGNORE.  The forward method inverts the mask
  before passing it to the encoder:
      src_key_padding_mask = ~padding_mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device selector
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select best available device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSE loss in the same units as target (yards).

    Uses eps to guard against NaN gradient when loss approaches zero.
    """
    return torch.sqrt(F.mse_loss(pred, target) + eps)


# ---------------------------------------------------------------------------
# Attention-capturing encoder layer (SC-4)
# ---------------------------------------------------------------------------

class AttentionCapturingEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer subclass that stores attention weights.

    Overrides forward() rather than _sa_block() because PyTorch 2.x uses a
    C++ fast-path in eval mode that bypasses _sa_block entirely.  The
    override calls self_attn directly with need_weights=True to capture the
    averaged attention map, then follows the standard post-attention
    residual + norm + feed-forward path.

    last_attn_weights: (batch, T, T) — per-head-averaged weights, detached.
    Set after every forward call; None only before the first call.

    The **kwargs in forward absorbs 'is_causal' and any future arguments.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with attention weight capture — bypasses C++ fast-path."""
        x_sa, weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_attn_weights = weights.detach()  # (batch, T, T)

        # Standard post-attention path (pre-norm=False, the default)
        x = self.norm1(src + self.dropout1(x_sa))
        x = self.norm2(x + self._ff_block(x))
        return x


# ---------------------------------------------------------------------------
# Trajectory Transformer
# ---------------------------------------------------------------------------

class TrajectoryTransformer(nn.Module):
    """Shared transformer backbone for Model A (input_dim=50) and B (input_dim=52).

    Args:
        input_dim:        Feature dimension per time step.  50 for Model A, 52 for B.
        d_model:          Internal transformer dimension (must be divisible by nhead).
        nhead:            Number of attention heads.
        num_layers:       Number of TransformerEncoderLayer stacks.
        dim_feedforward:  Feed-forward hidden dimension inside each encoder layer.
        dropout:          Dropout probability.  Pass 0.0 for overfit/ablation tests.
        capture_attention: If True, replace all encoder layers with
                           AttentionCapturingEncoderLayer so that
                           encoder.layers[0].last_attn_weights is available
                           after each forward pass (used by SC-4 test).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        capture_attention: bool = False,
    ) -> None:
        super().__init__()

        # Conv1d projection: input_dim -> d_model, preserves T with padding=1
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)

        # Choose encoder layer class
        layer_cls = AttentionCapturingEncoderLayer if capture_attention else nn.TransformerEncoderLayer

        encoder_layer = layer_cls(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
        )

        # enable_nested_tensor=False: disables the fast-path so that the
        # _sa_block override in AttentionCapturingEncoderLayer actually fires.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Regression head: d_model -> 2 (predicted x, y coordinates)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:            (batch, T, input_dim) — sequence of feature vectors.
            padding_mask: (batch, T) bool — True = real frame, False = padded.

        Returns:
            (batch, 2) predicted landing coordinates.
        """
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)      # (batch, input_dim, T)
        x = self.conv(x)             # (batch, d_model, T)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)      # (batch, T, d_model)

        # Invert mask: dataset True=real -> transformer True=ignore
        src_key_padding_mask = ~padding_mask

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pooling — only average over real (non-padded) frames
        mask_f = padding_mask.unsqueeze(-1).float()                    # (batch, T, 1)
        x_pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (batch, d_model)

        return self.head(x_pooled)   # (batch, 2)
