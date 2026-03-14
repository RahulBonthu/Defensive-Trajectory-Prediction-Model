"""
Overfit verification script — Phase 3 SC-2.

Verifies that TrajectoryTransformer can memorize a tiny fixed dataset of
100 synthetic samples in 200 epochs, confirming the architecture and
optimizer are correctly wired before full training (Phase 4).

Expected result: final_loss < initial_loss * 0.5 for both Model A and Model B.

Usage (from project root):
    python scripts/overfit_test.py

Note: This script will fail with ImportError until trajectory_model.py is
implemented in Plan 03-02. That is the expected RED state for Plan 03-01.
"""

import os
import torch
import torch.nn.functional as F

# This import fails (ImportError) until Plan 03-02 creates trajectory_model.py.
from src.model.trajectory_model import TrajectoryTransformer  # noqa: E402

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> torch.device:
    """Select best available device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSE loss in same units as target (yards).

    Uses eps to guard against NaN gradient when loss approaches zero.
    """
    return torch.sqrt(F.mse_loss(pred, target) + eps)


def run_overfit(input_dim: int, device: torch.device, n_samples: int = 100, n_epochs: int = 200) -> list[float]:
    """Run overfit test for a single input_dim variant.

    Args:
        input_dim: 50 for Model A, 52 for Model B.
        device: torch.device to run on.
        n_samples: Number of synthetic training samples.
        n_epochs: Number of training epochs.

    Returns:
        List of per-epoch RMSE loss values.

    Raises:
        AssertionError: If final_loss >= initial_loss * 0.5.
    """
    T = 25

    # Build model with dropout=0.0 to allow clean memorization.
    model = TrajectoryTransformer(input_dim=input_dim, dropout=0.0).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic fixed dataset — same seed for reproducibility.
    torch.manual_seed(42)
    x = torch.randn(n_samples, T, input_dim, device=device)
    mask = torch.ones(n_samples, T, dtype=torch.bool, device=device)
    target = torch.randn(n_samples, 2, device=device)

    losses: list[float] = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(x, mask)
        loss = rmse_loss(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  [input_dim={input_dim}] epoch {epoch:>3d}  loss={loss.item():.6f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    threshold = initial_loss * 0.5

    print(
        f"\n  [input_dim={input_dim}] initial={initial_loss:.6f}  "
        f"final={final_loss:.6f}  threshold={threshold:.6f}"
    )

    assert final_loss < threshold, (
        f"Overfit test FAILED for input_dim={input_dim}: "
        f"final_loss={final_loss:.6f} >= threshold={threshold:.6f} "
        f"(initial_loss * 0.5)"
    )

    print(f"  [input_dim={input_dim}] PASSED — loss reduced by "
          f"{100 * (1 - final_loss / initial_loss):.1f}%")
    return losses


def main() -> None:
    device = get_device()
    print(f"Device selected: {device}")
    print()

    print("=== Overfit test: Model A (input_dim=50) ===")
    run_overfit(input_dim=50, device=device)
    print()

    print("=== Overfit test: Model B (input_dim=52) ===")
    run_overfit(input_dim=52, device=device)
    print()

    print("All overfit tests PASSED.")


if __name__ == "__main__":
    main()
