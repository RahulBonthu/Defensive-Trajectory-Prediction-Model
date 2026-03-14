"""
evaluate_ablation.py — Phase 4 Plan 03.

Exports four pure-function evaluation utilities called by the plan 04-04
orchestration script after training to produce the ablation table and
statistical significance results.

Functions:
  collect_per_play_rmse        Per-sample RMSE collection from a DataLoader
  build_ablation_table         Summary DataFrame + CSV for two model variants
  run_significance_tests       Wilcoxon + paired t-test on aligned RMSE arrays
  compute_per_position_rmse    Mean RMSE broken out by defensive position

Helper:
  save_per_play_csv            Write per_play records to results/ as CSV
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pathlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, ttest_rel
from src.model.trajectory_model import TrajectoryTransformer, get_device

STRICT_POSITIONS = {"CB", "FS", "SS", "LB"}


# ---------------------------------------------------------------------------
# EVAL-01
# ---------------------------------------------------------------------------

def collect_per_play_rmse(
    model: TrajectoryTransformer,
    test_loader,
    device=None,
) -> list:
    """Collect per-sample RMSE from a DataLoader without batching the result.

    Parameters
    ----------
    model : TrajectoryTransformer
        Model to evaluate.
    test_loader : DataLoader
        Must be constructed with shuffle=False (caller's responsibility).
    device : torch.device | None
        Defaults to get_device() when None.

    Returns
    -------
    list[dict]
        One dict per test sample: {"rmse": float, "position": str}.
        Length equals len(test_loader.dataset).
    """
    if device is None:
        device = get_device()

    model.eval()
    per_play: list[dict] = []

    with torch.no_grad():
        for batch in test_loader:
            pred = model(
                batch["input"].to(device),
                batch["padding_mask"].to(device),
            )                                        # (batch, 2)
            target = batch["target_xy"].to(device)   # (batch, 2)

            # Per-sample RMSE — NOT rmse_loss() which returns batch mean scalar
            rmse_each = ((pred - target) ** 2).mean(dim=1).sqrt().cpu().numpy()  # (batch,)

            for pos, rmse_val in zip(batch["position"], rmse_each):
                per_play.append({"rmse": float(rmse_val), "position": pos})

    return per_play


# ---------------------------------------------------------------------------
# EVAL-02
# ---------------------------------------------------------------------------

def build_ablation_table(
    rmse_a_seeds: list,
    rmse_b_seeds: list,
) -> pd.DataFrame:
    """Build a summary DataFrame comparing two model variants across seeds.

    Uses the median-performing seed (by mean RMSE of Model A) for the paired
    significance tests. Saves results/ablation_table.csv as a side effect.

    Parameters
    ----------
    rmse_a_seeds : list[np.ndarray]
        One 1-D float array per seed for Model A.
    rmse_b_seeds : list[np.ndarray]
        One 1-D float array per seed for Model B (same order and length).

    Returns
    -------
    pd.DataFrame
        Columns: ["Model", "Mean RMSE", "Std RMSE", "Delta (A-B)",
                  "p (Wilcoxon)", "p (t-test)"]
    """
    # Aggregate across seeds
    mean_a = float(np.mean([r.mean() for r in rmse_a_seeds]))
    std_a  = float(np.std( [r.mean() for r in rmse_a_seeds])) if len(rmse_a_seeds) > 1 else 0.0
    mean_b = float(np.mean([r.mean() for r in rmse_b_seeds]))
    std_b  = float(np.std( [r.mean() for r in rmse_b_seeds])) if len(rmse_b_seeds) > 1 else 0.0

    # Select the median-performing seed index for paired significance tests
    if len(rmse_a_seeds) > 1:
        seed_means_a = [r.mean() for r in rmse_a_seeds]
        rep_idx = int(np.argsort(seed_means_a)[len(seed_means_a) // 2])
    else:
        rep_idx = 0

    rmse_a_rep = rmse_a_seeds[rep_idx]
    rmse_b_rep = rmse_b_seeds[rep_idx]

    sig = run_significance_tests(rmse_a_rep, rmse_b_rep)

    delta = mean_a - mean_b  # positive => Model A is worse => Model B helped

    df = pd.DataFrame([
        {
            "Model":       "Model A (no ball dest)",
            "Mean RMSE":   mean_a,
            "Std RMSE":    std_a,
            "Delta (A-B)": delta,
            "p (Wilcoxon)": sig["p_wilcoxon"],
            "p (t-test)":  sig["p_ttest"],
        },
        {
            "Model":       "Model B (with ball dest)",
            "Mean RMSE":   mean_b,
            "Std RMSE":    std_b,
            "Delta (A-B)": -delta,
            "p (Wilcoxon)": sig["p_wilcoxon"],
            "p (t-test)":  sig["p_ttest"],
        },
    ])

    # Save to results/
    pathlib.Path("results").mkdir(exist_ok=True)
    df.to_csv("results/ablation_table.csv", index=False)

    return df


# ---------------------------------------------------------------------------
# EVAL-03
# ---------------------------------------------------------------------------

def run_significance_tests(
    rmse_a: np.ndarray,
    rmse_b: np.ndarray,
) -> dict:
    """Run Wilcoxon signed-rank test and paired t-test on aligned RMSE arrays.

    Parameters
    ----------
    rmse_a : np.ndarray
        Per-play RMSE for Model A (1-D, shape (N,)).
    rmse_b : np.ndarray
        Per-play RMSE for Model B (1-D, same N and same play order).

    Returns
    -------
    dict
        {"stat_wilcoxon": float, "p_wilcoxon": float,
         "stat_ttest": float,    "p_ttest": float}

    Notes
    -----
    Wilcoxon alternative="greater": H1 is that Model A RMSE > Model B RMSE
    (i.e., Model B is better).  When all differences are zero the test raises
    ValueError — caught here, returning stat=0.0 and p=1.0.
    """
    try:
        stat_w, p_w = wilcoxon(rmse_a, rmse_b, alternative="greater")
    except ValueError:
        # All differences are zero — no signal, treat as non-significant
        stat_w, p_w = 0.0, 1.0

    stat_t, p_t = ttest_rel(rmse_a, rmse_b)

    return {
        "stat_wilcoxon": float(stat_w),
        "p_wilcoxon":    float(p_w),
        "stat_ttest":    float(stat_t),
        "p_ttest":       float(p_t),
    }


# ---------------------------------------------------------------------------
# EVAL-04
# ---------------------------------------------------------------------------

def compute_per_position_rmse(
    per_play: list,
) -> dict:
    """Compute mean RMSE broken out by defensive position.

    Parameters
    ----------
    per_play : list[dict]
        Each dict must have at least "rmse" (float) and "position" (str).

    Returns
    -------
    dict[str, float]
        Keys are exactly {"CB", "FS", "SS", "LB"}.  Positions with no
        samples receive float("nan").
    """
    buckets: dict[str, list] = {pos: [] for pos in STRICT_POSITIONS}

    for item in per_play:
        pos = item["position"]
        if pos in buckets:
            buckets[pos].append(item["rmse"])

    result: dict[str, float] = {}
    for pos in STRICT_POSITIONS:
        vals = buckets[pos]
        result[pos] = float(np.mean(vals)) if vals else float("nan")

    return result


# ---------------------------------------------------------------------------
# Helper: save per-play CSV
# ---------------------------------------------------------------------------

def save_per_play_csv(
    per_play: list,
    model_variant: str,
    seed: int,
    output_dir: str = "results",
) -> str:
    """Write per-play RMSE records to a CSV file.

    Called by the plan 04-04 orchestration script; not covered by unit tests.

    Parameters
    ----------
    per_play : list[dict]
        Records as returned by collect_per_play_rmse, possibly augmented with
        a "seed" key by the caller.
    model_variant : str
        Short label, e.g. "A" or "B". Used in the filename.
    seed : int
        Random seed used for this run.
    output_dir : str
        Directory in which to write the file. Created if it does not exist.

    Returns
    -------
    str
        Absolute path to the written CSV.
    """
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"per_play_rmse_{model_variant.lower()}_seed{seed}.csv"
    out_path = out_dir / fname

    records = [
        {"play_idx": i, "rmse": d["rmse"], "position": d["position"], "seed": seed}
        for i, d in enumerate(per_play)
    ]
    pd.DataFrame(records).to_csv(out_path, index=False)
    return str(out_path.resolve())
