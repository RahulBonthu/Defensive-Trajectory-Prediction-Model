# Phase 4: Model Training and Ablation Evaluation - Research

**Researched:** 2026-03-13
**Domain:** PyTorch training loop, wandb logging, multi-seed ablation, statistical significance testing
**Confidence:** HIGH

---

## Summary

Phase 4 is entirely an execution phase. All infrastructure is in place: the model class (`TrajectoryTransformer`), loss function (`rmse_loss`), device selector (`get_device`), and datasets (`DefensiveTrajectoryDataset`) are fully implemented and tested across 22 unit tests. The overfit test confirms architecture and optimizer wiring are correct for both model variants (Model A: 95.9% loss reduction, Model B: 98.4% over 200 epochs with dropout=0.0).

The training task is to wire a standard PyTorch training loop to the existing components, run both model variants under identical conditions across 3-5 seeds, and produce the ablation comparison table with statistical significance. All dependencies (scipy, wandb, tqdm) are already declared in pyproject.toml. No new libraries need to be installed. The statistical test (paired t-test or Wilcoxon signed-rank) operates on per-play RMSE vectors from the test set, which the evaluation script must collect.

The per-position subgroup analysis (CB, FS, SS, LB) is straightforward because `DefensiveTrajectoryDataset.__getitem__` already returns `position` in the output dict, so filtering test predictions by position requires no additional data joins.

**Primary recommendation:** Write one `train_model.py` script and one `evaluate_ablation.py` script. Do not combine them — training is slow and must be run once per seed; evaluation is fast and can be re-run without retraining.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | Both Model A and Model B trained with identical architecture hyperparameters and same train/val/test split | splits.json enforces game-week split; identical `TrajectoryTransformer` constructor except `input_dim`; same DataLoader seed |
| TRAIN-02 | RMSE (root mean square error in yards) as training loss | `rmse_loss` already implemented in `trajectory_model.py` — import and use directly |
| TRAIN-03 | Training runs logged to wandb with full config | wandb is in pyproject.toml; use `wandb.init(config=...)` at start of each run; log `train_loss` and `val_loss` per epoch |
| TRAIN-04 | Both models saved as checkpoints after training | `torch.save(model.state_dict(), "models/model_a_best.pt")` when val loss improves; save once per seed with seed in filename, then copy best to canonical name |
| EVAL-01 | Per-play RMSE computed and stored for test set | Loop over test DataLoader; compute `rmse_loss` per sample (batch_size=1 or store per-element MSE then sqrt); attach position label from batch dict |
| EVAL-02 | Ablation comparison table: mean RMSE, std, delta across 3-5 seeds | Average per-seed test RMSE; compute delta = mean_A - mean_B; report std across seeds |
| EVAL-03 | Statistical significance test (paired t-test or Wilcoxon) on per-play RMSE differences | `scipy.stats.ttest_rel` or `scipy.stats.wilcoxon` on aligned per-play RMSE vectors; p-value reported |
| EVAL-04 | Per-position subgroup RMSE reported separately for CB, FS, SS, LB | Filter per-play predictions by `position` key from dataset output dict; compute mean RMSE per group |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.10.0 | Training loop, checkpointing, DataLoader | Already in use; model and dataset depend on it |
| wandb | latest (in pyproject.toml) | Experiment logging, config tracking, run comparison | Explicitly required by TRAIN-03 and ROADMAP |
| scipy | >=1.14 | `ttest_rel`, `wilcoxon` for EVAL-03 | Already in pyproject.toml; standard for scientific stats |
| tqdm | >=4.0 | Epoch and batch progress bars | Already in pyproject.toml; low-overhead progress display |
| numpy | ==2.4 | Per-play RMSE accumulation, ablation table math | Already in use across project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | ==3.0.1 | Ablation results table, CSV export | Writing final ablation table to disk for Phase 5 consumption |
| json / pathlib | stdlib | Loading splits.json, managing checkpoint paths | Config and file I/O in training script |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual training loop | PyTorch Lightning | Lightning adds abstraction overhead; project is small enough that a plain loop is clearer and matches the overfit_test.py pattern already established |
| `ttest_rel` (paired t-test) | `wilcoxon` | Both are already in REQUIREMENTS.md as acceptable; paired t-test is simpler to interpret; Wilcoxon is more robust to non-normal RMSE distributions — use Wilcoxon as primary, t-test as secondary |
| wandb | MLflow / TensorBoard | wandb is explicitly required by TRAIN-03; do not substitute |

**Installation:** All dependencies already installed. No additional packages required.

---

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── train_model.py           # Training loop for one model variant + one seed
├── run_training.py          # Orchestration: call train_model.py for A and B across seeds
├── evaluate_ablation.py     # Load checkpoints, run test set, produce ablation table
models/
├── model_a_seed{N}_best.pt  # Per-seed checkpoints during training
├── model_b_seed{N}_best.pt
├── model_a_best.pt          # Canonical best checkpoint (lowest val loss across seeds)
├── model_b_best.pt
results/
├── per_play_rmse_a.csv      # Per-play RMSE for Model A (with position column)
├── per_play_rmse_b.csv      # Per-play RMSE for Model B (with position column)
├── ablation_table.csv       # Aggregated ablation comparison table
```

### Pattern 1: Standard PyTorch Training Loop
**What:** Epoch loop with train/val phases, best-model checkpoint on val loss improvement
**When to use:** Always — this is the only training pattern needed
**Example:**
```python
# Follows overfit_test.py pattern already established in Phase 3
from src.model.trajectory_model import TrajectoryTransformer, rmse_loss, get_device

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch["input"].to(device), batch["padding_mask"].to(device))
        loss = rmse_loss(pred, batch["target_xy"].to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["input"].to(device), batch["padding_mask"].to(device))
            val_losses.append(rmse_loss(pred, batch["target_xy"].to(device)).item())
    val_loss = float(np.mean(val_losses))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)

    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
```

### Pattern 2: Multi-Seed Training
**What:** Run identical training for both models across 3-5 independent seeds; collect per-seed test RMSE
**When to use:** Required for EVAL-02 and EVAL-03 (ablation table and significance test)
**Example:**
```python
SEEDS = [42, 123, 456, 789, 1337]  # 5 seeds
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Re-instantiate model and DataLoader with this seed's shuffle order
    # Train, evaluate on test set, store per-play RMSE vectors
```

### Pattern 3: Per-Play RMSE Collection for Statistical Test
**What:** Collect one RMSE value per test sample (not mean over batch); keep aligned vectors for Model A and B
**When to use:** Required for EVAL-01 (store per-play) and EVAL-03 (paired test requires aligned vectors)
**Example:**
```python
# Compute per-sample MSE, then sqrt each element
per_play_rmse = []
positions = []
with torch.no_grad():
    for batch in test_loader:
        pred = model(batch["input"].to(device), batch["padding_mask"].to(device))
        target = batch["target_xy"].to(device)
        # Per-sample MSE then sqrt — NOT mean over batch before sqrt
        mse_per_sample = ((pred - target) ** 2).mean(dim=1)  # (batch,)
        rmse_per_sample = torch.sqrt(mse_per_sample).cpu().numpy()
        per_play_rmse.extend(rmse_per_sample.tolist())
        positions.extend(batch["position"])  # list of str from dataset
```

### Pattern 4: wandb Run Configuration
**What:** Log full config at run start so each run is reproducible and comparable
**When to use:** Every training run (TRAIN-03)
**Example:**
```python
wandb.init(
    project="defensive-trajectory-prediction",
    name=f"model_{variant}_seed{seed}",
    config={
        "model_variant": variant,       # "A" or "B"
        "input_dim": input_dim,         # 50 or 52
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "seed": seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }
)
```

### Pattern 5: Statistical Significance Test
**What:** Paired Wilcoxon signed-rank test on per-play RMSE differences (aligned by test sample index)
**When to use:** Required for EVAL-03; use per-play vectors from a single representative seed or averaged
**Example:**
```python
from scipy.stats import wilcoxon, ttest_rel

# rmse_a and rmse_b are aligned arrays of length n_test_samples
stat, p_value = wilcoxon(rmse_a, rmse_b, alternative="greater")
# Interpretation: p < 0.05 means Model A is significantly WORSE than Model B
# Also run t-test for comparison:
t_stat, t_p = ttest_rel(rmse_a, rmse_b)
```

### Anti-Patterns to Avoid
- **Computing RMSE over the whole batch before per-play collection:** `rmse_loss(pred, target).item()` gives batch mean. For EVAL-01 and EVAL-03 you need per-sample values. Use `((pred - target) ** 2).mean(dim=1).sqrt()` to get a vector of length batch_size.
- **Using shuffle=True in the test DataLoader:** Test evaluation must be deterministic. Use `shuffle=False` for val and test loaders.
- **Using a single seed and calling it multi-seed:** The ablation table requires 3-5 independent seeds. Training must be re-run from scratch for each seed — do not just re-evaluate the same checkpoint.
- **Saving only the final checkpoint:** Use best-val-loss checkpointing. The final epoch may have overfit relative to the best val epoch.
- **Misaligning Model A and Model B per-play RMSE vectors:** The paired test requires that `rmse_a[i]` and `rmse_b[i]` correspond to the same test play. Ensure both models evaluate the same test DataLoader with `shuffle=False` and the same iteration order.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical significance | Custom permutation test | `scipy.stats.wilcoxon` or `ttest_rel` | Already in pyproject.toml; handles ties, edge cases, and provides exact p-values |
| Experiment tracking | Print-to-file logging | `wandb` | Required by TRAIN-03; wandb handles time series, config diffs, and run comparison |
| Progress display | Manual `print` in epoch loop | `tqdm` | Already in pyproject.toml; handles nested loops and ETA correctly |
| DataLoader worker count | Hard-coded `num_workers=0` | Tune based on platform | MPS (Apple Silicon) requires `num_workers=0`; CUDA supports >0. Use `get_device()` to branch |
| Checkpoint naming | Custom hash/timestamp scheme | `model_a_seed{N}_best.pt` convention | Simple, readable, directly satisfies TRAIN-04 artifact contract |

**Key insight:** scipy, wandb, and tqdm are already installed. The only new code needed is the training loop, evaluation loop, and file I/O glue — all proven patterns.

---

## Common Pitfalls

### Pitfall 1: MPS Fallback Not Set
**What goes wrong:** On Apple Silicon, certain PyTorch ops are not natively supported by MPS and silently fall back or raise errors.
**Why it happens:** `overfit_test.py` already sets `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`. Training scripts must do the same.
**How to avoid:** Add `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` before any PyTorch imports in training scripts.
**Warning signs:** Runtime errors mentioning unsupported MPS ops, or silently wrong outputs when device is MPS.

### Pitfall 2: DataLoader OOM with num_workers > 0 and MPS
**What goes wrong:** On macOS/MPS, multiprocess DataLoader workers can cause memory issues or hangs.
**Why it happens:** MPS memory is shared; forking workers can cause conflicts.
**How to avoid:** Use `num_workers=0` when device is MPS. This is safe because the dataset pre-builds its context index in `__init__` (established in Phase 2).
**Warning signs:** Training hangs or crashes after first epoch on Apple Silicon.

### Pitfall 3: Ablation Vectors Not Aligned
**What goes wrong:** Per-play RMSE for Model A and Model B reference different test plays, making the paired Wilcoxon test invalid.
**Why it happens:** If test DataLoader uses `shuffle=True` or different random seeds for DataLoader initialization, iteration order differs.
**How to avoid:** Create test DataLoaders with `shuffle=False` and the same `generator` seed (or simply no generator argument), ensuring both models iterate the same test set in the same order.
**Warning signs:** Paired Wilcoxon p-value is unusually low or high compared to intuition from mean RMSE difference.

### Pitfall 4: Per-Play RMSE vs Batch-Mean RMSE Confusion
**What goes wrong:** Using `rmse_loss(pred, target)` directly during evaluation records batch-mean RMSE, not per-play RMSE. EVAL-01 requires per-play values.
**Why it happens:** `rmse_loss` calls `F.mse_loss(pred, target)` which defaults to `reduction="mean"` over all elements.
**How to avoid:** During evaluation, compute `((pred - target) ** 2).mean(dim=-1).sqrt()` to get a (batch,) tensor of per-sample RMSE values.
**Warning signs:** Per-play CSV has fewer rows than expected (one per batch instead of one per sample).

### Pitfall 5: wandb Authentication in a New Environment
**What goes wrong:** First `wandb.init()` call prompts for API key interactively, blocking a script run.
**Why it happens:** wandb requires authentication on first use.
**How to avoid:** Run `wandb login` once in the project environment before executing training scripts. Alternatively, set `WANDB_API_KEY` environment variable.
**Warning signs:** Training script hangs at `wandb.init()` waiting for terminal input.

### Pitfall 6: Model A/B Input Dimension Mismatch at Checkpoint Load
**What goes wrong:** Loading `model_b_best.pt` into a `TrajectoryTransformer(input_dim=50)` instance raises a shape error.
**Why it happens:** `conv.weight` shape differs between input_dim=50 and input_dim=52.
**How to avoid:** Always pair checkpoint paths with the correct `input_dim` argument. Store variant name alongside checkpoint file.

---

## Code Examples

Verified patterns from the existing codebase:

### Instantiating Both Model Variants
```python
# Source: src/model/trajectory_model.py
from src.model.trajectory_model import TrajectoryTransformer, rmse_loss, get_device

device = get_device()

model_a = TrajectoryTransformer(
    input_dim=50, d_model=64, nhead=4, num_layers=2, dropout=0.1
).to(device)

model_b = TrajectoryTransformer(
    input_dim=52, d_model=64, nhead=4, num_layers=2, dropout=0.1
).to(device)
```

### Building DataLoaders from Existing Dataset
```python
# Source: src/data/dataset.py — DefensiveTrajectoryDataset interface
from src.data.dataset import DefensiveTrajectoryDataset
from torch.utils.data import DataLoader
import json, pathlib

splits = json.loads(pathlib.Path("data/splits.json").read_text())
# samples = output from build_samples() — already produced by Phase 1
# cleaned_df = pd.read_parquet("data/cleaned.parquet")

train_samples = [s for s in all_samples if s["gameId"] in splits["train_game_ids"]]
val_samples   = [s for s in all_samples if s["gameId"] in splits["val_game_ids"]]
test_samples  = [s for s in all_samples if s["gameId"] in splits["test_game_ids"]]

train_ds_a = DefensiveTrajectoryDataset(train_samples, cleaned_df, include_ball_destination=False)
train_ds_b = DefensiveTrajectoryDataset(train_samples, cleaned_df, include_ball_destination=True)
# val/test analogously

# num_workers=0 on MPS; can increase on CUDA
train_loader_a = DataLoader(train_ds_a, batch_size=256, shuffle=True,  num_workers=0)
val_loader_a   = DataLoader(val_ds_a,   batch_size=256, shuffle=False, num_workers=0)
test_loader_a  = DataLoader(test_ds_a,  batch_size=256, shuffle=False, num_workers=0)
```

### Saving and Loading Best Checkpoint
```python
# Save on val improvement
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), checkpoint_path)

# Load for evaluation
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
```

### Ablation Table Construction
```python
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

# rmse_a_seeds and rmse_b_seeds: list of arrays, one per seed
mean_a = np.mean([r.mean() for r in rmse_a_seeds])
std_a  = np.std( [r.mean() for r in rmse_a_seeds])
mean_b = np.mean([r.mean() for r in rmse_b_seeds])
std_b  = np.std( [r.mean() for r in rmse_b_seeds])

# Paired test on one representative seed (e.g., median-performing seed)
rmse_a_rep = rmse_a_seeds[representative_seed_idx]
rmse_b_rep = rmse_b_seeds[representative_seed_idx]
_, p_wilcoxon = wilcoxon(rmse_a_rep, rmse_b_rep, alternative="greater")
_, p_ttest    = ttest_rel(rmse_a_rep, rmse_b_rep)

ablation = pd.DataFrame({
    "Model":    ["A (no ball dest)", "B (with ball dest)"],
    "Mean RMSE": [mean_a, mean_b],
    "Std RMSE":  [std_a,  std_b],
    "Delta (A-B)": [mean_a - mean_b, 0.0],
    "p (Wilcoxon)": [p_wilcoxon, None],
    "p (t-test)":   [p_ttest,    None],
})
ablation.to_csv("results/ablation_table.csv", index=False)
```

### Per-Position Subgroup RMSE
```python
# position is a list of strings in each batch dict from DefensiveTrajectoryDataset
position_rmse = {"CB": [], "FS": [], "SS": [], "LB": []}

with torch.no_grad():
    for batch in test_loader:
        pred   = model(batch["input"].to(device), batch["padding_mask"].to(device))
        target = batch["target_xy"].to(device)
        rmse_each = ((pred - target) ** 2).mean(dim=1).sqrt().cpu().numpy()

        for pos, rmse_val in zip(batch["position"], rmse_each):
            if pos in position_rmse:
                position_rmse[pos].append(float(rmse_val))

for pos, vals in position_rmse.items():
    print(f"{pos}: mean RMSE = {np.mean(vals):.4f} ({len(vals)} samples)")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual overfit loop (scripts/overfit_test.py) | Full training loop with DataLoader, wandb, checkpointing | Phase 4 | Adds validation, wandb, early stopping via checkpoint |
| Single training run | Multi-seed runs (3-5 seeds) | Phase 4 | Required for statistically valid ablation comparison |
| No statistical test | Paired Wilcoxon + t-test | Phase 4 | Satisfies EVAL-03; makes the core finding defensible |

---

## Open Questions

1. **Batch size for full training**
   - What we know: 52,779 train samples; overfit_test used batch_size=100 (full batch on 100 samples)
   - What's unclear: Whether 256 or 512 is better given available GPU/MPS memory
   - Recommendation: Start with batch_size=256; if OOM on MPS, drop to 128. Log batch_size to wandb config.

2. **Number of epochs and early stopping criterion**
   - What we know: Overfit test converges in 200 epochs on 100 samples with lr=1e-3
   - What's unclear: How many epochs are needed on 52K samples; whether learning rate decay improves convergence
   - Recommendation: Start with 50 epochs and `ReduceLROnPlateau(patience=5)`. Watch wandb val loss curve. Phase success criterion is "decreases then plateaus" not a specific epoch count.

3. **Which seed(s) to use for the paired Wilcoxon test**
   - What we know: EVAL-03 requires paired per-play RMSE differences; EVAL-02 requires 3-5 seeds for mean/std
   - What's unclear: Whether to pool across seeds or pick a representative seed for the paired test
   - Recommendation: Run the paired test on each seed separately; report the median p-value. This is more conservative and more defensible than pooling across seeds (which inflates sample size artificially).

4. **Where build_samples() output is loaded from**
   - What we know: cleaned.parquet exists; splits.json exists; DefensiveTrajectoryDataset takes raw sample dicts + context_df
   - What's unclear: Whether samples are re-built from the parquet at DataLoader construction time or cached to disk
   - Recommendation: Check if a `samples.pkl` or similar cache exists in `data/`. If not, re-run `build_samples()` at start of training script. It ran successfully in Phase 1 (01-04) in acceptable time.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | Both models trained with identical hyperparameters | smoke | `pytest tests/test_training.py::test_identical_hyperparameters -x` | Wave 0 |
| TRAIN-02 | RMSE loss used as training objective | unit | `pytest tests/test_training.py::test_rmse_loss_used -x` | Wave 0 |
| TRAIN-03 | wandb logs train/val loss per epoch | smoke | `pytest tests/test_training.py::test_wandb_logging -x` (mock wandb) | Wave 0 |
| TRAIN-04 | model_a_best.pt and model_b_best.pt saved after training | integration | `pytest tests/test_training.py::test_checkpoints_saved -x` | Wave 0 |
| EVAL-01 | Per-play RMSE computed and stored | unit | `pytest tests/test_evaluation.py::test_per_play_rmse_shape -x` | Wave 0 |
| EVAL-02 | Ablation table has mean, std, delta columns | unit | `pytest tests/test_evaluation.py::test_ablation_table_columns -x` | Wave 0 |
| EVAL-03 | p-value from Wilcoxon/t-test reported | unit | `pytest tests/test_evaluation.py::test_significance_test_runs -x` | Wave 0 |
| EVAL-04 | Per-position RMSE for CB, FS, SS, LB | unit | `pytest tests/test_evaluation.py::test_per_position_rmse_keys -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_training.py` — covers TRAIN-01 through TRAIN-04 (training loop unit/smoke tests using synthetic data and mock wandb)
- [ ] `tests/test_evaluation.py` — covers EVAL-01 through EVAL-04 (evaluation and ablation table unit tests using synthetic per-play RMSE vectors)

*(The existing 22 tests in `tests/test_pipeline.py`, `tests/test_dataset.py`, `tests/test_model.py` are unaffected by Phase 4 and serve as regression guard.)*

---

## Sources

### Primary (HIGH confidence)
- `src/model/trajectory_model.py` (lines 1-190) — confirmed API: `TrajectoryTransformer(input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1)`, `rmse_loss(pred, target)`, `get_device()`
- `src/data/dataset.py` (lines 1-248) — confirmed: `DefensiveTrajectoryDataset.__getitem__` returns `{"input": Tensor(T,50/52), "padding_mask": Tensor(T,), "target_xy": Tensor(2,), "position": str}`
- `pyproject.toml` — confirmed installed: scipy>=1.14, wandb, tqdm>=4.0, torch>=2.10.0
- `scripts/overfit_test.py` — confirmed training loop pattern: Adam lr=1e-3, `rmse_loss`, `torch.manual_seed`, device handling
- `.planning/phases/03-model-architecture-and-training-infrastructure/03-02-SUMMARY.md` — confirmed hyperparameters: d_model=64, nhead=4, num_layers=2, dropout=0.1 (note: trajectory_model.py default is d_model=128; overfit_test does not override d_model, so uses 128; TRAIN-01 must lock the specific hyperparameter values used)
- `.planning/REQUIREMENTS.md` — ground-truth requirement text for TRAIN-01 through EVAL-04
- `tests/conftest.py` — confirmed fixture patterns; sample schema; position labels available in dataset output

### Secondary (MEDIUM confidence)
- `.planning/STATE.md` — sample counts: train 52,779 / val 7,497 / test 8,068 (strict CB/FS/SS/LB filter applied)

### Tertiary (LOW confidence)
- Epoch count recommendation (50 epochs) — based on training data size analogy; must validate against val loss curve

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries confirmed present in pyproject.toml; API confirmed in source files
- Architecture: HIGH — training loop pattern directly mirrors verified overfit_test.py; no new patterns introduced
- Pitfalls: HIGH for MPS/DataLoader (observed in Phase 3 decisions); MEDIUM for wandb auth (common practice)
- Statistical testing: HIGH — scipy.stats.wilcoxon and ttest_rel are standard; requirement text explicitly names them

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (stable stack; PyTorch and scipy APIs change slowly)

---

## Hyperparameter Clarification Note

The `trajectory_model.py` default for `d_model` is 128 (not 64 as mentioned in the additional context). The additional context states "d_model=64" but the actual source code at line 131 shows `d_model: int = 128`. The overfit_test.py also does not override d_model, meaning it used d_model=128 for the 95.9%/98.4% overfit results. TRAIN-01 requires identical hyperparameters for both models — the planner should specify the actual values from the source, not the summary:

**Confirmed hyperparameters from `trajectory_model.py` defaults:**
- d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1
