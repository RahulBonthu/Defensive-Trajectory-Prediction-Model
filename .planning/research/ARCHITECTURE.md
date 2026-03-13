# Architecture Research

**Domain:** Transformer-based sports player trajectory prediction (NFL tracking data)
**Researched:** 2026-03-13
**Confidence:** MEDIUM — based on training knowledge of trajectory prediction literature (Social Transformer, Trajectron++, GRIP, SoPhie) plus project-specific reasoning; WebSearch unavailable for verification

---

## Standard Architecture

### System Overview

The system is a research ML pipeline — not a service. It runs offline in four stages: data ingestion, feature engineering, model training, and evaluation/visualization. The two models (A and B) share all stages except the feature construction step where ball landing location is added or withheld.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA LAYER                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  NFL Big Data Bowl CSVs                                       │   │
│  │  tracking_*.csv  |  plays.csv  |  players.csv  |  games.csv  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE LAYER                             │
│                                                                      │
│  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐  │
│  │  Raw Loader     │   │  Preprocessor   │   │  Sample Builder  │  │
│  │  (CSV → pandas) │──▶│  (normalize,    │──▶│  (play → per-    │  │
│  │                 │   │   interpolate,  │   │   player sample) │  │
│  │                 │   │   flip, filter) │   │                  │  │
│  └─────────────────┘   └─────────────────┘   └────────┬─────────┘  │
└──────────────────────────────────────────────────────┬─┘────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                          │
│                                                                      │
│  ┌──────────────────────────┐   ┌──────────────────────────────┐   │
│  │  Base Feature Builder    │   │  Social Context Builder      │   │
│  │  (per-player kinematics: │   │  (all 22 player positions     │   │
│  │   x,y,speed,dir,accel,  │   │   at each frame as context   │   │
│  │   orientation)           │   │   tokens or flattened vec)   │   │
│  └────────────┬─────────────┘   └──────────────┬───────────────┘   │
│               │                                │                    │
│               └────────────────┬───────────────┘                    │
│                                ▼                                     │
│           ┌────────────────────────────────────┐                    │
│           │  Ablation Split                    │                    │
│           │  Model A input: kinematics + social│                    │
│           │  Model B input: kinematics + social│                    │
│           │                + ball_landing (x,y)│                    │
│           └──────────────┬─────────────────────┘                    │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Shared Architecture (instantiated twice)                     │  │
│  │                                                               │  │
│  │   Input Sequence (T frames x F features)                      │  │
│  │          │                                                    │  │
│  │          ▼                                                    │  │
│  │   1D Conv Layer  ──  extract local trajectory dynamics        │  │
│  │          │                                                    │  │
│  │          ▼                                                    │  │
│  │   Positional Encoding                                         │  │
│  │          │                                                    │  │
│  │          ▼                                                    │  │
│  │   Transformer Encoder  ──  N layers of self-attention         │  │
│  │   (attends over time steps; social context either             │  │
│  │    concatenated per-frame or as separate tokens)              │  │
│  │          │                                                    │  │
│  │          ▼                                                    │  │
│  │   Pooling / CLS token                                         │  │
│  │          │                                                    │  │
│  │          ▼                                                    │  │
│  │   Linear Head  →  predicted (x, y) at play end               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│   [Model A: F features = kinematics + social, no ball destination]  │
│   [Model B: F features = kinematics + social + ball_landing (x,y)] │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                                  │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  DataLoader      │  │  Trainer         │  │  Checkpoint      │  │
│  │  (train/val/test │  │  (loss=MSE,      │  │  (save best val  │  │
│  │   split, batch,  │  │   optimizer,     │  │   model weights) │  │
│  │   shuffle)       │  │   scheduler,     │  │                  │  │
│  │                  │  │   early stop)    │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│   Trained separately: model_a.pt   and   model_b.pt                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EVALUATION LAYER                                 │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  RMSE Calculator │  │  Error Distrib.  │  │  Breakdown by    │  │
│  │  (yards between  │  │  (histogram,     │  │  Position        │  │
│  │   pred vs actual │  │   CDF per model) │  │  CB / S / LB     │  │
│  │   ending pos)    │  │                  │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   VISUALIZATION LAYER                                │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  RMSE Comparison │  │  Trajectory      │  │  Attention       │  │
│  │  Bar / Box plot  │  │  Overlay Plot    │  │  Weight Heatmap  │  │
│  │  (A vs B)        │  │  (pred vs actual │  │  (optional, for  │  │
│  │                  │  │   on field)      │  │   poster)        │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | Responsibility | Notes |
|-----------|---------------|-------|
| Raw Loader | Read CSV files from disk into DataFrames | Joins tracking + plays + players on gameId/playId/nflId |
| Preprocessor | Normalize coordinates to LOS-relative; flip direction; interpolate missing frames; compute acceleration | Single pass over raw data; output saved to disk to avoid re-running |
| Sample Builder | Convert each (player, play) pair into a fixed-length sequence tensor | One sample = one defensive player on one play; produces (T, F) tensors |
| Base Feature Builder | Build per-frame kinematic feature vector for the target player | (x, y, speed, direction, orientation, acceleration) — 6 dims |
| Social Context Builder | Encode all other 22 players' (x,y) at each frame as social input | Approach: flatten 22x2=44 dims appended per frame, OR separate social tokens |
| Ablation Split | Add or withhold ball_landing (x, y) to produce Model A vs Model B input | Single flag at dataset construction time |
| 1D Conv Layer | Extract local temporal patterns from the input sequence | Kernel size 3–5; captures instantaneous trajectory shape before attention |
| Transformer Encoder | Self-attention over T time steps; learns which frames matter most | Standard PyTorch TransformerEncoderLayer; 2–4 layers sufficient for this task |
| Linear Head | Map pooled representation → predicted (x, y) | 2-dim regression output; trained with MSELoss |
| Trainer | Manage training loop, validation, early stopping, checkpoint saving | One trainer instance per model (A and B), identical hyperparameters |
| RMSE Calculator | Compute Euclidean distance error in yards on held-out test set | Primary metric: mean RMSE across all (player, play) samples |
| Visualization | Produce poster-quality figures comparing A vs B | matplotlib/seaborn; field overlay needs coordinate rescaling |

---

## Recommended Project Structure

```
defensive-trajectory/
├── data/
│   ├── raw/                    # Original Big Data Bowl CSVs (not committed)
│   └── processed/              # Preprocessed tensors / parquet files
│
├── src/
│   ├── data/
│   │   ├── loader.py           # CSV ingestion, join logic
│   │   ├── preprocessor.py     # Normalization, interpolation, flip
│   │   ├── sample_builder.py   # (player, play) → (T, F) tensor
│   │   ├── features.py         # Kinematic + social feature constructors
│   │   └── dataset.py          # PyTorch Dataset / DataLoader wrappers
│   │
│   ├── models/
│   │   ├── conv_encoder.py     # 1D conv temporal extractor
│   │   ├── transformer.py      # TransformerEncoder wrapper
│   │   ├── head.py             # Linear regression head
│   │   └── trajectory_model.py # Full model assembly (conv → transformer → head)
│   │
│   ├── training/
│   │   ├── trainer.py          # Train loop, val loop, early stopping
│   │   ├── config.py           # Hyperparameter dataclass (shared A and B)
│   │   └── scheduler.py        # LR scheduler logic
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # RMSE, per-position breakdown
│   │   └── ablation.py         # Side-by-side A vs B comparison logic
│   │
│   └── visualization/
│       ├── rmse_plots.py       # Bar/box/violin plots of error distributions
│       ├── trajectory_viz.py   # Field overlay: predicted vs actual paths
│       └── attention_viz.py    # (optional) attention weight heatmaps
│
├── scripts/
│   ├── preprocess.py           # Run preprocessing pipeline end-to-end
│   ├── train_model_a.py        # Train baseline (no ball destination)
│   ├── train_model_b.py        # Train directed model (with ball destination)
│   └── evaluate.py             # Run evaluation and generate figures
│
├── notebooks/
│   └── exploration.ipynb       # EDA, sanity checks on data
│
├── checkpoints/
│   ├── model_a_best.pt
│   └── model_b_best.pt
│
├── outputs/
│   └── figures/                # Generated plots for poster
│
├── requirements.txt
└── README.md
```

### Structure Rationale

- **src/data/:** Separating loader, preprocessor, and sample builder makes each step independently testable and cacheable. Processed data written to disk so preprocessing runs once.
- **src/models/:** Conv encoder, transformer, and head are separate modules so any one can be swapped or ablated without touching the others.
- **scripts/:** Executable entry points for each pipeline stage. Keeps notebooks clean for exploration and puts reproducible runs in scripts.
- **Two train scripts (not one with a flag):** Mirrors the project decision to train two separate models. Makes it trivially clear what each model saw during training.

---

## Architectural Patterns

### Pattern 1: Conv-then-Transformer for Time Series

**What:** Apply 1D convolution across the time dimension before feeding frames to the transformer encoder. The conv layer extracts local temporal gradients (rate of direction change, acceleration onset) that self-attention alone handles inefficiently at short sequences.

**When to use:** Always for trajectory prediction when the input sequence is short (5–50 frames). Self-attention has O(T²) attention cost, and raw position tokens carry no local context about trajectory shape.

**Trade-offs:** Adds a small number of learnable parameters. The kernel size controls how much local context is pre-extracted before attention. Kernel 3 = captures frame-to-frame change; kernel 5 = captures short burst dynamics.

**Example:**
```python
# In conv_encoder.py
import torch.nn as nn

class ConvTemporalEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # preserve sequence length
        )
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (batch, T, F) → transpose for Conv1d: (batch, F, T)
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        return x.transpose(1, 2)  # back to (batch, T, out_features)
```

### Pattern 2: Social Context as Concatenated Per-Frame Features

**What:** At each timestep t, append the (x, y) positions of all other 21 players to the target player's feature vector. This creates a wide per-frame feature but keeps the sequence structure clean.

**When to use:** When social context is positional and relatively low-dimensional (22 players x 2 = 44 extra dims). Alternative — separate social tokens in the transformer — is better if you want attention weights over specific players, but adds sequence length and complexity.

**Trade-offs:** Concatenation is simple and effective. Separate social tokens would allow the model to learn "which players matter" via attention, but doubles sequence length and may require more data. For v1 research scope, concatenation is the right call.

**Example:**
```python
# In features.py
def build_sample_features(
    player_frames: pd.DataFrame,    # T rows for target player
    all_player_frames: pd.DataFrame, # T rows x 22 players
    include_ball_destination: bool,
    ball_landing_xy: tuple[float, float] | None
) -> torch.Tensor:
    # Target player kinematics: (T, 6)
    kinematics = player_frames[["x", "y", "s", "dir", "o", "a"]].values

    # Social: flatten all 22 players' (x,y) at each frame: (T, 44)
    social = all_player_frames.pivot(
        index="frameId", columns="nflId", values=["x", "y"]
    ).values  # (T, 44)

    features = np.concatenate([kinematics, social], axis=1)  # (T, 50)

    if include_ball_destination and ball_landing_xy is not None:
        # Repeat constant ball destination across all T frames
        ball_dest = np.tile(ball_landing_xy, (len(features), 1))  # (T, 2)
        features = np.concatenate([features, ball_dest], axis=1)  # (T, 52)

    return torch.tensor(features, dtype=torch.float32)
```

### Pattern 3: Ablation via Dataset Flag (Not Model Masking)

**What:** The two models differ only in what features their training data contains. Model A's dataset is built with `include_ball_destination=False`; Model B's with `True`. Both use the identical model class, trainer, and hyperparameters.

**When to use:** Always for a clean ablation. Masking at inference time on a single model conflates "model never learned from this feature" with "model has the feature but it's zeroed out" — a different and weaker experiment.

**Trade-offs:** Requires building two separate datasets and training twice. That is the correct cost for a clean ablation.

---

## Data Flow

### Full Pipeline Flow

```
NFL Big Data Bowl CSVs
    │
    ▼
loader.py
    Read tracking_*.csv, plays.csv, players.csv
    Join on gameId + playId + nflId
    Output: raw merged DataFrame
    │
    ▼
preprocessor.py
    1. Filter to passing plays
    2. Normalize (x,y) relative to line of scrimmage
    3. Flip plays going left → standardize positive-X direction
    4. Filter target players: nflId where position in {CB, FS, SS, LB}
    5. Interpolate missing frames (linear, per player per play)
    6. Compute acceleration from speed delta if not provided
    Output: cleaned DataFrame, saved to data/processed/cleaned.parquet
    │
    ▼
sample_builder.py
    For each (nflId, gameId, playId):
        Extract T frames for target player
        Extract all 22 player positions at same T frames
        Extract ball_landing (x,y) from plays.csv (targetX, targetY or equivalent)
        Record ground-truth ending (x,y) for target player
    Output: list of Sample(features_a, features_b, label)
    │
    ▼  (splits into two dataset branches)
    │
    ├──▶ Dataset A (include_ball_destination=False)
    │       → DataLoader A → model_a training
    │
    └──▶ Dataset B (include_ball_destination=True)
            → DataLoader B → model_b training
    │
    ▼  (both produce .pt checkpoint files)
    │
    ▼
evaluate.py
    Load model_a_best.pt, model_b_best.pt
    Run inference on shared test set
    Compute RMSE per sample
    Compute per-position breakdowns
    Output: metrics dict + figures/
    │
    ▼
visualization/
    rmse_plots.py → bar chart A vs B, error distributions
    trajectory_viz.py → field overlay: predicted vs actual ending positions
```

### Key Data Shapes

| Stage | Shape | Notes |
|-------|-------|-------|
| Raw tracking row | scalar | One row per (player, frame, play) |
| Per-sample input (Model A) | (T, 50) | T frames; 6 kinematics + 44 social |
| Per-sample input (Model B) | (T, 52) | T frames; 6 kinematics + 44 social + 2 ball dest |
| After 1D conv | (T, hidden_dim) | hidden_dim recommended 64–128 |
| After transformer | (T, hidden_dim) | same sequence length |
| After pooling | (hidden_dim,) | mean-pool over T, or use learned CLS token |
| Model output | (2,) | predicted (x, y) in LOS-relative yards |
| Label | (2,) | actual ending (x, y) in LOS-relative yards |

---

## How the Ablation Study Shapes Architecture

The ablation study (Model A vs Model B) is the primary architectural driver. Every design choice must support a clean comparison:

1. **Identical architecture** — same class, same hyperparameters, same initialization seed. The only difference is the feature dimension of the input (50 vs 52).

2. **Same train/val/test splits** — both models must see the same plays. Splits are determined once, before any training, and saved to disk. Random seed fixed.

3. **Same training duration/budget** — identical epoch count, learning rate, scheduler, early stopping patience. Otherwise RMSE differences are confounded by training differences.

4. **Ball destination encoding** — appended as a constant across all T frames (not just the last frame). This gives the transformer equal access to this feature at every attention step.

5. **Evaluation on same test samples** — per-sample RMSE difference (B minus A) is the cleaner statistic than just aggregate RMSE; it shows which plays benefit most from knowing ball destination.

---

## How Social Features Integrate

Social context (all 22 players' positions) is treated as environmental state the target player is reacting to. The recommended integration is:

**Per-frame concatenation** (recommended for v1):
- At each frame t, flatten all 22 players' normalized (x, y) into a 44-dim vector
- Concatenate with the 6-dim kinematic vector for the target player → 50-dim per-frame feature
- The transformer then attends across T frames of this combined representation

**Why not separate social tokens:**
- Separate tokens would allow learning "CB attends to WR position more than OL" — interesting but not necessary for the ablation question
- Concatenation keeps sequence length = T (not T x 23), which is simpler and trains faster
- This is consistent with Social-LSTM/Social-GAN style context encoding

**Position normalization for social features:**
- All 22 player coordinates must be normalized to the same reference frame (LOS-relative, positive-X direction) as the target player
- This makes the social context invariant to field orientation

---

## Suggested Build Order

Dependencies flow strictly top-to-bottom. Each layer must work before the next is started.

```
1. Data Pipeline (loader → preprocessor → sample_builder)
   REASON: Everything downstream depends on clean, normalized samples.
   GATE: Sanity-check sample shapes and coordinate values in a notebook.

2. Dataset / DataLoader (dataset.py)
   REASON: Need to confirm batching works before touching the model.
   GATE: DataLoader yields (batch_features, batch_labels) of expected shape.

3. Model (conv_encoder → transformer → head → trajectory_model)
   REASON: Model is useless without data to run through it.
   GATE: Forward pass on dummy tensor produces (batch, 2) without error.

4. Trainer (train loop + val loop)
   REASON: Need model + data both working before training.
   GATE: Loss decreases on first 10 batches (overfits a tiny subset).

5. Train Model A (no ball destination)
   REASON: Baseline must exist before Model B can be compared against it.

6. Train Model B (with ball destination)
   REASON: After Model A is trained and validated, same process for B.

7. Evaluation (RMSE + per-position breakdown)
   REASON: Needs both trained models.
   GATE: Produces a numeric RMSE value per model.

8. Visualization
   REASON: Needs evaluation results.
   GATE: Figures match expected poster format.
```

---

## Anti-Patterns

### Anti-Pattern 1: Different Hyperparameters for Model A and Model B

**What people do:** Tune each model independently, then compare final RMSEs.

**Why it's wrong:** Differences in RMSE may reflect tuning effort rather than the ball destination feature. The ablation is only valid if both models are trained identically.

**Do this instead:** Fix all hyperparameters before training Model A. Use those exact same values for Model B. Do not tune Model B after seeing Model A's results.

### Anti-Pattern 2: Leaking Ball Destination Into the Test Set Label

**What people do:** Use the ball's actual landing position from the test play as the Model B input during evaluation.

**Why it's wrong:** In a real setting, ball landing location would need to come from a separate prediction (or be known in advance). However, for this ablation study the goal is to measure "does this information help if available?" — so using actual ball landing is acceptable and is the correct experimental design. The anti-pattern is using estimated/noisy ball landing at train time but perfect ball landing at test time, or vice versa.

**Do this instead:** Be explicit: Model B uses the actual recorded ball landing position (from plays.csv) both at train and test time. Document this clearly in the poster.

### Anti-Pattern 3: Predicting Full Trajectory Instead of Ending Location

**What people do:** Realize that ending-location regression loses information and pivot to per-frame predictions.

**Why it's wrong:** Full trajectory prediction multiplies label complexity by T and requires a different loss (summed over frames), making the ablation harder to interpret and the poster story more complex.

**Do this instead:** Stick to ending-location regression (scalar x,y output). Per-frame trajectory is listed as future work for good reason.

### Anti-Pattern 4: Variable-Length Sequences Without Padding Strategy

**What people do:** Feed variable-length play sequences to the DataLoader without deciding on a fixed T.

**Why it's wrong:** PyTorch's default batching requires uniform tensor shapes. Play lengths in the Big Data Bowl dataset vary (some plays are longer than others).

**Do this instead:** Pick a fixed T (e.g., T=20 frames, ~2 seconds of play action). Truncate long plays at the snap + T frames. Pad short plays with the last observed frame (post-pad, not pre-pad). Mask padded positions in the transformer using the `src_key_padding_mask` argument.

### Anti-Pattern 5: Processing All 22 Players as Targets

**What people do:** Include offensive players and special teamers in the training set "for more data."

**Why it's wrong:** The research question is about defensive player intent. Offensive players react differently (route running, blocking). Mixing them pollutes the learned representations.

**Do this instead:** Filter strictly to CB, FS, SS, LB after preprocessing, before sample building.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| preprocessor → sample_builder | parquet file on disk | Preprocessor writes once; sample_builder reads. Avoids reprocessing on every run. |
| sample_builder → dataset.py | in-memory list of Sample objects (or .pt tensors) | Can be serialized to disk if dataset is large |
| dataset.py → trainer.py | PyTorch DataLoader | Standard interface; trainer is agnostic to feature count |
| trainer.py → evaluate.py | .pt checkpoint files | Best val-loss weights saved; evaluate loads them independently |
| evaluate.py → visualization/ | metrics dict / CSV | Evaluation writes results; viz reads them. Keeps viz idempotent. |

### External Data Dependencies

| Source | What We Use | Notes |
|--------|-------------|-------|
| tracking_*.csv | Player (x,y,s,dir,o,a) per frame | Multiple files (one per week); need to concatenate |
| plays.csv | Ball landing location (targetX, targetY or passResult fields) | Check exact column names in the specific Big Data Bowl year's schema |
| players.csv | Player position (CB, FS, SS, LB, etc.) | Used for filtering to defensive players |
| games.csv | homeTeamAbbr / visitorTeamAbbr for field direction | Needed to determine which direction is "positive X" |

---

## Scaling Considerations

This is a research project — scaling is not a concern. The Big Data Bowl dataset is fixed-size (order of 10K–100K play samples across multiple seasons). All processing runs on a single machine. GPU training is beneficial but not required.

| Concern | This Project |
|---------|-------------|
| Dataset size | Fits comfortably in RAM (parquet + tensors < 5GB) |
| Training time | ~minutes per model on CPU; ~seconds per epoch on GPU |
| Parallelism | None needed; sequential A-then-B training is fine |

---

## Sources

- Social Transformer / Social-GAN patterns for pedestrian trajectory prediction (CVPR 2018, ICCV 2019) — LOW confidence (training knowledge, WebSearch unavailable)
- Trajectron++ multi-agent prediction architecture — LOW confidence (training knowledge)
- NFL Big Data Bowl dataset schema — MEDIUM confidence (training knowledge; verify column names against the specific competition year's data dictionary)
- PyTorch TransformerEncoder API — HIGH confidence (well-established, stable API)
- Conv-then-Transformer pattern for time-series — MEDIUM confidence (common in practice, e.g., PatchTST, Informer variants)

---

*Architecture research for: Transformer-based NFL Defensive Player Trajectory Prediction*
*Researched: 2026-03-13*
