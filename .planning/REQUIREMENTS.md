# Requirements: Defensive Trajectory Prediction Model

**Defined:** 2026-03-13
**Core Value:** Quantifiably demonstrate that knowing the ball's destination improves defensive player trajectory prediction accuracy via a clean transformer ablation study

## v1 Requirements

### Data Ingestion

- [ ] **DATA-01**: System accepts `nfl-big-data-bowl-2026-prediction.zip` and extracts `train/` and `test/` folders automatically
- [ ] **DATA-02**: System loads raw NFL Big Data Bowl CSV tracking files from extracted zip structure
- [ ] **DATA-03**: All changes are committed to GitHub before any step that prompts the user to provide the dataset

### Preprocessing

- [ ] **PREP-01**: Player coordinates are normalized relative to the line of scrimmage (subtract LOS x-coordinate)
- [ ] **PREP-02**: All offensive plays are flipped so the offense always moves in the positive X direction
- [ ] **PREP-03**: Direction angle features (`dir`, `o`) are encoded as sin/cos components to handle the 0°/360° discontinuity
- [ ] **PREP-04**: Missing frames are filled via linear interpolation on x, y, speed, and direction; sequences with >3 consecutive missing frames are flagged
- [ ] **PREP-05**: Acceleration is computed from velocity when not provided in the raw data
- [ ] **PREP-06**: Train/validation/test split is performed by game week (not random play) before any normalization statistics are computed — prevents data leakage

### Feature Engineering

- [ ] **FEAT-01**: Dataset is filtered to defensive positions only: CB, FS, SS, LB
- [ ] **FEAT-02**: Each player-play pair is treated as an independent motion sample
- [ ] **FEAT-03**: Fixed-length input sequences are constructed per player-play (observation window from snap + N frames); short plays are post-padded with masking
- [ ] **FEAT-04**: Social context features are assembled: all 22 players' (x, y, speed, direction) at each timestep are included as additional input channels
- [ ] **FEAT-05**: Ball landing location is extracted as a ground-truth feature from the dataset — injected into Model B inputs only; never present in Model A inputs
- [ ] **FEAT-06**: A unit test verifies that Model A training inputs contain zero ball destination information (leakage prevention)

### Model Architecture

- [ ] **MODEL-01**: A shared model class implements the full architecture: 1D Conv layer → Transformer Encoder → Linear output head
- [ ] **MODEL-02**: The 1D Conv layer processes the player's own kinematic time-series to extract local trajectory patterns before attention
- [ ] **MODEL-03**: The Transformer Encoder cross-references player trajectory tokens against social context (other players) to learn interaction effects
- [ ] **MODEL-04**: The linear output head predicts ending (x, y) position relative to the line of scrimmage (2 scalar outputs)
- [ ] **MODEL-05**: Model A is instantiated from the shared class with ball destination feature disabled
- [ ] **MODEL-06**: Model B is instantiated from the shared class with ball destination injected as a directed feature token

### Training

- [ ] **TRAIN-01**: Both Model A and Model B are trained with identical architecture hyperparameters and the same train/val/test split
- [ ] **TRAIN-02**: RMSE (root mean square error in yards) is used as the training loss function
- [ ] **TRAIN-03**: Training runs are logged to wandb with full config for reproducibility
- [ ] **TRAIN-04**: Both models are saved as checkpoints after training

### Evaluation

- [ ] **EVAL-01**: Both models are evaluated on the held-out test set; per-play RMSE is computed and stored for each prediction
- [ ] **EVAL-02**: An ablation comparison table reports mean RMSE, standard deviation, and delta (Model A − Model B) for both models
- [ ] **EVAL-03**: A statistical significance test (paired t-test or Wilcoxon signed-rank) is run on per-play RMSE differences between Model A and Model B
- [ ] **EVAL-04**: Per-position subgroup RMSE is reported separately for CB, FS, SS, and LB

### Visualization

- [ ] **VIZ-01**: Error distribution plots (histogram or CDF) compare Model A vs Model B per-play RMSE — shows whether improvement is systematic or outlier-driven
- [ ] **VIZ-02**: Trajectory overlay plots show predicted vs actual ending position for sampled plays, plotted on a scaled field diagram — primary poster visual

## v2 Requirements

### Interpretability

- **INTERP-01**: Transformer attention weight heatmaps visualize which input tokens receive highest attention in Model B
- **INTERP-02**: Expose attention weights via `nn.MultiheadAttention` with `need_weights=True` at inference time

### Extended Analysis

- **EXTEND-01**: Additional evaluation metrics (ADE/FDE) if reviewers or collaborators request them
- **EXTEND-02**: Training on full NFL BDB historical dataset across multiple years to increase statistical power

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full waypoint trajectory prediction | Changes output architecture and evaluation protocol entirely; different research question |
| Single model with masked ball feature | Introduces masking confound; two separate models is cleaner ablation (explicitly decided) |
| Offensive player prediction | Offensive routes are play-design driven, not reactive; fundamentally different problem |
| Hyperparameter search / NAS | Research question is about the feature (ball destination), not the best transformer config |
| Probabilistic / multi-modal output | Changes metric to NLL; different paper |
| Real-time inference / serving | Production concern; not a research deliverable |
| Web dashboard / interactive visualization | Static poster figures are the deliverable |
| Multi-sport generalization | Requires different datasets; scope expansion with no benefit to core hypothesis |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| PREP-01 | Phase 1 | Pending |
| PREP-02 | Phase 1 | Pending |
| PREP-03 | Phase 1 | Pending |
| PREP-04 | Phase 1 | Pending |
| PREP-05 | Phase 1 | Pending |
| PREP-06 | Phase 1 | Pending |
| FEAT-01 | Phase 2 | Pending |
| FEAT-02 | Phase 2 | Pending |
| FEAT-03 | Phase 2 | Pending |
| FEAT-04 | Phase 2 | Pending |
| FEAT-05 | Phase 2 | Pending |
| FEAT-06 | Phase 2 | Pending |
| MODEL-01 | Phase 3 | Pending |
| MODEL-02 | Phase 3 | Pending |
| MODEL-03 | Phase 3 | Pending |
| MODEL-04 | Phase 3 | Pending |
| MODEL-05 | Phase 3 | Pending |
| MODEL-06 | Phase 3 | Pending |
| TRAIN-01 | Phase 4 | Pending |
| TRAIN-02 | Phase 4 | Pending |
| TRAIN-03 | Phase 4 | Pending |
| TRAIN-04 | Phase 4 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-02 | Phase 4 | Pending |
| EVAL-03 | Phase 4 | Pending |
| EVAL-04 | Phase 4 | Pending |
| VIZ-01 | Phase 5 | Pending |
| VIZ-02 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 30
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-13 after initial definition*
