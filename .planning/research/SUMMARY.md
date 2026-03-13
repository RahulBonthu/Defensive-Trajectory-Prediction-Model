# Project Research Summary

**Project:** NFL Defensive Trajectory Prediction Model
**Domain:** Transformer-based sports player trajectory prediction (ML research / ablation study)
**Researched:** 2026-03-13
**Confidence:** MEDIUM — stack HIGH, features HIGH, architecture MEDIUM, pitfalls MEDIUM

## Executive Summary

This is a focused ML research project producing an academic poster with a single, testable scientific claim: does knowledge of where the ball will land improve a transformer model's ability to predict where a defensive NFL player ends up at the end of a play? The project is structured as a clean ablation study — Model A trains without ball destination, Model B trains with it, everything else is held constant, and the difference in RMSE is the finding. The project is scoped deliberately: endpoint (x, y) regression only, no waypoint prediction, no hyperparameter search, no web interface. The output is a static poster with RMSE comparison tables and field overlay visualizations.

The recommended approach is a vanilla PyTorch pipeline using a 1D convolutional temporal encoder feeding into a standard TransformerEncoder, followed by a linear regression head. Social context (all 22 players) is encoded by concatenating per-frame (x, y) positions alongside the target player's kinematics — no separate player tokens needed for v1. The key architectural discipline is that the two models differ only in input feature dimension (50 vs 52 features per frame); everything else — hyperparameters, architecture class, training budget, optimizer, splits — must be identical. Ball destination is appended as a constant repeated across all T frames so the transformer has equal access to it at every attention step.

The most critical risk is ball coordinate leakage into Model A during preprocessing. If the feature pipeline is built carelessly, ball destination information can be baked into the input tensor for both models before any training occurs, invalidating the entire ablation. A close second risk is using a random play-level train/test split instead of a game-level or temporal split, which inflates test performance due to within-game player correlations. Both of these are silent failures — the code runs fine and produces numbers that look plausible. The mitigation is explicit: lock feature gating in code before any training, write unit tests asserting Model A's tensor has no ball-destination columns, and enforce game-level split by asserting `set(train_game_ids) & set(test_game_ids) == set()`.

---

## Key Findings

### Recommended Stack

The standard Python ML ecosystem covers everything this project needs. PyTorch 2.10 is the correct choice for a from-scratch transformer on numerical time-series data — it provides `nn.TransformerEncoder`, `nn.Conv1d`, and `nn.MultiheadAttention` natively, and its eager-mode training loop makes the ablation boundary (different input tensors, identical everything else) easy to implement without framework abstraction getting in the way. HuggingFace Transformers and PyTorch Lightning are explicitly anti-recommended: they add abstraction over the training loop that makes surgical control of the input feature dimension awkward. Full research files: see STACK.md.

**Core technologies:**
- Python 3.11 / PyTorch 2.10.0: deep learning framework — eager mode is ideal for research iteration on custom architectures
- pandas 3.0.1 + NumPy 2.4: CSV ingestion and coordinate transforms — pandas for tabular joins, NumPy for LOS-relative normalization and array ops
- scikit-learn 1.8.0: `StandardScaler` for feature normalization and `train_test_split` utilities — do not reinvent
- matplotlib 3.10.8 + seaborn 0.13.x: poster-quality static figures — matplotlib for field overlay plots, seaborn for RMSE distribution comparison charts
- scipy 1.14.x: linear interpolation for missing tracking frames — more flexible than pandas `.interpolate()` for variable-rate gap handling
- wandb: experiment tracking — free tier sufficient for two model runs; makes ablation results reproducible and shareable
- pytest 8.x: unit testing preprocessing logic — essential for catching leakage and normalization bugs silently

**Critical version note:** pandas 3.x requires numpy 2.x. Do not pin numpy < 2.0. Avoid Python 3.10 (reaching EOL); Python 3.11 is the sweet spot for this stack.

### Expected Features

The research is organized around a core dependency chain: every feature downstream of data ingestion is blocked on having correct, canonical, LOS-relative coordinates. Get normalization right first; everything else follows. Full research files: see FEATURES.md.

**Must have (table stakes — research is invalid without these):**
- Raw tracking data ingestion (tracking_*.csv, plays.csv, players.csv, games.csv) — the dataset itself
- Coordinate normalization to line of scrimmage with play-direction flip — all plays must share a canonical orientation
- Defensive player filtering (CB, FS, SS, LB only) — research question is scoped to defenders reacting to ball intent
- Fixed-length sequence construction per (player, play) with consistent padding — model input format
- Missing frame linear interpolation with interpolated-frame flag — data quality prerequisite
- Social context features: all 22 players' (x, y, s, dir) at each frame — table stakes for trajectory prediction; omitting this makes the model a naive baseline
- Ball landing location as directed feature injected into Model B only — the ablation variable; must be isolated from Model A
- Two separately trained models (A: no ball destination, B: with ball destination) — clean ablation, not a masked model
- Ending (x, y) prediction head — endpoint regression is the research question
- RMSE evaluation in yards — standard metric; report per-play errors to enable statistical testing
- Game-week-level train/val/test split — honest generalization; prevent within-game leakage
- Ablation comparison table with statistical significance test — the core deliverable

**Should have (differentiators for a compelling poster):**
- 1D convolutional layer for local trajectory extraction before transformer — architecturally motivated, extracts kinematic patterns self-attention handles inefficiently at short sequences
- Error distribution visualization (histogram or CDF per model) — shows whether improvement is systematic or outlier-driven
- Trajectory overlay plots on field — makes improvement viscerally legible to a poster audience
- Per-position subgroup analysis (CB / FS / SS / LB) — tests whether ball intent helps some positions more than others
- Attention weight heatmaps — interpretability evidence that the model uses ball intent as intended

**Defer (v2+):**
- Full waypoint trajectory prediction — multiplies output dimensionality and changes the research question
- Probabilistic endpoint distribution — requires NLL loss, different evaluation protocol, different paper
- Multi-sport generalization — different datasets, different feature schemas
- Hyperparameter search / NAS — scope explosion; the research question is about the feature, not the best architecture config

### Architecture Approach

The system is a four-stage offline research pipeline: data ingestion, feature engineering, model training, and evaluation/visualization. There is no serving layer and no streaming — all processing is batch on a single machine. The critical architectural insight is that the ablation boundary is drawn at dataset construction time, not at the model level. Model A and Model B are the same class instantiated twice with different input feature dimensions (50 vs 52 per frame). This makes the comparison clean and the code transparent. Full research files: see ARCHITECTURE.md.

**Major components:**
1. Data Pipeline (loader → preprocessor → sample_builder): reads CSVs, normalizes coordinates, interpolates gaps, filters defensive players, produces (T, F) input tensors cached to disk
2. Feature Engineering (features.py + dataset.py): builds per-frame kinematic + social context vectors; applies ablation flag to include or exclude ball destination; wraps in PyTorch Dataset/DataLoader
3. Model (ConvTemporalEncoder → TransformerEncoder → LinearHead): 1D conv extracts local trajectory dynamics, transformer attends across T frames, linear head outputs (x, y)
4. Trainer: identical hyperparameters and training budget for both models; early stopping on val loss; checkpoints model_a_best.pt and model_b_best.pt
5. Evaluation (metrics.py + ablation.py): per-sample RMSE, per-position breakdown, X/Y component errors, statistical significance test on paired per-play RMSE deltas
6. Visualization (rmse_plots.py + trajectory_viz.py + attention_viz.py): poster-quality figures comparing Model A vs B RMSE distributions and field overlay plots

### Critical Pitfalls

1. **Ball coordinate leakage into Model A** — Enforce an explicit `FeatureSet` config/enum that governs input tensor columns before any training. Write a unit test that loads a Model A batch and asserts ball-destination columns are absent. Normalize coordinates to line of scrimmage (snap location), never to ball landing location.

2. **Random play-level split instead of game-level split** — Split by `gameId` (or week). Assert `set(train_game_ids) & set(test_game_ids) == set()`. Use a temporal split: earlier weeks for train, later weeks for val/test.

3. **Coordinate normalization inconsistency** — Apply play-direction flip (`playDirection == "left"` → mirror x, add 180° to angles) in the preprocessor. Validate by overlaying 50+ normalized plays: all offense should move in +X direction. Bimodal X distribution means the flip was not applied.

4. **Missing attention mask for padded sequences** — Always pass `src_key_padding_mask` to `nn.MultiheadAttention`. Verify on a known padded sequence that padded positions receive near-zero attention weight. Track interpolated frames with a boolean flag; exclude them from acceleration computation.

5. **Ablation results without statistical testing** — Compute per-play RMSE for both models and run a paired t-test or Wilcoxon signed-rank test. Report p-value and RMSE ± std across 3-5 seeds. A single mean RMSE number is insufficient for any academic claim.

---

## Implications for Roadmap

Based on the dependency chain established in feature research and the architecture build-order recommendation, the project naturally decomposes into five phases. The data pipeline is a hard gate on everything else — no model can be trained until normalized, filtered, validated samples exist. Visualization is the final phase because it requires trained model outputs.

### Phase 1: Data Pipeline and Validation

**Rationale:** The entire project depends on having correct, canonical, LOS-relative player-play samples. Coordinate normalization inconsistency and wrong split strategy are HIGH-recovery-cost bugs that must be fixed before any training occurs. This phase builds the foundation and validates it visually before proceeding.

**Delivers:** Cleaned parquet file of normalized plays, fixed-length (T, F) input tensors for all (player, play) pairs, game-level train/val/test split saved to disk, sanity-check visualizations proving normalization is correct.

**Addresses:** Raw data ingestion, coordinate normalization, play-direction flip, defensive player filtering, missing frame interpolation, sequence construction, social context assembly.

**Avoids:** Ball coordinate leakage (feature gate locked here), random split pitfall (game-level split enforced here), coordinate normalization inconsistency (validated visually here), interpolated-frame masking setup, direction encoding (sin/cos of `dir` column applied here).

**Gate:** 50-play overlay visualization shows all offense moving in +X; unit test asserting Model A tensor has no ball-destination columns passes; split assertion `set(train_games) & set(test_games) == set()` passes.

### Phase 2: Feature Engineering and Dataset Wrappers

**Rationale:** Builds on validated preprocessed data. The ablation boundary (Model A vs Model B feature sets) must be implemented and tested here before any model code is written — retrofitting feature gating after the model is trained is the primary leakage risk.

**Delivers:** `features.py` with base kinematic + social context builder; `dataset.py` with ablation flag producing two PyTorch Datasets; DataLoader yielding (batch_features, batch_labels) of expected shapes for both model variants.

**Addresses:** Ball landing location extraction and isolation as ablation variable, social context per-frame concatenation (50 dims for Model A, 52 for Model B), StandardScaler normalization of features.

**Avoids:** Social context token ordering artifacts (canonical player ordering enforced here), ball destination leakage into Model A (feature gate tested here).

**Gate:** DataLoader for both model variants yields tensors of expected shape; Model A batch asserted to not contain ball-destination columns.

### Phase 3: Model Architecture and Training Infrastructure

**Rationale:** Model and trainer are straightforward to build once data shapes are known. Building them after the data pipeline is confirmed avoids the common mistake of designing the model first and retrofitting the data pipeline to its assumptions.

**Delivers:** `ConvTemporalEncoder`, `TransformerEncoder` wrapper, `LinearHead`, `TrajectoryModel` assembly; `Trainer` with MSE loss, Adam optimizer, LR scheduler, early stopping, checkpoint saving.

**Addresses:** 1D conv + transformer + linear head architecture; `src_key_padding_mask` for padded sequences; `need_weights=True` on attention for heatmap support; identical hyperparameter config shared by both models.

**Avoids:** Different hyperparameters for Model A and B (single config class), full-trajectory prediction scope creep, variable-length sequence handling without padding mask.

**Gate:** Forward pass on dummy (batch, T, 50) and (batch, T, 52) tensors produces (batch, 2) without error; loss decreases on a 100-sample overfit test.

### Phase 4: Model Training and Ablation Evaluation

**Rationale:** Once data and model infrastructure are validated, train both models and produce the core research finding. Statistical significance testing is part of this phase — it is not an afterthought.

**Delivers:** model_a_best.pt, model_b_best.pt; per-sample RMSE for both models on the test set; ablation comparison table with mean RMSE ± std across 3-5 seeds, per-position breakdown, p-value on paired RMSE delta; X/Y component errors and mean signed error.

**Addresses:** Model A training (no ball destination), Model B training (with ball destination), RMSE evaluation, per-position subgroup analysis, statistical significance test.

**Avoids:** Comparing models trained for different epochs (same early stopping patience), cherry-picking single seed result (3-5 seeds required), test set contamination (touched only once here), evaluating on incomplete passes (filter to complete plays only).

**Gate:** Both models converge (val loss decreases then plateaus); ablation delta is consistent across seeds; p-value computed and reported.

### Phase 5: Visualization and Poster Figures

**Rationale:** Visualization requires trained model outputs and evaluation results. This phase is last by necessity. The deliverable is the poster — all figures must be poster-quality (300 DPI, consistent style).

**Delivers:** RMSE comparison bar/violin plots (A vs B), error distribution histograms or CDFs, trajectory overlay plots on scaled field diagram (predicted vs actual ending positions for top-N most-improved plays), optional attention weight heatmaps if interpretability enhances the poster narrative.

**Addresses:** Error distribution visualization, trajectory overlay plots, attention weight heatmaps (P2 feature), poster figure export at 300 DPI.

**Uses:** matplotlib with `plt.style.use('seaborn-v0_8-paper')`, seaborn for distribution plots, `fig.savefig(..., dpi=300, bbox_inches='tight')`.

**Gate:** All figures are poster-ready; trajectory overlays correctly show LOS-relative coordinates; visualizations confirm that Model B improvement is genuine (not coordinate normalization artifact).

### Phase Ordering Rationale

- Phases 1-2 (data) must precede Phase 3 (model) because the model's input dimension is determined by the feature engineering decisions, and the ablation gate must be tested before any training.
- Phase 3 (architecture + trainer) must precede Phase 4 (training) because the trainer infrastructure and hyperparameter config must exist before either model can be trained.
- Phase 4 (training + evaluation) must precede Phase 5 (visualization) because all figures require model outputs and per-sample RMSE values.
- The entire architecture research confirms this strict top-to-bottom dependency with explicit gate checks at each stage boundary.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (Data Pipeline):** NFL Big Data Bowl CSV schema column names vary by competition year. The exact column name for ball landing location in `plays.csv` (e.g., `targetX`, `absoluteYardlineNumber`, or similar) must be verified against the specific dataset year being used before implementation. The `absoluteYardlineNumber` + 10-yard end-zone offset is a known gotcha.
- **Phase 2 (Feature Engineering):** The optimal approach for social context (per-frame concatenation vs. separate tokens) is well-reasoned in research but not empirically validated for this specific dataset. Start with concatenation for v1; revisit if attention weight heatmaps suggest the model is not learning spatial relationships.

Phases with standard patterns (skip additional research):
- **Phase 3 (Model Architecture):** PyTorch `nn.TransformerEncoder`, `nn.Conv1d`, and `nn.MultiheadAttention` are mature, stable APIs with extensive documentation. The conv-then-transformer pattern for short time series is well-established. No additional research needed.
- **Phase 4 (Training):** Standard PyTorch training loop with Adam, cosine or step LR scheduler, and MSE loss. Statistical significance testing (paired t-test / Wilcoxon) is straightforward with scipy. No additional research needed.
- **Phase 5 (Visualization):** matplotlib field overlay and seaborn distribution plots are well-documented. No additional research needed.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core library versions verified against official documentation (PyTorch, pandas, NumPy, scikit-learn, matplotlib). NFL-specific tooling choice (direct CSV load vs. nfl_data_py) is correct and well-reasoned. |
| Features | HIGH | Feature scope and dependency chain are internally consistent and align with project scope in PROJECT.md. Pedestrian trajectory prediction conventions (social context, RMSE/FDE) are well-established. MVP definition is clear. |
| Architecture | MEDIUM | PyTorch API patterns are HIGH confidence. Social Transformer / Trajectron++ patterns are MEDIUM (training knowledge, no live literature access during research). Specific NFL BDB column names for ball landing location need verification against the actual dataset year. |
| Pitfalls | MEDIUM | All critical pitfalls are based on well-established ML research practices and NFL BDB community patterns. Confidence is MEDIUM because WebSearch was unavailable during research — findings are based on training knowledge through Aug 2025. The pitfalls identified are structurally sound and independently verifiable. |

**Overall confidence:** MEDIUM-HIGH — the research is internally consistent and covers all major risk areas. The primary gap (NFL BDB schema column name verification) is resolvable at project kickoff by inspecting the actual CSV headers.

### Gaps to Address

- **NFL BDB column name for ball landing location:** The exact column(s) in `plays.csv` that encode where the ball lands (e.g., `targetX`/`targetY`, or derived from ball tracking rows at `max(frameId)`) must be verified against the specific competition year's data dictionary before Phase 1 begins. Do not assume column names from one year match another.
- **Dataset year and sample size:** The research assumes a single-season dataset (~8,000-12,000 pass plays). The actual competition year and total play count should be confirmed early — if sample count is smaller than expected, 3-5 seed variance testing becomes more critical (higher variance per seed) and per-position subgroup analysis may have too few samples for statistical significance.
- **MPS vs. CUDA training environment:** The research provides correct guidance for both Apple Silicon (MPS) and CUDA GPU environments. The team should confirm the training environment at project start and set up the correct PyTorch installation to avoid discovering a CPU-only environment mid-training.

---

## Sources

### Primary (HIGH confidence)
- https://pytorch.org/get-started/locally/ — PyTorch 2.10.0 stable version, CUDA/MPS installation commands
- https://pandas.pydata.org/docs/whatsnew/ — pandas 3.0.1 release verification
- https://numpy.org/doc/stable/ — NumPy 2.4 current stable
- https://scikit-learn.org/stable/whats_new.html — scikit-learn 1.8.0 verification
- https://matplotlib.org/stable/users/release_notes.html — matplotlib 3.10.8 verification
- PROJECT.md — authoritative project scope, architecture decisions, out-of-scope items
- NFL Big Data Bowl dataset schema (x, y, s, a, dir, o, event fields) — well-documented public dataset

### Secondary (MEDIUM confidence)
- NFL Big Data Bowl Kaggle competition community notebooks (2019-2024) — preprocessing patterns, play-direction handling, coordinate normalization conventions
- Pedestrian trajectory prediction literature (Social Force, Social GAN, Trajectron++) — social context encoding patterns, ADE/FDE metrics, ablation study conventions
- PyTorch TransformerEncoder API patterns — conv-then-transformer for time series, attention masking

### Tertiary (LOW confidence — needs validation)
- Social Transformer / GRIP / SoPhie architecture patterns — training knowledge, WebSearch unavailable during research
- Specific NFL BDB column names in plays.csv (e.g., `targetX`, `targetY`) — verify against actual competition year data dictionary before implementation

---
*Research completed: 2026-03-13*
*Ready for roadmap: yes*
