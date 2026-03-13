# Feature Research

**Domain:** NFL defensive player trajectory prediction — ML research project with ablation study
**Researched:** 2026-03-13
**Confidence:** MEDIUM (PROJECT.md is authoritative; NFL BDB schema and trajectory prediction conventions are well-established; 2025 paper verification not available)

---

## Feature Landscape

### Table Stakes (Research is Invalid Without These)

These are the features that any serious trajectory prediction paper must have. A reviewer will reject the work if these are absent.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Raw tracking data ingestion | NFL BDB data is the dataset — can't run without loading it | LOW | CSVs with x, y, s (speed), a (accel), dir, o (orientation), event columns per frame at ~10Hz |
| Coordinate normalization to line of scrimmage | All plays must share a common reference frame; mixing absolute field coords produces garbage | LOW | Subtract LOS x; flip plays so offense always moves in +x direction |
| Player position filtering (CB, FS, SS, LB) | Study is scoped to defensive positions reacting to ball intent; mixing in OL corrupts signal | LOW | Filter on `position` field; treat each player-play as independent sample |
| Sequence construction per player-play | Model input is a time series; need fixed-length sequences with consistent padding strategy | MEDIUM | Decide on observation window length (e.g., first N frames of play); document the choice |
| Missing frame interpolation | NFL BDB has occasional dropped frames; gaps break sequential models | LOW | Linear interpolation on x, y, speed, direction; flag if >3 consecutive frames missing |
| Social context features (all 22 players) | Defensive player movement is conditioned on teammate and opponent positions; omitting context is a baseline regression model, not state-of-art | MEDIUM | Encode all 22 players' (x,y,s,dir) at each timestep as additional input tokens or feature channels |
| Ball landing location as directed feature | This IS the research question — Model B receives this, Model A does not; must be carefully injected (not leaked) | MEDIUM | Encode as a fixed appended token or feature; verify no temporal leakage into Model A |
| Two separately trained models (Model A / Model B) | The ablation design; a single masked model is an alternative but was explicitly rejected as a confound | HIGH | Same architecture, same hyperparameters, same train/val/test split — only input feature differs |
| Ending (x,y) prediction head | Output is the final position at play end; this is the regressed quantity | LOW | Two scalar outputs relative to LOS; not waypoints at each step |
| RMSE evaluation in yards | Standard metric for continuous position regression in sports tracking literature | LOW | Euclidean distance between predicted and actual ending (x,y); report in yards for interpretability |
| Train/validation/test split | Needed for honest generalization claim; test set must be held out before any modeling | LOW | Split by game or week to avoid within-game data leakage across plays |
| Ablation comparison table | Core deliverable — side-by-side RMSE for Model A vs Model B | LOW | Report mean RMSE, std dev, and statistical significance if sample size allows |

---

### Differentiators (What Makes This Ablation Compelling)

These features elevate the work from "we trained a model" to "we have a finding worth publishing or presenting."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| 1D conv layer for local trajectory extraction | Explicitly captures player's physical motion pattern before feeding into transformer attention; architecturally motivated, not arbitrary | MEDIUM | Apply Conv1D over the player's own feature sequence to extract short-range kinematic patterns before cross-attention with context |
| Transformer cross-attention between player and context | Attention mechanism allows the model to learn which teammates/opponents matter for each player's trajectory — interpretable by design | HIGH | Player's trajectory as query; social context (other 22 players) as keys/values; enables attention weight analysis |
| Ball-as-intent vs ball-as-physics distinction | The research insight: ball landing is a cognitive anchor for defenders, not just another positional input — framing this clearly is the scientific contribution | LOW (framing) / HIGH (validation) | Frame in writeup: separates "knowing physics" from "knowing intent"; drives literature review positioning |
| Per-position subgroup analysis | Breaking RMSE down by CB, FS, SS, LB tests whether ball intent helps some positions more than others (e.g., CB reacting to WR route vs LB reading run) | MEDIUM | Run the same ablation comparison within each defensive position group; adds depth to findings |
| Error distribution visualization (not just mean RMSE) | Distribution plots show whether ball destination reduces catastrophic errors (tail behavior) or improves average case — a qualitatively different finding | LOW | Histogram or CDF of per-play RMSE for Model A vs Model B; identifies whether improvement is systematic or outlier-driven |
| Trajectory overlay plots on field | Showing predicted vs actual paths on a scaled field diagram makes the improvement viscerally legible for a poster audience | MEDIUM | Plot top-N most improved plays and top-N where ball destination hurts; visual debugging also validates preprocessing |
| Attention weight heatmaps | If the transformer attends more to the ball/receiver when ball destination is provided, this is interpretability evidence that the model "uses" the feature as intended | HIGH | Visualize which input tokens receive highest attention; confirms model learns ball-intent signal, not spurious correlation |
| Statistical significance test on RMSE delta | Paired t-test or Wilcoxon signed-rank test on per-play RMSE confirms improvement is not noise | LOW | Essential for academic credibility; trivial to add once per-play errors are computed |

---

### Anti-Features (Deliberately Out of Scope for v1)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full waypoint trajectory prediction | More comprehensive, shows where player goes at each step not just final position | Multiplies output dimensionality, evaluation complexity, and training difficulty by T timesteps; completely changes the research question | Predict ending location only; note waypoint prediction as future work in the paper |
| Single model with masked ball feature | Simpler codebase; only one model to train | Masking strategies introduce their own confound — the model may learn to detect the masking signal; explicitly rejected in PROJECT.md | Two separately trained models; cleaner ablation |
| Offensive player prediction | "Coverage" requires modeling both sides | Offensive route is driven by play design, not reaction to the ball; fundamentally different prediction problem | Treat offensive player positions as context features (input) not prediction targets |
| Real-time inference / low-latency serving | Could make results more dramatic | Production deployment requires a separate engineering effort; wastes time that should go to research quality | Batch inference over the test set; report throughput as a footnote if needed |
| Hyperparameter search / neural architecture search | More rigorous, higher accuracy possible | Scope explosion; the research question is about the feature (ball destination), not about finding the best transformer config; hyperparameter tuning can change results for wrong reasons | Fix a reasonable architecture and hyperparams for both models; document them |
| Multi-sport generalization | Broader impact | Requires different datasets, different normalization, different feature schemas; doubles the data work without advancing the core hypothesis | Keep to NFL BDB; mention generalization as future work |
| Probabilistic / multi-modal output (distribution over endpoints) | Richer representation; captures uncertainty | Changes evaluation from RMSE to NLL or FDE/ADE; significantly harder to train; not what the project specified | Point prediction of ending (x,y) with RMSE; future work can add uncertainty quantification |
| Web dashboard / interactive visualization | Makes results more accessible | Engineering time with no research value; poster is the deliverable | Static matplotlib/seaborn figures exported as high-res PNGs for the poster |

---

## Feature Dependencies

```
[Data ingestion + raw loading]
    └──requires──> [Coordinate normalization]
                       └──requires──> [Position filtering (CB/S/LB)]
                                          └──requires──> [Sequence construction per player-play]
                                                             └──requires──> [Missing frame interpolation]
                                                                                ├──requires──> [Social context feature assembly]
                                                                                └──requires──> [Ball landing feature extraction]

[Model A training]
    └──requires──> [Sequence construction per player-play]
    └──requires──> [Social context feature assembly]
    └──requires──> [Train/val/test split]

[Model B training]
    └──requires──> [All Model A requirements]
    └──requires──> [Ball landing feature extraction]

[Ablation comparison table]
    └──requires──> [Model A training + evaluation]
    └──requires──> [Model B training + evaluation]
    └──requires──> [RMSE evaluation]

[Per-position subgroup analysis]
    └──requires──> [Ablation comparison table]
    └──requires──> [Position label preserved through pipeline]

[Error distribution visualization]
    └──requires──> [Per-play RMSE for both models]

[Trajectory overlay plots]
    └──requires──> [Model A + B inference on test set]
    └──requires──> [Coordinate normalization] (to plot back on field)

[Attention weight heatmaps]
    └──requires──> [Model B training]
    └──requires──> [Transformer implementation with accessible attention weights]

[Statistical significance test]
    └──requires──> [Per-play RMSE for both models]

[1D conv layer]
    └──feeds──> [Transformer encoder] (output is token sequence for attention)

[Transformer cross-attention]
    └──requires──> [Social context feature assembly] (context tokens)
    └──requires──> [1D conv layer output] (player query tokens)
```

### Dependency Notes

- **Coordinate normalization requires raw ingestion:** The normalization logic (subtract LOS, flip direction) depends on having the raw x/y and play direction fields loaded correctly. Any error here propagates through every downstream feature.
- **Ball landing feature must be isolated:** The ball's ending location must be extracted as a ground-truth label from the data, then injected ONLY into Model B at train time. Any leakage into Model A invalidates the ablation.
- **Train/val/test split must precede any training:** Split by game week (not random play) to prevent within-game data leakage. This decision must be made before preprocessing so no statistics from the test set contaminate normalization parameters.
- **Attention heatmaps require architecture transparency:** The transformer implementation must expose attention weights. This is straightforward with PyTorch's `nn.MultiheadAttention` with `need_weights=True`, but must be planned at architecture design time, not retrofitted.
- **1D conv output feeds transformer:** The conv layer is not standalone — its output sequence is the input to the transformer encoder. The channel dimensions must be designed together.

---

## MVP Definition

### Launch With (v1) — Minimum for Valid Ablation Results

These are necessary and sufficient to produce the core research finding and poster deliverable.

- [ ] Raw data loading + coordinate normalization + position filtering — without this, nothing runs
- [ ] Missing frame interpolation — data quality prerequisite
- [ ] Sequence construction per player-play — model input format
- [ ] Social context feature assembly (all 22 players) — richer baseline; explicitly in scope per PROJECT.md
- [ ] Ball landing location extraction (ground truth for Model B feature; label for both) — the ablation variable
- [ ] Train/val/test split by game week — honest evaluation
- [ ] 1D conv + Transformer encoder + linear output head — the shared architecture
- [ ] Model A training (no ball destination) — baseline
- [ ] Model B training (with ball destination) — experimental condition
- [ ] RMSE evaluation on test set for both models — primary metric
- [ ] Ablation comparison table (mean RMSE, delta, statistical test) — core finding
- [ ] Error distribution visualization (histogram or CDF) — supports the finding visually
- [ ] Trajectory overlay plots (predicted vs actual on field, sampled plays) — poster visual

### Add After Core Results Are Valid (v1.x)

- [ ] Per-position subgroup analysis (CB vs S vs LB breakdown) — adds depth if overall result is positive
- [ ] Attention weight heatmaps — interpretability layer; include if transformer implementation easily exposes weights
- [ ] Additional evaluation metrics (ADE if stakeholders request, though RMSE is sufficient per PROJECT.md) — only if reviewers ask

### Future Consideration (v2+)

- [ ] Full waypoint trajectory prediction — requires output architecture redesign and new evaluation protocol
- [ ] Probabilistic endpoint distribution — changes metric to NLL; different paper
- [ ] Multi-sport generalization — different datasets
- [ ] Online/streaming inference — production concern, not research concern

---

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|----------------|---------------------|----------|
| Data pipeline (load, normalize, filter, interpolate) | HIGH | LOW | P1 |
| Sequence construction + social context assembly | HIGH | MEDIUM | P1 |
| Ball landing feature extraction + injection | HIGH | LOW | P1 |
| Train/val/test split (game-week level) | HIGH | LOW | P1 |
| 1D Conv + Transformer + linear head architecture | HIGH | HIGH | P1 |
| Model A training + RMSE eval | HIGH | MEDIUM | P1 |
| Model B training + RMSE eval | HIGH | MEDIUM | P1 |
| Ablation comparison table + statistical test | HIGH | LOW | P1 |
| Error distribution visualization | HIGH | LOW | P1 |
| Trajectory overlay plots on field | HIGH | MEDIUM | P1 |
| Per-position subgroup analysis | MEDIUM | LOW | P2 |
| Attention weight heatmaps | MEDIUM | MEDIUM | P2 |
| Hyperparameter tuning / grid search | LOW | HIGH | P3 |
| Web dashboard / interactive viz | LOW | HIGH | P3 |
| Multi-sport generalization | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for valid research results and poster
- P2: Strengthens findings, add when core pipeline is complete
- P3: Nice to have, explicit anti-feature for v1

---

## Related Work / Comparable Research Patterns

This project sits at the intersection of three research lineages. Features expected in each:

| Feature | Pedestrian Trajectory Prediction | Sports Analytics (NBA/soccer) | NFL Big Data Bowl Papers |
|---------|----------------------------------|-------------------------------|--------------------------|
| Social context (neighbor positions) | Table stakes (Social Force Model, Social GAN) | Table stakes | Table stakes |
| Goal/destination as input | Core research question in many papers | Used in basketball "expected shot location" | Novel for defensive reaction |
| Transformer attention over agents | Standard (AgentFormer, Trajectron++) | Emerging | Emerging |
| RMSE / ADE / FDE as metrics | ADE + FDE standard | Varies by task | RMSE standard for endpoint tasks |
| Ablation of input features | Standard in any feature-engineering paper | Standard | Standard |
| Interpretability (attention viz) | Differentiator | Differentiator | Differentiator |

Our approach borrows the social context + attention paradigm from pedestrian prediction literature and applies it to NFL defense with a ball-intent feature that has no direct analog in pedestrian work. This framing strengthens the novelty claim.

---

## Sources

- PROJECT.md (authoritative scope, architecture decisions, out-of-scope items) — HIGH confidence
- NFL Big Data Bowl dataset schema (x, y, s, a, dir, o, event fields) — HIGH confidence (well-documented public dataset, stable across competition years)
- Pedestrian trajectory prediction literature conventions (ADE/FDE metrics, social context encoding, Social GAN / Trajectron++ / AgentFormer patterns) — MEDIUM confidence (established by ~2022, current papers iterate on these)
- Transformer-based trajectory prediction research (attention for agent interaction modeling) — MEDIUM confidence
- Ablation study design conventions in ML papers (same architecture, vary single input feature, statistical significance testing) — HIGH confidence

---

*Feature research for: NFL defensive player trajectory prediction with ball-intent ablation*
*Researched: 2026-03-13*
