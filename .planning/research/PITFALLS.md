# Pitfalls Research

**Domain:** Transformer-based NFL player trajectory prediction with ablation study design
**Researched:** 2026-03-13
**Confidence:** MEDIUM (training data through Aug 2025; WebSearch unavailable — findings based on established ML research practices and NFL Big Data Bowl community patterns)

---

## Critical Pitfalls

### Pitfall 1: Ball Coordinate Leakage into Model A (the Baseline)

**What goes wrong:**
Model A (physics-only baseline, no ball landing location) accidentally receives ball-position information — either directly through a feature that encodes ball XY at the end of the play, or indirectly through a normalized coordinate frame that is anchored to the ball's final landing spot. If Model A sees any signal derived from where the ball ends up, the ablation comparison is invalid: Model B's advantage will be understated or the entire study is confounded.

**Why it happens:**
The NFL Big Data Bowl dataset records ball position for every tracking frame, including the final frame. When engineers build a shared preprocessing pipeline that computes a "ball-relative" coordinate frame or includes ball XY in the social context tensor for all 22 players, Model A and Model B receive identical inputs by default. Ablation differences are then applied at training time but the leaked feature was already baked into every sample.

A second leak vector: coordinate normalization relative to line of scrimmage requires knowing the snap location, which is legitimately fine — but if the normalization step instead references the ball's *landing* frame, the ending ball destination is encoded in every player's normalized (x, y) at every timestep.

**How to avoid:**
- Define the ablation boundary in code before any training loop: a `FeatureSet` enum or config flag that governs exactly which columns are present in the input tensor.
- Model A's feature set must never include: ball final XY, ball landing frame position, or any coordinate transformation anchored to ball destination.
- Normalize all coordinates relative to line of scrimmage (snap location), not relative to ball landing. This is the correct normalization regardless.
- Add a test: load a batch for Model A, assert that the ball-destination columns are absent from the tensor. This test should fail loud if the pipeline is accidentally changed.
- Code review the feature construction function specifically looking for implicit ball-end leakage before any training run.

**Warning signs:**
- Model A RMSE approaches Model B RMSE too closely (within 0.2 yards) — suggests both are seeing similar information.
- Removing ball landing feature from Model B produces no RMSE degradation — suggests the feature wasn't truly separated.
- Validation loss for Model A improves unusually fast in early epochs.

**Phase to address:**
Data preprocessing and feature engineering phase — the feature set schema and the `FeatureSet` config must be locked before any model training begins.

---

### Pitfall 2: Random Play Split Instead of Temporal or Game-Level Split

**What goes wrong:**
The dataset is split train/val/test by randomly shuffling individual plays across weeks/games, so plays from the same game (or even the same drive) appear in both training and test sets. Because player motion patterns within a game are correlated (same players, same game conditions, same defensive scheme), the model memorizes game-level patterns rather than generalizing. Test RMSE looks good but the model has effectively "seen" the test players already.

**Why it happens:**
`sklearn.train_test_split` on a DataFrame of plays is the default, and it shuffles randomly. NFL Big Data Bowl datasets span multiple weeks; engineers assume "different plays = independent samples" without accounting for player or game-level correlation.

**How to avoid:**
- Split by `gameId` (or by week), not by play index. All plays from a given game go entirely into one partition.
- A temporal split is cleanest for research validity: use earlier weeks for training, later weeks for validation and test. This mirrors real-world deployment (predicting future games from past games).
- If using Big Data Bowl competition data (which is often a single season), split by week: e.g., Weeks 1-12 train, Weeks 13-15 val, Weeks 16-17 test.
- Assert in the split code that `set(train_game_ids) & set(test_game_ids) == set()`.

**Warning signs:**
- Training RMSE and test RMSE are very close from the start of training — may indicate test set leakage rather than a well-generalized model.
- Ablation delta (Model B minus Model A RMSE) varies wildly depending on random seed — suggests the split is contaminating results.

**Phase to address:**
Data preprocessing phase — the split strategy must be decided and implemented before any model training, and documented explicitly in the research methodology.

---

### Pitfall 3: Coordinate Normalization Inconsistency Across Plays

**What goes wrong:**
NFL tracking data contains plays from both left-to-right and right-to-left offensive directions. If all plays are not flipped to a canonical orientation (all offense moving in positive X direction), the model sees two different "languages" for the same physical event. A defensive back dropping back in zone coverage looks like a forward rush depending on which half of the data the play came from. The model learns a muddled representation and RMSE is higher than it should be.

A second sub-issue: the coordinate system origin. Raw NFL tracking data uses absolute field coordinates (0-120 yards long, 0-53.3 yards wide). Player positions in the red zone near the opponent's end zone and positions near mid-field have very different absolute coordinate values for the same motion pattern. Without normalizing to line of scrimmage, the model must learn a position-invariant representation instead of receiving one.

**Why it happens:**
The dataset's `playDirection` column ("left" or "right") is easy to overlook. Engineers who load the data and plot a few plays don't notice the issue unless they overlay many plays together and see the double-mirror distribution.

**How to avoid:**
- Apply direction normalization early in the pipeline: for all plays where `playDirection == "left"`, mirror x-coordinates as `x_norm = 120 - x` and mirror direction angles (add 180 degrees mod 360).
- Normalize all (x, y) coordinates relative to line of scrimmage location at snap. This is the `absoluteYardlineNumber` field.
- After normalization, verify by plotting: all offensive linemen should cluster near (0, ±n) and all play motion should proceed in positive X direction.
- Write a sanity-check assertion: after normalization, the mean starting X position of all offensive players should be near 0, not bimodally distributed at ±40 yards.

**Warning signs:**
- Scatter plot of starting player positions shows a bimodal distribution along the X axis.
- The model predicts players moving "backwards" (negative X) on a significant fraction of test plays.
- Loss curves are stable but RMSE is higher than expected for the architecture complexity.

**Phase to address:**
Data preprocessing phase — coordinate normalization must be validated with visualizations before any features are computed from coordinates.

---

### Pitfall 4: Overfitting on Position-Specific Subsets (CB, S, LB Class Imbalance)

**What goes wrong:**
The dataset is filtered to defensive positions (CB, FS, SS, LB), but the number of samples per position is highly unequal. In a typical NFL Big Data Bowl dataset spanning one season (~8,000-12,000 plays), CBs appear far more frequently than SSs or LBs in certain coverage schemes. If the model is trained on a combined pool of all defensive positions without position-aware sampling, it learns to minimize RMSE for the most common position at the expense of the rarer ones. RMSE reported as a single number looks acceptable but is dominated by CB performance.

Additionally, treating each player-play pair as an independent sample (as specified in the project) from 15-22 plays per game means the effective dataset is large in row count but highly correlated — many samples share the same players and game context.

**Why it happens:**
RMSE averages across all samples indiscriminately, hiding per-position variance. Researchers check aggregate RMSE, see a reasonable number, and don't break it down.

**How to avoid:**
- Always report RMSE broken down by position (CB, FS, SS, LB) in addition to aggregate RMSE.
- Consider stratified sampling within train/val/test splits to ensure each split has proportional representation of each position.
- If sample counts per position are severely imbalanced (>3x ratio), either oversample rare positions or explicitly note the limitation in the research.
- The ablation result (Model A vs B delta) should also be reported per-position, not just in aggregate.

**Warning signs:**
- Per-position RMSE breakdown shows one position (likely SS or FS) with much higher error than others.
- Val loss decreases monotonically but per-position metrics fluctuate — the model is optimizing for the majority class.
- The ablation delta is large for CBs but near-zero for LBs — may indicate the ball destination feature is only useful for certain coverage roles.

**Phase to address:**
Data analysis phase (before model training) — position distribution analysis and stratified split decisions; also evaluation phase — per-position reporting must be built into the evaluation script.

---

### Pitfall 5: RMSE as the Only Metric Obscuring Directional Errors

**What goes wrong:**
RMSE in yards is the correct primary metric for this problem and is interpretable. However, RMSE alone hides the *direction* of errors. A model that consistently predicts players too far downfield by 3 yards has the same RMSE as a model that makes random 3-yard errors in all directions — but these represent fundamentally different failure modes. The first indicates systematic bias (possibly a coordinate normalization error or the model not understanding zone vs. man coverage); the second is irreducible uncertainty.

For a research poster, reporting only aggregate RMSE also makes it hard to demonstrate *why* ball destination helps — the visual story (trajectory overlays) is as important as the number.

**Why it happens:**
RMSE is easy to compute and universally understood, so it becomes the only metric. Directional analysis requires extra work (computing signed X-error and Y-error separately, error roses, or bias plots).

**How to avoid:**
- Report both aggregate RMSE and its X and Y components separately: `RMSE_x` and `RMSE_y`. Defensive player motion is asymmetric — they move more in X (downfield/backfield) than Y (lateral). Understanding which axis benefits from ball destination is more informative.
- Compute Mean Signed Error (not just RMSE) to detect systematic bias: if mean signed X-error is consistently negative, the model is predicting players end up too far upfield.
- For the poster, produce trajectory overlay plots showing predicted vs. actual ending positions for a curated set of plays — these are more compelling than a single number.
- Consider Final Displacement Error (FDE) as a complementary metric — it is commonly used in pedestrian/vehicle trajectory prediction literature and provides a natural comparison to published work.

**Warning signs:**
- Mean signed error is large in one direction but RMSE looks acceptable — the model has systematic bias.
- Model B RMSE is only marginally better than Model A but trajectory visualizations show a clearly better spatial distribution — RMSE alone undersells the result.

**Phase to address:**
Evaluation phase — build the metrics module to include X/Y decomposition, signed error, and visualization from the start, not as an afterthought.

---

### Pitfall 6: Transformer Attention Over Social Context Without Positional Grounding

**What goes wrong:**
The transformer encoder receives the positions of all 22 players as social context tokens. If player tokens are not positionally differentiated (i.e., the model doesn't know which token is which player type), the attention mechanism may learn spurious correlations — e.g., attending to the nearest offensive player for all defensive positions, regardless of coverage role. Worse, if player ordering in the token sequence is inconsistent across plays (e.g., players are ordered by jersey number in some plays and by position group in others), the model learns an ordering artifact instead of a spatial relationship.

**Why it happens:**
Transformer inputs require a sequence. The natural choice is to iterate over players in whatever order the DataFrame provides them, which varies by play. Position embeddings for language transformers (encoding sequence index) are inappropriate for spatial context — player 1 is not semantically "before" player 2 the way token 1 precedes token 2 in a sentence.

**How to avoid:**
- Enforce a canonical player token ordering across all plays: e.g., always order by position group (QB, WRs, TEs, OL, RB, then defensive by position group), then by jersey number within group.
- Add a learned position-type embedding for each player token (a `player_type_embedding` lookup table indexed by position category: QB, WR, CB, etc.) rather than using sequential positional encodings.
- Alternatively, use relative spatial encodings: instead of indexing player 1, 2, 3... inject the (dx, dy) displacement of each social-context player relative to the focal defensive player's current position as part of that player's token.
- Verify attention weight distributions during development: the focal player should attend most heavily to nearby players, not to players with low jersey numbers.

**Warning signs:**
- Permuting player token order at inference time changes predictions significantly — indicates the model is learning ordering artifacts.
- Attention weights are concentrated on the first few tokens regardless of the play — ordering bias.
- Model performs differently on plays where the focal defensive player's jersey number is low vs. high.

**Phase to address:**
Feature engineering and model architecture phase — the social context token representation must be designed before the transformer architecture is finalized.

---

### Pitfall 7: Missing Frames and Interpolation Introducing False Smoothness

**What goes wrong:**
NFL Big Data Bowl tracking data has occasional missing frames (dropped tracking samples at 10Hz). Linear interpolation fills these gaps, which is reasonable for position but produces artificially smooth velocity and acceleration signals. When these interpolated frames are used to compute derived features (acceleration from velocity delta), the result is near-zero acceleration for interpolated spans, which is physically incorrect — players may have been accelerating or decelerating sharply during those frames.

A related issue: plays of different length (snap to whistle) have different sequence lengths. If sequences are padded to a fixed length without masking, the transformer attends to padding tokens and the loss is computed over padded timesteps.

**Why it happens:**
`pd.interpolate(method='linear')` is one line of code and appears to work. Computing acceleration as `(v_t - v_{t-1}) / dt` is also straightforward and doesn't flag interpolated frames. Padding without masking is the default in many PyTorch DataLoader patterns.

**How to avoid:**
- Track interpolated frames with a boolean flag column (`is_interpolated`). Exclude interpolated frames from acceleration computation — use `NaN` for acceleration at interpolated timesteps and fill with position-mean acceleration for that play.
- Use attention masks in the transformer to zero-attention on padding tokens. PyTorch's `nn.MultiheadAttention` accepts a `key_padding_mask` argument.
- Log statistics on missing frame rates per game/week — if a dataset week has >5% missing frames, flag it for exclusion or special handling.
- Consider fixed-length truncation instead of padding: take only the first N frames after snap, where N is chosen to cover 95% of play durations. Shorter plays are padded; longer plays are truncated. This bounds the padding problem.

**Warning signs:**
- Acceleration feature distribution shows suspicious spike at zero that is larger than expected — sign of over-interpolation.
- Loss decreases differently on shorter plays vs. longer plays — padding masking issue.
- Validation RMSE is higher for plays with many missing frames.

**Phase to address:**
Data preprocessing phase — interpolation strategy, derived feature computation, and sequence padding/masking must all be handled in the same step with explicit validation.

---

### Pitfall 8: Validating the Ablation Comparison Without Statistical Testing

**What goes wrong:**
The ablation result is reported as "Model B RMSE = 4.2 yards, Model A RMSE = 4.7 yards — ball destination improves prediction by 0.5 yards." This is presented as the finding without any measure of whether the difference is statistically significant or stable across different random seeds, train/test splits, or subsets of plays. A reviewer or professor asks: "Is this difference real or noise?" and there is no answer.

This is a common failure mode in ML ablation papers and is particularly problematic on small-to-medium datasets like a single NFL season (~8,000-12,000 plays filtered to defensive players).

**Why it happens:**
ML research culture historically focused on benchmark comparisons without significance tests. But for a focused ablation study with a single research question, this is the core claim and it must be robust.

**How to avoid:**
- Run both models with 3-5 different random seeds. Report mean RMSE ± standard deviation for each model.
- Perform a paired t-test or Wilcoxon signed-rank test on per-play RMSE differences (each play is a paired observation for Model A vs. Model B). Report the p-value.
- Report confidence intervals on the RMSE delta, not just the point estimate.
- Test on multiple evaluation subsets: by position, by game situation (redzone vs. midfield), by coverage type if labels are available.

**Warning signs:**
- The RMSE delta changes sign (Model A occasionally beats Model B) when the random seed changes — the signal is not robust.
- RMSE delta is largest on the smallest position subset — likely a high-variance artifact.

**Phase to address:**
Evaluation phase — the evaluation script must be written to run multiple seeds and report statistical summaries, not just a single RMSE number.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Single random seed for both models | Faster iteration | Ablation result may not be reproducible; no variance estimate | Never — always run at least 3 seeds before reporting results |
| Shared preprocessing pipeline for Model A and B without explicit feature gating | Simpler code | Ball destination leaks into Model A if pipeline changes | Never — feature gating must be explicit and tested |
| Random play split instead of game-level split | One line of code | Invalid test set; inflated RMSE performance | Never for final results; acceptable for debugging only |
| Aggregate RMSE only (no per-position breakdown) | Simpler reporting | Hides position-specific model failures; weakens research claims | Acceptable in initial development; never in final evaluation |
| Skipping attention mask for padded sequences | Faster initial implementation | Model attends to padding; loss is incorrect on variable-length plays | Acceptable for fixed-length truncation only; never for variable-length padding |
| Using absolute (not LOS-relative) coordinates | No normalization needed | Model must learn position-invariance it should be given; higher RMSE | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| NFL Big Data Bowl CSV schema | Assuming consistent column names across competition years — each year's dataset has different column naming conventions | Pin to a specific year's schema; write a schema validator at load time that asserts expected columns exist |
| NFL tracking `dir` column | Treating `dir` (player facing direction, 0-360 degrees) as a continuous feature without handling the 0/360 wraparound discontinuity | Convert `dir` to sin/cos components: `sin(dir_rad)`, `cos(dir_rad)` — this removes the discontinuity |
| `absoluteYardlineNumber` for LOS normalization | Field is 0-100 but the field is 120 yards; end zones are not included | Add 10 yards to `absoluteYardlineNumber` to get true field position before normalization |
| Play-level vs. frame-level joins | Joining play metadata (game situation, down, distance) to tracking frames — easy to create duplicate rows if join key is not unique | Always join on `(gameId, playId)` and assert row count is unchanged after join |
| Ball carrier vs. ball position | Ball position in tracking data is the ball's physical location; ball "landing location" for the ablation study is the ball's position at the end of the play (last frame before tackle/OOB) | Compute ball landing location as the ball XY at `max(frameId)` per play; do not use ball position at snap |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading entire tracking CSV into memory per epoch | Training loop is slow; memory usage spikes | Pre-process raw CSVs into a tensor cache (e.g., `.pt` files, one per play); load lazily with a custom Dataset | Immediately on datasets >2GB, which NFL multi-season data exceeds |
| Recomputing social context features (all 22 player positions) inside the training loop | GPU utilization is low because CPU is bottlenecked on feature construction | Precompute all input tensors offline and save to disk; training loop only does I/O and forward pass | Any dataset >5,000 plays with all-22 social context |
| Transformer with large sequence length and all-22 social context | Attention matrix is O(n^2) in sequence length × n_players; training slows quadratically | Cap sequence length (e.g., first 50 frames = 5 seconds at 10Hz); use linear attention if needed | At sequence length >100 with 22 players per frame |

---

## Research Validity Mistakes

These are not software bugs but failures that invalidate the research claim.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Reporting only the best seed's RMSE for each model | Cherry-picked result; ablation delta may not be real | Report mean ± std across 3-5 seeds |
| Comparing models trained for different numbers of epochs | Model B may win simply because it trained longer | Use early stopping with the same patience for both models; report best val RMSE epoch |
| Including the test set in hyperparameter tuning | Test set RMSE is optimistic; over-fit to test data | Hyperparameter tuning on val set only; test set touched exactly once |
| Ball landing location derived from Model A's predictions (circular dependency) | Model A can accidentally receive a "predicted" ball destination | Ball landing location must always be ground truth from the dataset, not a model output |
| Evaluating on plays where the ball destination is not reliable (e.g., incomplete passes, penalties) | Ball "landing location" is meaningless for these plays | Filter to complete runs and complete passes only; document exclusion criteria |

---

## "Looks Done But Isn't" Checklist

- [ ] **Coordinate normalization:** Verify plays are all in canonical positive-X orientation — plot 50 plays overlaid before any training begins. If you see a symmetric mirror image, normalization is not applied.
- [ ] **Feature gate test:** Add a unit test that instantiates the Model A input pipeline, loads one batch, and asserts ball-destination columns are absent from the tensor. This test must be in the test suite and run before training.
- [ ] **Split validation:** Print `len(set(train_games) & set(test_games))` — this must be 0. Make this an assertion, not a print.
- [ ] **Attention masking:** Verify padded frames have zero attention weight by inspecting attention outputs on a known padded sequence. Do not assume the mask is applied correctly.
- [ ] **Direction encoding:** Verify `dir` column is converted to sin/cos before use. Plot the raw `dir` distribution and the encoded sin/cos — the bimodal peak at 0/360 in raw data should disappear.
- [ ] **Per-position RMSE:** The evaluation script must print RMSE for CB, FS, SS, LB separately. A single aggregate number is not sufficient for the research claim.
- [ ] **Statistical significance:** Before finalizing results, run both models with at least 3 seeds and compute a p-value on the per-play RMSE difference.
- [ ] **Ball landing location source:** Verify the ball landing XY is computed from the dataset's final frame, not from any model output or prediction.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Ball leakage into Model A discovered after training | HIGH — must retrain from scratch | Audit feature pipeline, write leakage test, fix feature gate, retrain both models; results before fix are invalid |
| Random split instead of game split discovered after training | HIGH — must re-split and retrain | Implement game-level split, regenerate train/val/test splits, retrain; old results are invalid for reporting |
| Coordinate normalization not applied discovered mid-training | MEDIUM — can fix and resume | Fix normalization, recompute cached tensors, retrain from epoch 0 (don't resume from checkpoint — weights learned wrong representations) |
| Single seed results only | LOW — just run more seeds | Re-run with 3-5 seeds using the existing trained model code; compute variance |
| Missing attention mask | MEDIUM | Add `key_padding_mask` to transformer call, retrain; partial recovery by checking if plays have variable lengths in dataset |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Ball coordinate leakage into Model A | Data preprocessing / Feature engineering | Unit test asserting Model A tensor contains no ball-destination columns |
| Random split instead of temporal/game split | Data preprocessing | Assert `set(train_game_ids) & set(test_game_ids) == set()` |
| Coordinate normalization inconsistency | Data preprocessing | Overlay visualization of 50+ normalized plays; all offense moves in +X direction |
| Position imbalance / per-position RMSE hidden | Data analysis + Evaluation | Per-position RMSE reported in evaluation script output |
| RMSE-only metric obscuring directional bias | Evaluation phase | Evaluation script computes RMSE_x, RMSE_y, mean signed error, and FDE |
| Social context token ordering artifacts | Architecture design | Attention weight inspection on known plays; permutation test |
| Missing frame interpolation + padding masking | Data preprocessing | Padding mask passed to transformer; interpolated-frame flag in dataset |
| Ablation without statistical testing | Evaluation phase | p-value reported; multi-seed runs documented in results |

---

## Sources

- NFL Big Data Bowl competition submissions and kernels (Kaggle, 2019-2024) — community-documented preprocessing patterns and known schema quirks
- Alahi et al. "Social Force" and successor trajectory prediction literature — social context token design patterns and common pitfalls in pedestrian prediction (directly applicable to multi-agent sports tracking)
- Vaswani et al. attention mechanism documentation — key_padding_mask behavior in PyTorch nn.MultiheadAttention
- Salzmann et al. "Trajectron++" (2020) — temporal split methodology and per-agent RMSE reporting conventions
- Zhou et al. NFL tracking model papers (Big Data Bowl 2021, 2022) — coordinate normalization conventions and play direction handling
- General ML ablation study methodology: Sculley et al. "Hidden Technical Debt in Machine Learning Systems"; Bouthillier et al. "Unreproducible Research is Reproducible" (2019) — multi-seed variance requirements

---

*Pitfalls research for: Transformer-based NFL defensive player trajectory prediction with ablation study*
*Researched: 2026-03-13*
