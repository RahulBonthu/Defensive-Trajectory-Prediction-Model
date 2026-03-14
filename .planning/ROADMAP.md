# Roadmap: Defensive Trajectory Prediction Model

## Overview

Five sequential phases deliver the complete ablation study pipeline. The dependency chain is strict: correct, canonical player-play samples must exist before features can be constructed; input shapes must be known before the model can be built; model and trainer infrastructure must be validated before training runs; and trained model outputs must exist before any poster figure can be produced. Each phase gate-checks its output before the next phase begins, preventing the silent failures (leakage, wrong splits, mismatched coordinates) that would invalidate the research findings.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Pipeline and Validation** - Load, normalize, and validate raw NFL tracking data into canonical LOS-relative player-play samples
- [x] **Phase 2: Feature Engineering and Dataset Wrappers** - Build kinematic + social context feature vectors and PyTorch Datasets for both model variants with ablation gate enforced (completed 2026-03-14)
- [ ] **Phase 3: Model Architecture and Training Infrastructure** - Implement shared Conv-Transformer model class and identical training harness for both models
- [ ] **Phase 4: Model Training and Ablation Evaluation** - Train Model A and Model B, evaluate on held-out test set, produce ablation comparison with statistical significance
- [ ] **Phase 5: Visualization and Poster Figures** - Generate all poster-quality figures from trained model outputs and evaluation results

## Phase Details

### Phase 1: Data Pipeline and Validation
**Goal**: Correctly normalized, LOS-relative, temporally split player-play samples exist on disk and are visually validated
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, PREP-01, PREP-02, PREP-03, PREP-04, PREP-05, PREP-06
**Success Criteria** (what must be TRUE):
  1. `nfl-big-data-bowl-2026-prediction.zip` is unzipped and the `train/` and `test/` folders are extracted and accessible before any data loading begins
  2. Running the data pipeline on the zip file produces cleaned parquet/CSV samples without manual intervention
  3. A 50-play overlay visualization shows all offense moving in the positive X direction (confirming the play-direction flip is applied correctly)
  4. The train/val/test split assertion `set(train_game_ids) & set(test_game_ids) == set()` passes (no within-game leakage)
  5. Sequences with more than 3 consecutive missing frames are flagged and reported; all others have interpolated x, y, speed, and direction values
  6. Acceleration values are present for every frame in the output tensors
**Plans**: 5 plans

Plans:
- [x] 01-01-PLAN.md — Project scaffold: pyproject.toml, .gitignore, pytest config, synthetic fixtures, test stubs for all 8 requirements
- [x] 01-02-PLAN.md — Pipeline implementation: loader.py, preprocessor.py, sample_builder.py, run_pipeline.py, validate_normalization.py
- [x] 01-03-PLAN.md — DATA-03 gate: commit all code to GitHub, then prompt for dataset zip upload
- [x] 01-04-PLAN.md — Pipeline execution: run against real data, inspect schema, confirm splits.json and cleaned.parquet
- [x] 01-05-PLAN.md — Visual gate: human confirms 50-play overlay normalization, commit artifacts, close Phase 1

### Phase 2: Feature Engineering and Dataset Wrappers
**Goal**: Two PyTorch Datasets — one for each model variant — yield correctly shaped tensors with the ablation boundary provably enforced
**Depends on**: Phase 1
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06
**Success Criteria** (what must be TRUE):
  1. DataLoader for Model A yields tensors of shape (batch, T, 50) and DataLoader for Model B yields tensors of shape (batch, T, 52)
  2. A unit test asserting that Model A input tensors contain zero ball-destination columns passes without error
  3. Only CB, FS, SS, and LB player-play samples appear in the dataset (offensive and special-teams players excluded)
  4. Padded positions in short-sequence samples carry a boolean masking flag that downstream model code can consume
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Test scaffold: 6 failing stubs in test_dataset.py + minimal_samples/minimal_context_df fixtures in conftest.py
- [ ] 02-02-PLAN.md — Dataset implementation: DefensiveTrajectoryDataset with context index, social context assembly, ablation boundary (TDD, all 6 tests GREEN)
- [ ] 02-03-PLAN.md — Integration smoke test: DataLoader shapes against real cleaned.parquet + human verify

### Phase 3: Model Architecture and Training Infrastructure
**Goal**: A single shared model class instantiates cleanly for both Model A and Model B, and the training harness produces decreasing loss on a small overfit test
**Depends on**: Phase 2
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06
**Success Criteria** (what must be TRUE):
  1. A forward pass on a dummy (batch, T, 50) tensor and a dummy (batch, T, 52) tensor each produce a (batch, 2) output tensor without error
  2. Training loss decreases monotonically on a 100-sample overfit test for both model variants
  3. Both Model A and Model B are instantiated from the same class with only input dimension differing — verified by inspecting the config object
  4. Padded sequence positions receive near-zero attention weight (verified on a known padded sequence via attention weight inspection)
**Plans**: TBD

### Phase 4: Model Training and Ablation Evaluation
**Goal**: Both models are fully trained and the core research finding — the RMSE delta between Model A and Model B with statistical significance — is computed and stored
**Depends on**: Phase 3
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. model_a_best.pt and model_b_best.pt exist as saved checkpoints after training completes
  2. Validation loss for both models decreases then plateaus (confirmed via wandb run logs), indicating convergence
  3. An ablation comparison table exists reporting mean RMSE, standard deviation, and delta (Model A minus Model B) across 3-5 seeds for both models
  4. A p-value from a paired t-test or Wilcoxon signed-rank test on per-play RMSE differences is reported alongside the ablation table
  5. Per-position RMSE is reported separately for CB, FS, SS, and LB
**Plans**: TBD

### Phase 5: Visualization and Poster Figures
**Goal**: All poster-quality figures are produced at 300 DPI and correctly represent the ablation findings using LOS-relative coordinates
**Depends on**: Phase 4
**Requirements**: VIZ-01, VIZ-02
**Success Criteria** (what must be TRUE):
  1. An error distribution plot (histogram or CDF) exists comparing Model A vs Model B per-play RMSE, exported at 300 DPI
  2. Trajectory overlay plots exist showing predicted vs actual ending positions for sampled plays on a scaled field diagram, with LOS at x=0
  3. All figures use a consistent visual style and are ready to embed in a poster without further editing
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in strict sequential order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline and Validation | 5/5 | Complete | 2026-03-13 |
| 2. Feature Engineering and Dataset Wrappers | 3/3 | Complete    | 2026-03-14 |
| 3. Model Architecture and Training Infrastructure | 0/TBD | Not started | - |
| 4. Model Training and Ablation Evaluation | 0/TBD | Not started | - |
| 5. Visualization and Poster Figures | 0/TBD | Not started | - |
