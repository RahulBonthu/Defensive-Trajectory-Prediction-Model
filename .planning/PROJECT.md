# Defensive Trajectory Prediction Model

## What This Is

A transformer-based deep learning system that predicts where NFL defensive players (CB, S, LB) end up at the end of a play, using sub-second player tracking data from the NFL Big Data Bowl dataset. The core research question: does providing the ball's landing location as a directed feature allow the model to better capture player intent and produce more accurate non-linear trajectory predictions?

## Core Value

Quantifiably demonstrate that knowing the ball's destination improves defensive player trajectory prediction accuracy — measured via RMSE on ending (x,y) position — using an ablation study comparing two separately trained transformer models.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Load and preprocess NFL Big Data Bowl tracking data (x,y, speed, orientation, direction, position, intent for all 22 players + ball landing location)
- [ ] Normalize player coordinates relative to line of scrimmage; flip all offensive plays to positive X axis
- [ ] Handle missing frames via linear interpolation; compute acceleration from velocity when not provided
- [ ] Filter and isolate samples for defensive positions: CB, FS, SS, LB
- [ ] Treat each player's play sequence as an independent motion sample
- [ ] Include other player positions as social context features (all 22 players' locations)
- [ ] Train Model A: transformer without ball landing location (physics-only baseline)
- [ ] Train Model B: transformer with ball landing location as directed feature
- [ ] 1D convolution layer to extract player's current physical trajectory from time-series input
- [ ] Transformer encoder that cross-references player speed/direction against ball trajectory
- [ ] Linear output layer predicting ending (x,y) position relative to line of scrimmage
- [ ] Evaluate both models using RMSE (yards between predicted and actual ending location)
- [ ] Generate ablation comparison: error distribution with vs without ball destination
- [ ] Produce result visualizations suitable for poster (graphs of RMSE comparison, trajectory overlays)

### Out of Scope

- Full trajectory prediction (waypoints at each timestep) — v1 predicts ending location only; full path is future work
- Offensive player prediction — focused on defensive players reacting to ball intent
- Real-time inference / production deployment — this is a research project
- Single model with masked inference — v1 uses two separately trained models for cleaner ablation

## Context

- Dataset: NFL Big Data Bowl (publicly available, high-fidelity sub-second tracking data)
- Research framing: ablation study — single architecture, two training conditions
- Evaluation: RMSE in yards (standard for continuous motion prediction)
- Deliverable: working training pipeline + ablation results + research poster
- Team: Archit Kumar, Rahul Bonthu
- Social features (other player trajectories as input) are in scope for v1

## Constraints

- **Data**: NFL Big Data Bowl dataset — must work within its schema and available features
- **Architecture**: Transformer encoder is the chosen model family; not experimenting with other architectures in v1
- **Metric**: RMSE is the primary evaluation metric — consistency with sports tracking literature
- **Scope**: Research project — results need to be clear and visualizable for poster format

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Two separate models (not one model with masking) | Cleaner ablation — eliminates confounds from masking strategy | — Pending |
| Transformer over LSTM/CNN | Token-based attention allows experimenting with which features to amplify/suppress | — Pending |
| Predict ending location only (not full trajectory) | Simplifies problem, matches poster scope | — Pending |
| Include social features (all 22 player positions) | Richer context, aligns with future work direction | — Pending |
| RMSE as primary metric | Standard for continuous motion prediction; interpretable in yards | — Pending |

---
*Last updated: 2026-03-13 after initialization*
