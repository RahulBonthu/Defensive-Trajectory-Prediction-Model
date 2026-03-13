# Stack Research

**Domain:** Sports player trajectory prediction with transformers (NFL defensive players)
**Researched:** 2026-03-13
**Confidence:** HIGH for versions (verified against official docs); HIGH for core library choices (standard ML ecosystem); MEDIUM for NFL-specific tooling

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11+ | Runtime | Stable, async improvements, faster CPython; avoid 3.12+ if any Cython-based deps need catching up |
| PyTorch | 2.10.0 | Deep learning framework, transformer training | Dominant framework for research; `nn.TransformerEncoder`, `nn.Conv1d`, `nn.MultiheadAttention` all natively available; TensorFlow is production-oriented but weak for research iteration |
| pandas | 3.0.1 | NFL Big Data Bowl CSV loading and preprocessing | NFL tracking data ships as CSV; pandas is the standard for heterogeneous tabular data; 3.0 has Copy-on-Write by default which prevents silent bugs in preprocessing pipelines |
| NumPy | 2.4 | Numerical arrays, coordinate transforms, interpolation | Underlying array layer; coordinate normalization (relative to LOS, flip to positive X) is a pure NumPy operation; avoid doing this in pandas for speed |
| scikit-learn | 1.8.0 | Train/val/test splits, StandardScaler normalization, RMSE metric | `sklearn.model_selection.train_test_split` and `StandardScaler` are the standard for reproducible splits and feature normalization; don't reinvent |
| matplotlib | 3.10.8 | Trajectory overlay plots and RMSE distribution plots for poster | Fine-grained subplot control needed for poster-quality figures; seaborn sits on top for statistical plots but matplotlib is the foundation |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| seaborn | 0.13.x | Statistical visualization (error distribution plots, box plots) | Use for RMSE distribution comparison charts between Model A and Model B; cleaner API than raw matplotlib for statistical plots |
| tqdm | 4.x | Training loop progress bars | Wrap DataLoader iteration and epoch loops; adds ~zero overhead and makes long training sessions readable |
| scipy | 1.14.x | Linear interpolation for missing tracking frames | `scipy.interpolate.interp1d` is the standard for filling gaps in positional time-series; more flexible than pandas `.interpolate()` for variable-rate gaps |
| pytest | 8.x | Unit testing preprocessing and data pipeline | Test coordinate normalization, interpolation, feature engineering — these are the silent failure points |
| jupyter / jupyterlab | 4.x | Exploratory data analysis, per-play visualization | Mandatory for EDA on tracking data; also useful for generating poster figures interactively |
| wandb | 0.x (latest) | Experiment tracking, ablation logging | Track both model runs with identical hyperparameter configs; log training RMSE, val RMSE per epoch; makes ablation comparison reproducible and shareable |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| conda or venv | Environment isolation | Use conda if CUDA GPU training is needed on local machine; pip+venv is sufficient for CPU/cloud training |
| black | Code formatting | Opinionated formatter; set line-length 88; prevents style debates on a 2-person team |
| jupyter nbconvert | Export notebooks to figures | Convert EDA notebooks to static HTML or PDF for archival alongside poster |

---

## Installation

```bash
# Core ML stack
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
# For CPU-only (no GPU):
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cpu

# Data science stack
pip install "pandas==3.0.1" "numpy==2.4" "scikit-learn==1.8.0"

# Visualization
pip install "matplotlib==3.10.8" "seaborn>=0.13"

# Utilities
pip install "scipy>=1.14" "tqdm>=4.0" "wandb" "pytest>=8.0"

# Development
pip install "jupyterlab>=4.0" "black"
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| PyTorch 2.10 | TensorFlow / Keras | Only if team already has TF expertise and needs Keras's high-level API; for custom transformer architectures with 1D conv + cross-attention, PyTorch's imperative style is faster to iterate on |
| PyTorch built-in `nn.TransformerEncoder` | HuggingFace Transformers | HuggingFace is appropriate when fine-tuning a pre-trained language model; this project trains from scratch on sports tracking data (no text), so HuggingFace adds weight without benefit |
| PyTorch built-in `nn.TransformerEncoder` | x-transformers (lucidrains) | x-transformers has useful extras (rotary embeddings, flash attention) but adds a dependency for a research prototype; use native PyTorch unless hitting positional encoding scaling issues |
| wandb | MLflow | MLflow requires self-hosted infrastructure; wandb's free tier is sufficient for a 2-person research project with two model runs |
| scipy interpolation | pandas `.interpolate()` | pandas interpolate is fine for simple linear gaps; use scipy when interpolation needs to be along the time axis with variable-rate tracking frames or when you need bounds control |
| seaborn | plotly | plotly is better for interactive exploration; seaborn is better for static poster-quality statistical figures |
| Standard `nn.TransformerEncoder` | Performer / Linformer | These linear-attention variants are needed only if sequence length > 1000 tokens; NFL plays are short sequences (typically 15–80 frames at 10fps), so full attention is not a computational bottleneck |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| TensorFlow 2.x | Research iteration is slower due to graph-mode defaults and Keras abstraction hiding gradient-level control; PyTorch's eager mode is the standard for academic research | PyTorch 2.10 |
| HuggingFace `transformers` library | Overkill for a from-scratch sports model; tokenizer/model card ecosystem adds friction with no benefit for numerical time-series data | `torch.nn.TransformerEncoder` + `torch.nn.TransformerEncoderLayer` directly |
| Keras (standalone) | Same as TensorFlow concerns; custom 1D conv + attention stacking is awkward in Keras subclassing API | PyTorch `nn.Module` |
| `nfl_data_py` (nflverse) | Pulls live NFL data from the nflverse API, not the Big Data Bowl competition dataset; schemas differ and `nfl_data_py` does not include the sub-second player tracking columns (x, y, s, a, dir, o) | Direct CSV load from Kaggle competition download |
| Ray Tune / Optuna (hyperparameter search) | This is an ablation study comparing two training conditions, not a hyperparameter sweep; adding tuning infrastructure increases scope with no research value in v1 | Fixed hyperparameters, document them explicitly |
| PyTorch Lightning | Adds abstraction layer over training loop; fine for large-scale projects but makes it harder to surgically control how ball landing location is injected as a directed feature vs. excluded (the core ablation variable) | Vanilla PyTorch training loop with manual optimizer/scheduler control |

---

## Stack Patterns by Variant

**If training on a local Mac (MPS / Apple Silicon):**
- Use `pip install torch torchvision` (no CUDA index URL)
- Set device to `torch.device("mps")` if `torch.backends.mps.is_available()` else `"cpu"`
- MPS is significantly faster than CPU for this model size; avoid CUDA-specific ops that don't have MPS equivalents

**If training on a CUDA GPU (e.g., Colab, Lambda, RunPod):**
- Use CUDA 12.8 index URL: `https://download.pytorch.org/whl/cu128`
- Enable `torch.compile()` (available since PyTorch 2.0) on the model for ~20% training speedup with no code changes
- Use `torch.cuda.amp.autocast()` (automatic mixed precision) for faster training on modern GPUs

**If sequence lengths grow beyond 80 frames (future work only):**
- Consider adding flash attention via `torch.nn.functional.scaled_dot_product_attention` (available natively in PyTorch 2.x with SDPA backend)
- Not needed for v1 (typical play = 15–80 frames)

**For poster figure generation:**
- Use `matplotlib` with `rcParams` set to a consistent style (`plt.style.use('seaborn-v0_8-paper')`)
- Export at 300 DPI with `fig.savefig(..., dpi=300, bbox_inches='tight')`

---

## NFL Big Data Bowl Data Structure

The competition data ships as CSV files with the following key tables (MEDIUM confidence — schema verified from multiple community notebooks, official Kaggle schema page not directly accessible):

| File | Key Columns | Notes |
|------|-------------|-------|
| `tracking_week_N.csv` | `gameId, playId, nflId, frameId, time, x, y, s, a, dis, o, dir, event` | Primary tracking data; one row per player per frame at ~10fps; x/y in yards from back-left corner of field |
| `plays.csv` | `gameId, playId, possessionTeam, defensiveTeam, yardlineSide, yardlineNumber, offenseFormation, down, yardsToGo, passResult, offensePlayResult` | Play metadata; used to identify pass plays and filter |
| `players.csv` | `nflId, displayName, position, height, weight` | Player metadata; used to filter CB, FS, SS, LB |
| `games.csv` | `gameId, season, week, homeTeamAbbr, visitorTeamAbbr` | Game metadata |

Key preprocessing notes:
- x/y origin is fixed field corner; must convert to line-of-scrimmage-relative coordinates
- All plays must be flipped so offense moves in +x direction (home and away teams move in opposite directions)
- Ball position is tracked as a separate `nflId` row with `displayName = "football"`
- Frame rate is nominally 10fps but can have gaps; linear interpolation is standard for missing frames

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| torch==2.10.0 | Python 3.11–3.13 | Avoid Python 3.10 (reaching EOL); Python 3.11 is the sweet spot |
| torch==2.10.0 | CUDA 12.6, 12.8, 13.0 | Multiple CUDA versions supported via index URL; cu128 is the most broadly available on cloud providers |
| pandas==3.0.1 | numpy>=2.0 | pandas 3.x requires numpy 2.x; do not pin numpy<2.0 |
| scikit-learn==1.8.0 | numpy>=2.0, scipy>=1.14 | Consistent with numpy 2.4 |
| matplotlib==3.10.8 | numpy>=2.0 | No conflicts with rest of stack |

---

## Sources

- https://pytorch.org/get-started/locally/ — Confirmed PyTorch 2.10.0 as current stable, installation commands (HIGH confidence)
- https://pandas.pydata.org/docs/whatsnew/ — Confirmed pandas 3.0.1 released February 17, 2026 (HIGH confidence)
- https://numpy.org/doc/stable/ — Confirmed NumPy 2.4 as current stable (HIGH confidence)
- https://scikit-learn.org/stable/whats_new.html — Confirmed scikit-learn 1.8.0 (HIGH confidence)
- https://matplotlib.org/stable/users/release_notes.html — Confirmed matplotlib 3.10.8 released November 2025 (HIGH confidence)
- NFL Big Data Bowl schema — MEDIUM confidence; based on consistent patterns across multiple published Kaggle notebooks and the 2024/2025 competitions; direct Kaggle data page was not accessible for verification
- PyTorch `nn.TransformerEncoder`, `nn.Conv1d` availability — HIGH confidence; these have been stable PyTorch APIs since PyTorch 1.1 and 1.0 respectively; confirmed present in 2.x
- Library choices (wandb, scipy, seaborn) — HIGH confidence; standard choices in sports analytics and ML research papers using this exact problem framing

---

*Stack research for: NFL defensive player trajectory prediction with transformers*
*Researched: 2026-03-13*
