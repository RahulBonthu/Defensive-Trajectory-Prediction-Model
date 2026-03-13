# Phase 2: Feature Engineering and Dataset Wrappers - Research

**Researched:** 2026-03-13
**Domain:** PyTorch Dataset/DataLoader, social context feature assembly, ablation boundary enforcement
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FEAT-01 | Filter dataset to defensive positions only: CB, FS, SS, LB | Verified in cleaned.parquet: 1,056,888 CB + 476,865 FS + 392,421 SS + 31 LB rows = 68,344 unique player-play samples across all splits |
| FEAT-02 | Each player-play pair is an independent motion sample | sample_builder.build_samples() already groups by (gameId, playId, nflId) — Dataset wraps these groups |
| FEAT-03 | Fixed-length input sequences post-padded with masking | sample_builder already produces `frames` (T, 8) and `padding_mask` (T,) bool; Phase 2 Dataset extends to (T, 50/52) |
| FEAT-04 | Social context: all 22 players' (x, y, speed, direction) at each timestep | BDB 2026 has max 17 players/play-frame; 'social context' must be padded to fixed width; feature math resolves to: own 8 features + 21 other players' (x, y) = 42 = 50 total |
| FEAT-05 | Ball landing injected into Model B inputs only; never in Model A | ball_land_x / ball_land_y confirmed in cleaned.parquet, 0 nulls, normalized range -21.75 to +65.42 (x), -30.68 to +30.56 (y) |
| FEAT-06 | Unit test: Model A training inputs contain zero ball destination information | Test must assert tensor[:, :, 50:] sums to 0 (or asserts input width == 50) for all Model A batches |
</phase_requirements>

---

## Summary

Phase 2 converts the 4.88M-row `cleaned.parquet` into two PyTorch `Dataset` subclasses — one for each model variant — yielding fixed-shape tensors ready for the transformer. The hard constraint is the ablation boundary: Model A tensors must contain exactly 50 features per timestep with zero ball-destination information, while Model B tensors must contain exactly 52 features with `ball_land_x` and `ball_land_y` appended as constant-across-time channels. A unit test (FEAT-06) must make this boundary unforgeable.

The key engineering challenge is social context assembly. The BDB 2026 dataset tracks 9–17 players per play (median 13), not the full 22 players assumed in the requirements. The feature count math resolves cleanly as: **8 own kinematic features + 42 social context features (21 other players × 2 position features each) = 50**. Other players beyond 21, and missing slots, are zero-padded. This interpretation is consistent with the (batch, T, 50) requirement and avoids inflating the feature vector with speed/direction for context players (which would push the total to 88+).

The `sample_builder.build_samples()` function in Phase 1 already produces per-player-play sample dicts with `frames (T, 8)`, `padding_mask (T,)`, `target_xy (2,)`, and `ball_target_xy (2,)`. Phase 2 builds on top of this: the Dataset `__getitem__` assembles the full (T, 50) or (T, 52) tensor from these Phase 1 samples plus per-frame social context lookups into the cleaned.parquet.

**Primary recommendation:** Build a `DefensiveTrajectoryDataset(Dataset)` class parameterized by `include_ball_destination: bool` that is instantiated twice — once for Model A, once for Model B. The ablation boundary is enforced by this single boolean, tested by FEAT-06.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10.0+cpu (installed) | Dataset/DataLoader, tensor ops | Project dependency; already installed |
| pandas | 3.0.1 (installed) | Read parquet, per-play groupby | Already used in Phase 1 |
| numpy | 2.4 (installed) | Array construction, zero-padding | Already used in sample_builder |
| pyarrow | installed | Parquet I/O backend for pandas | Already in pyproject.toml |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=8.0 | Unit tests including FEAT-06 leakage test | All test tasks |
| json (stdlib) | — | Load splits.json | Loading train/val/test game ID sets |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Building tensors in `__getitem__` | Pre-materializing all tensors into `.pt` files | Pre-materialization uses more disk but faster DataLoader; deferred to Phase 3 if training is slow |
| Zero-padding short sequences | Truncating short plays | Padding preserves all plays; masking tells model which frames are real |

**Installation:** No new dependencies needed. All required libraries are in pyproject.toml.

---

## Architecture Patterns

### Recommended Project Structure
```
src/
├── data/
│   ├── loader.py          # Phase 1 (unchanged)
│   ├── preprocessor.py    # Phase 1 (unchanged)
│   ├── sample_builder.py  # Phase 1 (minor: strict position filter)
│   └── dataset.py         # NEW: DefensiveTrajectoryDataset
tests/
├── conftest.py            # Add fixtures for dataset tests
├── test_pipeline.py       # Phase 1 (unchanged)
└── test_dataset.py        # NEW: FEAT-01 through FEAT-06 tests
```

### Pattern 1: Single Parameterized Dataset Class (Ablation Boundary)
**What:** One `Dataset` subclass with `include_ball_destination: bool` — eliminates drift between two separate classes.
**When to use:** Any time two variants differ by exactly one boolean feature toggle.
**Example:**
```python
# Source: PyTorch Dataset docs (https://pytorch.org/docs/stable/data.html)
import torch
from torch.utils.data import Dataset

class DefensiveTrajectoryDataset(Dataset):
    def __init__(self, samples: list[dict], context_df: pd.DataFrame,
                 sequence_length: int = 25, include_ball_destination: bool = False):
        self.samples = samples          # list of dicts from sample_builder
        self.context_df = context_df    # full cleaned.parquet for social context lookup
        self.seq_len = sequence_length
        self.include_ball = include_ball_destination
        # Build play-frame lookup: (gameId, playId, frameId) -> array of (nflId, x, y)
        self._build_context_index()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        # Assemble (T, 50) or (T, 52) tensor
        tensor = self._build_tensor(sample)
        return {
            "input": tensor,                         # (T, 50) or (T, 52)
            "padding_mask": torch.tensor(sample["padding_mask"], dtype=torch.bool),  # (T,)
            "target_xy": torch.tensor(sample["target_xy"], dtype=torch.float32),     # (2,)
            "position": sample["position"],
        }
```

### Pattern 2: Social Context Assembly
**What:** For each frame in the target player's sequence, look up all OTHER players' (x, y) at that frame, sort canonically by nflId, pad/truncate to exactly 21 slots.
**When to use:** Every `__getitem__` call — the 42 social context features.
**Example:**
```python
def _assemble_social_context(self, game_id: int, play_id: int,
                              frame_ids: list[int], target_nfl_id: int) -> np.ndarray:
    """Return (T, 42) array — 21 other players * (x, y) per frame, zero-padded."""
    N_CONTEXT = 21
    result = np.zeros((len(frame_ids), N_CONTEXT * 2), dtype=np.float32)
    play_frames = self._context_index.get((game_id, play_id), {})
    for t, fid in enumerate(frame_ids):
        others = sorted(
            [(nfl_id, xy) for nfl_id, xy in play_frames.get(fid, {}).items()
             if nfl_id != target_nfl_id],
            key=lambda item: item[0]  # sort by nflId for determinism
        )
        for slot, (_, xy) in enumerate(others[:N_CONTEXT]):
            result[t, slot * 2 : slot * 2 + 2] = xy
    return result
```

### Pattern 3: Context Index (Pre-Built in `__init__`)
**What:** Pre-index `(gameId, playId, frameId, nflId) -> (x, y)` at Dataset construction time to avoid per-`__getitem__` DataFrame scans.
**When to use:** Always — eliminates O(N) pandas lookup per sample.
**Example:**
```python
def _build_context_index(self):
    """Build {(gameId, playId): {frameId: {nflId: np.array([x, y])}}} index."""
    index = {}
    for row in self.context_df[["gameId","playId","frameId","nflId","x","y"]].itertuples():
        key = (row.gameId, row.playId)
        if key not in index:
            index[key] = {}
        if row.frameId not in index[key]:
            index[key][row.frameId] = {}
        index[key][row.frameId][row.nflId] = np.array([row.x, row.y], dtype=np.float32)
    self._context_index = index
```

### Anti-Patterns to Avoid
- **Querying pandas DataFrame inside `__getitem__`:** O(N) scan on 4.88M rows per sample is prohibitively slow. Pre-build the context index in `__init__`.
- **Building separate ModelA_Dataset and ModelB_Dataset classes:** Two classes diverge. Use a single class with `include_ball_destination` boolean.
- **Including the target player in their own social context:** Must exclude `target_nfl_id` from the 21 context slots — otherwise the model sees its own future position as a context feature.
- **Non-deterministic player ordering in social context:** Sort by `nflId` ascending for reproducibility — different ordering each epoch would cause training instability.
- **Leaking ball_land into Model A via social context:** ball_land_x/y are play-level constants in cleaned.parquet. They must ONLY be appended when `include_ball_destination=True`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batching + shuffling | Custom batch sampler | `torch.utils.data.DataLoader` | Handles worker processes, pin_memory, shuffle seed |
| Tensor batching from variable-length lists | Manual np.stack loop | DataLoader with `collate_fn` or default collate | Default collate stacks same-shape tensors correctly |
| Train/val/test split | Split logic in Dataset | Pre-computed `splits.json` from Phase 1 | Game-disjoint split already done; re-splitting would cause leakage |

**Key insight:** Phase 1 already did the hard work (normalization, splits, ball landing extraction). Phase 2 is purely about assembling the right tensor shape from what exists.

---

## Common Pitfalls

### Pitfall 1: Feature Count Mismatch
**What goes wrong:** (batch, T, 50) requirement fails — tensor comes out as (batch, T, 88) or (batch, T, 42) or another shape.
**Why it happens:** Misinterpreting FEAT-04 as "all 22 players × 4 features (x,y,s,dir) = 88". The only consistent interpretation yielding exactly 50 is: 8 own kinematic features + 21 other players × 2 position features (x,y only) = 50.
**How to avoid:** Hard-code `CONTEXT_PLAYERS = 21` and `CONTEXT_FEATURES_PER_PLAYER = 2` as module constants. Assert `tensor.shape[-1] == 50` in `__getitem__` during development.
**Warning signs:** DataLoader yields batches of shape (B, T, 88) or (B, T, 58).

### Pitfall 2: Position Filter Mismatch
**What goes wrong:** ILB/OLB/MLB samples appear in the dataset (FEAT-01 violated).
**Why it happens:** Phase 1 `sample_builder.py` uses a broad `DEFENSIVE_POSITIONS` set that includes ILB, OLB, MLB, DE, DT. Phase 2 must apply the strict 4-position filter: `{"CB", "FS", "SS", "LB"}`.
**How to avoid:** Apply `df[df["position"].isin({"CB","FS","SS","LB"})]` filter in Phase 2 dataset construction, or update `DEFENSIVE_POSITIONS` in sample_builder. The Phase 1 sample count of 93,824 was based on the broad filter; Phase 2 strict filter yields **68,344 samples** (52,779 train / 7,497 val / 8,068 test).
**Warning signs:** Dataset `__len__` returns ~93K instead of ~68K.

### Pitfall 3: Ball Destination Leakage in Model A
**What goes wrong:** FEAT-06 test fails — ball_land features present in Model A tensors.
**Why it happens:** ball_land_x/y are columns in cleaned.parquet and exist on every row. If the social context builder naively concatenates all available columns, leakage occurs.
**How to avoid:** Never include ball_land_x/y in the per-frame feature loop. Append them as the LAST 2 features ONLY when `include_ball_destination=True`, and only from the play-level constant (not from per-frame context).
**Warning signs:** Model A tensor[:, :, 50:52] is non-zero, or tensor.shape[-1] == 52.

### Pitfall 4: Social Context at Padded Timesteps
**What goes wrong:** Padded frames contain non-zero social context features, confusing the transformer.
**Why it happens:** Frame IDs beyond the play's actual length don't exist in the cleaned.parquet. If the context index lookup returns stale data from the last real frame, padded timesteps become non-zero.
**How to avoid:** The context assembly loop should return all-zeros for frame IDs not present in `_context_index[key]`. The `dict.get(fid, {})` pattern naturally handles this.
**Warning signs:** `tensor[~padding_mask]` (padded positions) contains non-zero values in the social context columns.

### Pitfall 5: Memory Pressure from Full DataFrame in DataLoader Workers
**What goes wrong:** DataLoader with `num_workers > 0` crashes with OOM or pickle errors.
**Why it happens:** pandas DataFrames don't multiprocess-pickle cleanly; 4.88M rows at multiple workers = 4+ copies in RAM.
**How to avoid:** Convert the context index to pure Python dicts/numpy in `_build_context_index` before DataLoader is created. Avoid holding the original `context_df` DataFrame on the Dataset after index construction (can set `self.context_df = None`).
**Warning signs:** DataLoader worker processes crash with `BrokenPipeError` or `RuntimeError: DataLoader worker`.

### Pitfall 6: Sequence Length Mismatch with Phase 1 sample_builder
**What goes wrong:** Dataset receives Phase 1 samples with `frames` shape (50, 8) but Phase 2 uses T=25.
**Why it happens:** `sample_builder.build_samples()` default is `sequence_length=50`. If Phase 2 Dataset uses T=25, the shapes are inconsistent.
**How to avoid:** Either (a) rebuild Phase 1 samples with `sequence_length=T` matching Phase 2, or (b) slice `sample["frames"][:T]` and `sample["padding_mask"][:T]` in `__getitem__`. The second option avoids rerunning the full pipeline.
**Warning signs:** Tensor shape assertion fails with (50, 50) instead of (25, 50).

---

## Code Examples

Verified patterns from official sources:

### PyTorch Dataset Subclass (Canonical Pattern)
```python
# Source: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
from torch.utils.data import Dataset

class DefensiveTrajectoryDataset(Dataset):
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Must return consistently shaped tensors for DataLoader default collate
        ...
```

### DataLoader Construction (both variants)
```python
# Source: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
from torch.utils.data import DataLoader

train_dataset_a = DefensiveTrajectoryDataset(
    samples=train_samples,
    context_df=context_df,
    sequence_length=25,
    include_ball_destination=False,   # Model A
)
train_dataset_b = DefensiveTrajectoryDataset(
    samples=train_samples,
    context_df=context_df,
    sequence_length=25,
    include_ball_destination=True,    # Model B
)

loader_a = DataLoader(train_dataset_a, batch_size=64, shuffle=True, num_workers=0)
loader_b = DataLoader(train_dataset_b, batch_size=64, shuffle=True, num_workers=0)

# Verify shapes
batch = next(iter(loader_a))
assert batch["input"].shape == (64, 25, 50), f"Model A shape wrong: {batch['input'].shape}"

batch = next(iter(loader_b))
assert batch["input"].shape == (64, 25, 52), f"Model B shape wrong: {batch['input'].shape}"
```

### FEAT-06 Leakage Prevention Test
```python
# Unit test: Model A input tensors contain zero ball-destination columns
def test_model_a_no_ball_destination(train_samples, context_df):
    dataset = DefensiveTrajectoryDataset(
        samples=train_samples[:100],
        context_df=context_df,
        include_ball_destination=False,
    )
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["input"].shape[-1] == 50, (
            f"Model A feature dim should be 50, got {item['input'].shape[-1]}"
        )
    # Negative check: ensure Model B has 52
    dataset_b = DefensiveTrajectoryDataset(
        samples=train_samples[:100],
        context_df=context_df,
        include_ball_destination=True,
    )
    for i in range(len(dataset_b)):
        item = dataset_b[i]
        assert item["input"].shape[-1] == 52
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.utils.data.Dataset` with in-`__getitem__` pandas queries | Pre-built lookup dict in `__init__`, pure numpy in `__getitem__` | Standard practice ~2019+ | 10-100x faster DataLoader iteration |
| `map`-style Dataset for all use cases | `IterableDataset` for streaming large datasets | PyTorch 1.2 | Not needed here — 68K samples fit in RAM |
| Manual train/val/test partition in Dataset | External split file loaded once, passed to Dataset | Standard | Already done in Phase 1 via splits.json |

**Deprecated/outdated:**
- Storing tensors as separate `.pt` files per sample then loading them in `__getitem__`: replaced by in-memory index for datasets this size (68K samples × 25 frames × 50 features ≈ 85 MB float32 — fits comfortably in RAM).

---

## Feature Count Derivation

This is the most important design decision for Phase 2 — the reasoning must be explicit:

```
Model A: (batch, T, 50)
Model B: (batch, T, 52) = Model A + ball_land_x + ball_land_y

Feature decomposition for 50:
  Own kinematics (8 features):
    x, y, s, a_computed, dir_sin, dir_cos, o_sin, o_cos

  Social context (42 features):
    21 other players × (x, y) = 42

  Total: 8 + 42 = 50 ✓

Why 21 other players × (x,y) only, not 22 × 4:
  - BDB 2026 tracks 9-17 players per play (never 22)
  - Using 22 × 4 = 88 contradicts the (batch, T, 50) requirement
  - Including only (x,y) for context players is a deliberate design choice:
    the model's job is to learn interaction from positions, not predict
    social kinematics
  - The target player is excluded from its own social context (target nflId)
  - Slots beyond actual players are zero-padded (deterministic by nflId sort)
```

---

## Data Facts (Confirmed from cleaned.parquet)

| Fact | Value | Source |
|------|-------|--------|
| cleaned.parquet total rows | 4,880,579 | Phase 1 execution |
| cleaned.parquet columns | 30 | Phase 1 execution |
| Defensive positions in strict filter | CB, FS, SS, LB | FEAT-01 requirement |
| CB rows | 1,056,888 | Verified from parquet |
| FS rows | 476,865 | Verified from parquet |
| SS rows | 392,421 | Verified from parquet |
| LB rows | 31 | Verified from parquet |
| Total samples after strict filter | 68,344 | Verified: train=52,779 / val=7,497 / test=8,068 |
| Typical players per play-frame | 12-13 (max 17) | Verified from parquet groupby |
| Median frames per player-play | 26 | Verified — 39% under 25, 96.6% under 50 |
| ball_land_x range (normalized) | -21.75 to +65.42 | Verified from parquet |
| ball_land_y range (normalized) | -30.68 to +30.56 | Verified from parquet |
| Null ball_land_x values | 0 | Verified from parquet |
| PyTorch version | 2.10.0+cpu | Installed in project env |

---

## Open Questions

1. **Sequence length T value**
   - What we know: Phase 1 sample_builder defaults to T=50; 96.6% of plays are under 50 frames; 39% under 25; median is 26 frames.
   - What's unclear: The requirements say "(batch, T, 50)" — T is left unspecified. T=25 captures the median well but pads 39% of samples. T=50 is the Phase 1 default but has high padding overhead for short plays.
   - Recommendation: Use T=25 as the default (matches the median, minimizes padding). Make it a configurable constant. Document the choice.

2. **Whether to rebuild Phase 1 samples or slice in Dataset**
   - What we know: Phase 1 build_samples() ran with sequence_length=50, so sample["frames"] has shape (50, 8).
   - What's unclear: Should Phase 2 rerun sample_builder with T=25, or slice in Dataset.__getitem__?
   - Recommendation: Slice in __getitem__ — avoids rerunning the 30-minute pipeline. If T changes, no pipeline re-run needed.

3. **ILB/OLB/MLB position handling in sample_builder**
   - What we know: sample_builder.DEFENSIVE_POSITIONS includes ILB/OLB/MLB (broad filter from Phase 1). The 93,824 count in STATE.md was from this broad filter.
   - What's unclear: Should we update DEFENSIVE_POSITIONS in sample_builder, or filter in Dataset constructor?
   - Recommendation: Filter in Dataset constructor only (pass `valid_positions={"CB","FS","SS","LB"}` to build_samples or filter the sample list before passing to Dataset). Keeps sample_builder backward-compatible.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` testpaths = ["tests"] |
| Quick run command | `pytest tests/test_dataset.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FEAT-01 | Only CB/FS/SS/LB appear in dataset | unit | `pytest tests/test_dataset.py::test_position_filter -x` | Wave 0 |
| FEAT-02 | Each player-play pair is an independent sample | unit | `pytest tests/test_dataset.py::test_player_play_independence -x` | Wave 0 |
| FEAT-03 | Fixed-length sequences with post-padding and masking flag | unit | `pytest tests/test_dataset.py::test_sequence_padding_and_mask -x` | Wave 0 |
| FEAT-04 | Social context assembled as 42-feature (21 other players × x,y) array | unit | `pytest tests/test_dataset.py::test_social_context_shape -x` | Wave 0 |
| FEAT-05 | ball_land features present in Model B, absent in Model A | unit | `pytest tests/test_dataset.py::test_ball_destination_model_b -x` | Wave 0 |
| FEAT-06 | Model A input tensors contain zero ball-destination columns (shape 50) | unit | `pytest tests/test_dataset.py::test_no_ball_leakage_model_a -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_dataset.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_dataset.py` — covers FEAT-01 through FEAT-06 (6 tests)
- [ ] `src/data/dataset.py` — the Dataset class itself (created in Wave 1)

*(Existing infrastructure: `tests/conftest.py` with `tracking_df` fixture, `tests/test_pipeline.py` with 8 passing tests. New conftest fixtures needed: `minimal_samples_fixture`, `minimal_context_df_fixture` for dataset unit tests.)*

---

## Sources

### Primary (HIGH confidence)
- **Verified against cleaned.parquet directly** — all data facts (row counts, position counts, column names, ball_land ranges, sample counts per split, players-per-play-frame distribution, frames-per-player-play distribution)
- **src/data/sample_builder.py** — Phase 1 implementation; feature columns, padding_mask structure, ball_target_xy extraction
- **src/data/preprocessor.py** — normalization pipeline; ball_land_x/y transformation confirmed
- **PyTorch docs (https://pytorch.org/docs/stable/data.html)** — Dataset/DataLoader API, `__getitem__` contract, DataLoader parameters
- **pyproject.toml** — confirmed torch==2.10.0, pandas==3.0.1, numpy==2.4 installed

### Secondary (MEDIUM confidence)
- **Phase 1 SUMMARYs (01-04, 01-05)** — confirmed ball landing column names, split counts, dataset format

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — torch 2.10.0 installed and verified, no new dependencies needed
- Feature count (50/52): HIGH — derived arithmetically from (batch, T, 50) requirement + verified player-per-play counts; consistent with Phase 1 code
- Architecture (single class, context index): HIGH — standard PyTorch pattern, verified against official docs
- Data facts (sample counts, position counts): HIGH — queried directly from cleaned.parquet
- Pitfalls: HIGH — most derived from direct inspection of Phase 1 code and data

**Research date:** 2026-03-13
**Valid until:** 2026-04-12 (stable domain — PyTorch Dataset API is stable, data facts are fixed)
