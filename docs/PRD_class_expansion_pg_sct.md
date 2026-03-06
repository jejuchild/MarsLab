# PRD: MarsLandformNet Class Expansion — Patterned Ground & Scalloped Terrain

**Version**: 1.0  
**Date**: 2026-03-05  
**Status**: Draft — awaiting approval  
**Author**: MarsLab AI Pipeline  

---

## 1. Overview

Expand the MarsLandformNet V4b classifier from **4 classes** (LDA, LVF, CCF, OTHER) to **6 classes** by adding:

| ID | Code | Full Name | Description |
|----|------|-----------|-------------|
| 4  | **PG** | Patterned Ground | Polygonal networks (thermal contraction cracks) visible in HiRISE, 5–30m diameter, indicative of near-surface ice |
| 5  | **SCT** | Scalloped Terrain | Asymmetric rimless depressions (thermokarst), 10m–1km, formed by sublimation of subsurface ice |

Both are **periglacial/thermokarst landforms** strongly associated with shallow ground ice — critical for ISRU prospecting.

### Current State

| Metric | Value |
|--------|-------|
| Model | FiLMClassifier (DINOv2-Base + LoRA + FiLM MOLA conditioning) |
| Classes | LDA (0), LVF (1), CCF (2), OTHER (3) |
| F1 (macro) | 0.776 |
| Training tiles | ~85K (Levy 2014 polygon labels) |
| Deploy checkpoint | `marslandform_v4b_deploy.pt` |

### Goal

| Metric | Target |
|--------|--------|
| Classes | LDA, LVF, CCF, **PG**, **SCT**, OTHER |
| Old-class F1 | ≥ 0.75 (max 2.6% regression from 0.776) |
| New-class F1 (PG) | ≥ 0.50 |
| New-class F1 (SCT) | ≥ 0.55 |
| Macro F1 (6-class) | ≥ 0.60 |
| Inference overhead | < 1ms additional (from 4→6 output neurons) |

---

## 2. Data Collection Plan

### 2.1 Scalloped Terrain (SCT) — Best Resourced

#### Primary: Wang et al. 2026 — Northern Hemisphere Polygon Map
- **Source**: [zenodo.18689392](https://doi.org/10.5281/zenodo.18689392)
- **Format**: GIS shapefile, polygon-level annotations
- **Coverage**: Entire Mars northern hemisphere (CTX-scale)
- **Method**: Deep-learning segmentation → polygon extraction, with morphometric attributes per depression
- **Approach**: 
  1. Download `Scalloped Terrain.zip` (~301 MB)
  2. Intersect SCT polygons with HiRISE footprints from `index.geojson` (18,013 images)
  3. For matching HiRISE images, tile (224×224) and label tiles overlapping SCT polygons as `SCT`
  4. Non-overlapping tiles in the same image → `OTHER`

#### Secondary: Séjourné 2018 — Utopia Planitia Grid Map
- **Source**: [zenodo.1404143](https://doi.org/10.5281/zenodo.1404143)
- **Format**: Grid-cell shapefile (~20 km cells), 13 landform classes including SCT
- **Use**: Regional context / weak labels to supplement Wang et al. — validate SCT locations match, identify high-SCT-density HiRISE footprints

#### Optional: Mantegazza 2025 — HiRISE-Resolution Utopia Planitia
- **Source**: [zenodo.17855988](https://doi.org/10.5281/zenodo.17855988)
- **Format**: 3 shapefiles (geomorphological map, meter-scale troughs, small polygons)
- **Coverage**: 42–52°N, 92–102°E
- **Risk**: Files may be restricted pending publication. Check availability before relying on this.

#### SCT Data Target
| Split | Tiles | Notes |
|-------|-------|-------|
| Train | ~8,000–15,000 | From Wang et al. polygon intersections |
| Val | ~1,500–3,000 | Spatially disjoint from train |
| Test | ~1,500–3,000 | Spatially disjoint from train+val |

### 2.2 Patterned Ground (PG) — Requires More Effort

**Problem**: No publicly available polygon-level HiRISE shapefile exists for patterned ground equivalent to Levy LDA/LVF/CCF.

#### Strategy A: HiRISE Catalog Mining (Primary)
1. **Query HiRISE catalog** via ODE REST API for images with titles/descriptions containing: `polygon`, `patterned ground`, `thermal contraction`, `periglacial polygon`
2. **Filter to mid-high latitudes** (40°–80°N, 40°–80°S) where PG is common
3. **Download browse images** and tile (224×224)
4. **Image-level labels**: All tiles from a PG-tagged HiRISE image get `PG` label (weak supervision)
5. **Cleanup**: Manual spot-check ~200 images, remove false positives

#### Strategy B: Bandeira Polygon Database (If Available)
- **Source**: Bandeira et al. 2014, USGS Astrogeology
- Contact: Trent Hare (hare@usgs.gov) or check JMARS layers
- If available: polygon-level outlines → direct tile labeling

#### Strategy C: Self-Labeling Bootstrapping
1. Train a preliminary PG binary classifier on Strategy A's weak labels
2. Apply to all mid-high-latitude HiRISE tiles
3. Human review top-confidence predictions → refine labels
4. Retrain with cleaned labels

#### PG Data Target
| Split | Tiles | Notes |
|-------|-------|-------|
| Train | ~5,000–10,000 | From HiRISE catalog + weak labels |
| Val | ~1,000–2,000 | Spatially disjoint |
| Test | ~1,000–2,000 | Spatially disjoint |

### 2.3 Old-Class Exemplar Buffer

To prevent catastrophic forgetting, we store exemplars from existing 4 classes:

| Class | Exemplars (train) | Strategy |
|-------|-------------------|----------|
| LDA | 2,000 | Stratified random from existing train tiles |
| LVF | 2,000 | Stratified random (all if fewer than 2K) |
| CCF | 2,000 | Stratified random |
| OTHER | 2,000 | Stratified random |

These are **sampled once** from the V4b training set and stored as a fixed replay buffer.

---

## 3. Model Integration Strategy

### 3.1 Chosen Approach: FC Expansion + Knowledge Distillation (LwF)

**Why**: Minimal architecture change, near-zero inference overhead, well-studied for class-incremental learning.

**Not chosen**:
- Multi-head (SplitCosineLinear): Guarantees zero old-class regression, but backbone features may not optimize for new classes
- Hierarchical classifier: Adds routing step (+5–15ms), error compounding, overkill for 6 classes
- LoRA adapters: Only beneficial with <500 new-class tiles; adds deployment complexity
- Full retrain from scratch: Works but wastes existing model quality and violates "minimize retraining" constraint

### 3.2 Architecture Changes

```
BEFORE (V4b):
  DINOv2-Base + LoRA → FiLM(MOLA 25-dim) → MLP(768→128→4)

AFTER (V5):
  DINOv2-Base + LoRA → FiLM(MOLA 25-dim) → MLP(768→128→6)
                                                          ^^
```

**Changes required**:

| Component | Change |
|-----------|--------|
| `FiLMClassifier.__init__` | `num_classes=4` → `num_classes=6` |
| `config.py` | `V3_CLASSES = ["LDA", "LVF", "CCF", "OTHER"]` → `["LDA", "LVF", "CCF", "PG", "SCT", "OTHER"]` |
| `config.py` | `V3_NUM_CLASSES = 4` → `6`, `V3_NUM_LANDFORM_CLASSES = 3` → `5` |
| `pipeline.py` | `V3_CLASSES` update, `CLASS_THRESHOLDS` add PG/SCT entries |
| Last linear layer | Initialize new rows (PG, SCT) with small Gaussian noise (σ=0.01), copy existing 4 rows from V4b checkpoint |

**FiLM layer**: No change needed. It modulates 768-dim visual features based on MOLA — completely class-agnostic. The MOLA signatures for PG (flat, high-latitude plains) and SCT (mid-latitude thermokarst depressions) are naturally distinct.

### 3.3 Training Protocol

```
Phase 1: Weight Initialization
  - Load marslandform_v4b_deploy.pt
  - Expand classifier.4 (nn.Linear): copy weights[:4,:] from V4b, init weights[4:6,:] ~ N(0, 0.01)
  - Freeze a copy of V4b as "teacher" model

Phase 2: Fine-Tune (Colab GPU)
  - Backbone LR: 1e-5 (very conservative — preserve learned features)
  - FiLM layers LR: 5e-5 (needs to learn PG/SCT MOLA signatures)
  - Classifier head LR: 1e-3
  - Loss: λ · KD_loss(logits[:4], teacher_logits[:4]) + CE_loss(logits[:6], targets)
  - λ = 3.0 (knowledge distillation weight — tune on val if needed)
  - Temperature T = 2 (for KD soft targets)
  - Batch composition: 50% new-class tiles + 50% old-class exemplars (stratified)
  - Epochs: 40–60
  - Early stopping: monitor old-class macro-F1; stop if drops > 3%
  - EMA decay: 0.996 (same as V4b)
  - Label smoothing: 0.1
  - MixUp: alpha=0.3 (within same class group only — don't mix PG with LDA)

Phase 3: Threshold Calibration
  - Run inference on val set
  - Compute per-class optimal thresholds (F1-maximizing)
  - Expected: PG threshold ~0.5–0.6, SCT threshold ~0.5–0.7
```

### 3.4 Confusion Risk Matrix

| New Class | Likely Confused With | Reason | Mitigation |
|-----------|---------------------|--------|------------|
| **PG** ↔ **CCF** | Both show regular surface textures | MOLA features differ: PG = flat plains, CCF = crater interiors | FiLM conditioning should separate |
| **SCT** ↔ **LDA** | Both show rounded/lobate depressions | Scale differs: SCT smaller, asymmetric rimless | Ensure diverse training scales |
| **PG** ↔ **OTHER** | Subtle polygon networks in low-res tiles | Confidence filtering helps | Set higher PG threshold |
| **SCT** ↔ **OTHER** | Partially degraded scallops | Wang et al. data includes size attributes → filter tiny ones | Size-based quality filter |

---

## 4. Implementation Plan

### Phase A: Data Collection & Preparation (2–4 hours)

| Step | Action | Output |
|------|--------|--------|
| A.1 | Download Wang et al. 2026 SCT shapefile from Zenodo | `Data/external_datasets/wang_2026_scalloped/` |
| A.2 | Download Séjourné 2018 Utopia grid from Zenodo | `Data/external_datasets/sejourne_2018_utopia/` |
| A.3 | Intersect SCT polygons with HiRISE footprints | List of HiRISE image IDs containing SCT |
| A.4 | Download HiRISE browse images for SCT matches (aria2c) | `Data/HiRISE/midlat_browse/` |
| A.5 | Query ODE API for PG-tagged HiRISE images | List of PG HiRISE image IDs |
| A.6 | Download PG browse images (aria2c) | `Data/HiRISE/midlat_browse/` |
| A.7 | Generate tile labels for SCT (polygon intersection) | `tile_labels_sct.json` |
| A.8 | Generate tile labels for PG (image-level weak labels) | `tile_labels_pg.json` |
| A.9 | Merge with existing V4b labels, create unified `tile_labels_v5.json` | `tile_labels_v5.json` |
| A.10 | Spatial split (train/val/test, 20km exclusion radius) | `tile_splits_v5.json` |
| A.11 | Sample 2K exemplars per old class from V4b train set | `exemplar_buffer_v5.json` |

### Phase B: Model Training (Colab GPU, ~2–4 hours)

| Step | Action | Output |
|------|--------|--------|
| B.1 | Prepare Colab data package (tiles + MOLA + labels + splits + V4b checkpoint) | `colab_v5_data.tar.gz` |
| B.2 | Create `colab_v5_expansion_training.ipynb` with LwF protocol | Notebook |
| B.3 | Train on Colab: DINOv2+LoRA backbone + FiLM + 6-class head | `marslandform_v5_best.pt` |
| B.4 | Export deploy checkpoint | `marslandform_v5_deploy.pt` |

### Phase C: Integration & Evaluation (~1–2 hours)

| Step | Action | Output |
|------|--------|--------|
| C.1 | Update `config.py`: V5 class definitions (6 classes) | |
| C.2 | Update `pipeline.py`: load V5, add PG/SCT thresholds | |
| C.3 | Update `heatmap.py`: add PG/SCT colors | |
| C.4 | Update `hirise-api/`: add PG/SCT to report rendering | |
| C.5 | Run eval on test set — per-class F1, confusion matrix | `eval_v5_report.json` |
| C.6 | Spot-check: run API on 10 known PG/SCT HiRISE images | Visual verification |
| C.7 | Update `index.geojson` if new images were downloaded | |

---

## 5. File Changes

### New Files
```
scripts/marslandform_v2/prepare_v5_colab_data.py      # Data packaging
scripts/marslandform_v2/data/collect_pg_sct_labels.py  # Label generation for new classes
scripts/marslandform_v2/colab_v5_expansion_training.ipynb  # Training notebook
Data/external_datasets/wang_2026_scalloped/            # SCT shapefile
Data/external_datasets/sejourne_2018_utopia/           # Utopia grid
```

### Modified Files
```
scripts/marslandform_v2/config.py                      # V5 class defs, 6 classes
scripts/marslandform_v2/models/film_classifier.py      # num_classes=6 (constructor param only)
backend/analysis/hirise_landforms/pipeline.py          # V5 checkpoint, V5_CLASSES, thresholds
backend/analysis/hirise_landforms/heatmap.py           # PG/SCT colors
backend/analysis/hirise_landforms/models.py            # CLASS_THRESHOLDS PG/SCT entries
hirise-api/routers/analyze.py                          # Report rendering for 6 classes
hirise-api/core/crism_bridge.py                        # Landform→ISRU score mapping for PG/SCT
```

---

## 6. Edge Cases & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Wang et al. SCT polygons are CTX-scale, not HiRISE-scale | Intersection yields imprecise tile labels (some "SCT" tiles may not show scallops at HiRISE res) | Accept as weak labels; confidence threshold at inference filters false positives |
| PG image-level labels are noisy (not all tiles in a "polygon" image show polygons) | ~30–50% label noise for PG class | Bootstrap cleaning (Phase C self-labeling), or use MixUp to soften label noise |
| Catastrophic forgetting despite KD | Old-class F1 drops > 3% | Increase λ (KD weight), add more exemplars, or fall back to multi-head approach |
| Too few PG training tiles (<3K) | PG F1 < 0.3 | Augment aggressively (rotation, flip, scale); lower target to F1≥0.35 for V5, improve in V6 |
| Mantegazza 2025 data restricted | Lose HiRISE-res SCT polygons | Wang et al. CTX-scale data is sufficient; Séjourné supplements |
| PG ↔ CCF confusion | Both have regular textures | Add hard-negative mining: CCF tiles near PG latitude range as explicit negatives |

---

## 7. Success Criteria

| Criterion | Threshold | Validation Method |
|-----------|-----------|-------------------|
| Old-class macro-F1 (LDA+LVF+CCF) | ≥ 0.75 | Test set eval |
| PG class F1 | ≥ 0.50 | Test set eval |
| SCT class F1 | ≥ 0.55 | Test set eval |
| 6-class macro-F1 | ≥ 0.60 | Test set eval |
| Inference latency | < 0.4s per tile (was 0.36s) | Profiling |
| Old-class confusion matrix | No class with >5% new-class contamination | Confusion matrix |
| MarsLab + hirise-api | All endpoints return 200 with 6-class output | Browser verification |

---

## 8. Out of Scope (for this PRD)

- [ ] Context module (neighbor tile conditioning) — deferred, requires separate perf comparison
- [ ] Bayesian MOLA post-processing — explicitly rejected by user
- [ ] CRISM mineral model changes — unrelated; CRISM pipeline stays as-is
- [ ] ISRU accessibility formula rework — PG/SCT will use existing `ice_landform` sub-score with added class mappings
- [ ] Rover-scale classification (S5Mars) — different modality, not applicable

---

## 9. Timeline Estimate

| Phase | Duration | Blocker |
|-------|----------|---------|
| A: Data Collection | 2–4 hours | Zenodo download speed; ODE API rate limits |
| B: Training | 2–4 hours | Colab GPU availability |
| C: Integration | 1–2 hours | None (straightforward code changes) |
| **Total** | **5–10 hours** | |

---

## 10. Version Naming

- **V5**: 6-class model with PG + SCT (this PRD)
- Model file: `marslandform_v5_deploy.pt`
- Config key: `V5_CLASSES = ["LDA", "LVF", "CCF", "PG", "SCT", "OTHER"]`
