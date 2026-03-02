# MarsLab — SWIM Ice Detection & HiRISE Landform Integration PRD

**Version**: 1.0
**Date**: 2026-03-02
**Status**: Draft — Awaiting Confirmation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Goals](#2-goals)
3. [Scope](#3-scope)
4. [Architecture Overview](#4-architecture-overview)
5. [Module A: HiRISE Landform Classification](#5-module-a-hirise-landform-classification)
6. [Module B: Neutron Spectroscopy Ice Detection](#6-module-b-neutron-spectroscopy-ice-detection)
7. [Module C: TES Thermal Inertia Ice Indicator](#7-module-c-tes-thermal-inertia-ice-indicator)
8. [Module D: SHARAD Radar Surface Power](#8-module-d-sharad-radar-surface-power)
9. [Module E: SHARAD Radar Dielectric (Subsurface)](#9-module-e-sharad-radar-dielectric-subsurface)
10. [Module F: Geomorphic Ice Mapping](#10-module-f-geomorphic-ice-mapping)
11. [Module G: SWIM Consistency Fusion](#11-module-g-swim-consistency-fusion)
12. [Frontend Components](#12-frontend-components)
13. [File Structure](#13-file-structure)
14. [Data Dependencies](#14-data-dependencies)
15. [Implementation Phases](#15-implementation-phases)
16. [Edge Cases & Failure Modes](#16-edge-cases--failure-modes)
17. [References](#17-references)

---

## 1. Executive Summary

Integrate two major science capabilities into MarsLab:

**A) HiRISE Landform Classification** — Expose the existing MarsLandformNet V2 model (DINOv2 + MIL classifier) as a backend API module. Users select a HiRISE image on the map → backend runs inference → returns per-tile class predictions (LDA, LVF, CCF, GLF, Background) with attention heatmaps overlaid on the map.

**B) SWIM Subsurface Water Ice Mapping** — Integrate all six ice-detection methods from the SWIM project (Morgan & Putzig et al. 2025, PSJ 6:29) into MarsLab's existing ice analysis infrastructure. Each method becomes a queryable layer. A fusion module combines them into per-pixel ice consistency scores at three depth ranges (0–1 m, 1–5 m, >5 m).

MarsLab already has partial infrastructure for both: `scripts/marslandform_v2/` contains the trained model pipeline; `swim_router.py` serves pre-rendered SWIM tiles; `thermal_inertia.py` provides TES point queries; `ice_evidence/` implements multi-criteria fusion. This PRD designs the complete integration that ties them into the web UI.

---

## 2. Goals

### HiRISE Landform Classification
- Classify glacial/periglacial landforms (LDA, LVF, CCF, GLF) in HiRISE browse images via the web UI
- Display per-tile attention heatmaps showing which image regions drove the classification
- Make classification results queryable alongside existing instrument data in the Inspector panel
- Support both the custom MarsLandformNet V2 model and Mars-Bench ViT (HuggingFace, Apache 2.0) as selectable backends

### SWIM Ice Detection
- Provide per-pixel ice evidence scores for each of the six SWIM methods
- Compute composite ice consistency maps at three depth ranges (0–1 m, 1–5 m, >5 m)
- Allow users to toggle individual method layers on/off to understand which evidence drives the score
- Integrate with existing thermal inertia, SHARAD, and ice evidence modules — extend, don't duplicate
- Serve SWIM products at ~3 km/pixel resolution over the study region (60°S–60°N, <+1 km elevation)

---

## 3. Scope

### In Scope
- Backend API endpoints for all 7 modules (A through G)
- Frontend layer controls, map overlays, and Inspector panel integration
- Data ingestion pipelines for SWIM GeoTIFF products from swim.psi.edu
- HiRISE landform inference on user-selected images
- Ice consistency fusion computation from individual method scores

### Out of Scope
- Raw instrument data processing from PDS (SHARAD EDR SAR focusing, GRS raw counts)
  - We consume pre-derived SWIM products and existing MarsLab data, not raw PDS Level 0
- Retraining ML models (use existing V2 weights or Mars-Bench pretrained)
- The probabilistic SWIM framework (Courville et al. LPSC 2026) — future work
- MARSIS integration (SHARAD only for this phase)
- Mobile/tablet-specific UI

---

## 4. Architecture Overview

### Existing Patterns (MUST follow)

**Analysis Module Pattern:**
```
backend/analysis/<module_name>/
├── __init__.py          # Public API exports
├── models.py            # Pydantic request/response models
├── pipeline.py          # Core computation logic
└── utils.py             # Module-specific helpers (optional)
```

**Router Pattern:**
```python
# backend/api/<module>_router.py
from fastapi import APIRouter
router = APIRouter(prefix="/api/<module>", tags=["<Module Name>"])
```

**Registration in app.py:**
```python
from api.<module>_router import router as <module>_router
app.include_router(<module>_router)
```

**Data Storage:**
- Pre-computed grids: NumPy `.npy` or GeoTIFF in `backend/data/<module>/`
- Spatial indices: GeoJSON in `backend/data/<module>/index.geojson`
- Configuration: Module constants in pipeline or config files

**Frontend Pattern:**
- API client: `frontend/src/api/<module>.ts`
- Map layer: CesiumJS ImageryProvider or Entity overlay
- Panel component: `frontend/src/components/<ModuleName>Panel.tsx`
- Integration: Added to MainPage layer controls and Inspector tabs

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (React + CesiumJS)                 │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐│
│  │ Layer Toggle  │  │ Inspector    │  │ SWIM Ice Consistency Panel ││
│  │ Panel (per    │  │ Panel (point │  │ (composite maps, depth     ││
│  │ SWIM method)  │  │ query data)  │  │ range selector, legend)    ││
│  └──────┬───────┘  └──────┬───────┘  └────────────┬───────────────┘│
│         │                 │                        │                │
│  ┌──────┴─────────────────┴────────────────────────┴───────────────┐│
│  │                    API Client Layer                              ││
│  │  swim.ts · hirise_landforms.ts · thermal_inertia.ts (existing)  ││
│  └──────────────────────────┬──────────────────────────────────────┘│
└─────────────────────────────┼──────────────────────────────────────┘
                              │ HTTP
┌─────────────────────────────┼──────────────────────────────────────┐
│                      FastAPI Backend                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      API Routers                              │   │
│  │  hirise_landforms_router  swim_ice_router  swim_router(ext)   │   │
│  │  thermal_inertia(ext)     sharad_highres(ext)                 │   │
│  └──────┬───────────────────────┬───────────────────────────────┘   │
│         │                       │                                    │
│  ┌──────┴───────────────────────┴───────────────────────────────┐   │
│  │                    Analysis Modules                           │   │
│  │  hirise_landforms/   swim_neutron/    swim_sharad_surface/    │   │
│  │  swim_sharad_dielectric/  swim_geomorphic/  swim_fusion/      │   │
│  │  ice_evidence/(ext)  thermal_inertia.py(ext)                  │   │
│  └──────┬───────────────────────┬───────────────────────────────┘   │
│         │                       │                                    │
│  ┌──────┴───────────────────────┴───────────────────────────────┐   │
│  │                    Data Layer                                 │   │
│  │  backend/data/swim/         (GeoTIFF products, ~3 km/px)     │   │
│  │  backend/data/hirise_landforms/  (model weights, tile cache)  │   │
│  │  backend/data/tes_ti/       (existing TES 20ppd grid)         │   │
│  │  backend/data/sharad/       (existing SHARAD RDR index)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Module A: HiRISE Landform Classification

### Overview
Wrap the existing MarsLandformNet V2 pipeline (in `scripts/marslandform_v2/`) as a backend analysis module that accepts a HiRISE browse image URL or product ID, runs DINOv2 + MIL inference, and returns classification results.

### Data Source
- **Input**: HiRISE browse images (JPEG, ~1024×1024 to ~4096×4096 px) fetched from HiRISE browse server or cached locally
- **Auxiliary**: MOLA DEM topographic features (23 features: slope, TPI, TRI, roughness, curvature at 3 scales + 2 global) — extracted from existing MarsLab DEM data
- **Model weights**: MarsLandformNet V2 checkpoint in `backend/data/hirise_landforms/weights/`
- **Alternative weights**: Mars-Bench ViT checkpoint from HuggingFace (`Mirali33/mars-bench-landmark-classification`)

### Algorithm

```
1. Receive HiRISE product ID or image URL
2. Fetch browse image → resize to 224×224 tile grid (V2 config: TILE_SIZE=224)
3. For each tile:
   a. Extract DINOv2-ViT-B/14 embeddings (768-dim) via frozen backbone (or LoRA-finetuned)
   b. Extract MOLA topographic features (23-dim) from DEM at tile center coordinates
4. Feed all tile embeddings + topo features into MIL AttentionClassifier:
   a. Multi-head gated attention computes per-tile attention weights
   b. Cross-modal gated residual fusion merges visual + topographic
   c. Image-level class probabilities via weighted bag representation
5. Return: top class, class probabilities, per-tile attention map, confidence
```

### Model Architecture (from existing codebase)

| Component | Config |
|---|---|
| Backbone | DINOv2-ViT-B/14 (768-dim), optional LoRA (rank=8, alpha=16) |
| MIL Classifier | AttentionMILClassifier: gated attention (4 heads), FC layers (768→256→5) |
| Topographic features | 23 MOLA features per tile (3 scales × 7 metrics + 2 global) |
| Fusion | GatedResidualFusion: visual + topo with learnable gate |
| Loss | Focal Loss (gamma=2.0, alpha per class, label smoothing=0.05) |
| Classes | 5: LDA, LVF, CCF, GLF, BACKGROUND |
| Input | 224×224 px tiles (ImageNet-normalized) |

### Mars-Bench Alternative

| Component | Config |
|---|---|
| Architecture | Vision Transformer (ViT) |
| Classes | 7: other, crater, dark_dune, streak, bright_dune, impact, edge |
| Weights | `Mirali33/mars-bench-landmark-classification` (Apache 2.0) |
| Input | TBD (inspect checkpoint — likely 224×224) |

**Note**: The two models classify **different landform taxonomies**. V2 targets glacial features (LDA/LVF/CCF/GLF) for ice science. Mars-Bench targets general landmarks (crater/dune/streak). The user should select which model to use. Default: V2 for ice-related analysis, Mars-Bench for general classification.

### API Endpoints

```
POST /api/hirise-landforms/classify
  Body: { product_id: str, model: "v2" | "mars-bench", include_heatmap: bool }
  Response: {
    job_id: str,                           // async job ID
    status: "queued",
    estimated_seconds: int                  // ~10-30s depending on image size
  }

GET /api/hirise-landforms/jobs/{job_id}
  Response (pending): {
    job_id: str,
    status: "queued" | "processing",
    progress: float,                        // 0-1, tile processing progress
    submitted_at: str                       // ISO timestamp
  }
  Response (completed): {
    job_id: str,
    status: "completed",
    result: {
      product_id: str,
      model_used: str,
      prediction: {
        top_class: str,
        probabilities: { class_name: float },
        confidence: float
      },
      tiles: [
        {
          x: int, y: int,                    // tile grid position
          attention_weight: float,            // 0-1, MIL attention
          lat: float, lon: float             // tile center coordinates
        }
      ],
      heatmap_url: str | null               // pre-rendered attention overlay PNG
    }
  }
  Response (failed): {
    job_id: str,
    status: "failed",
    error: str
  }

GET /api/hirise-landforms/status
  Response: {
    models_loaded: [str],
    device: str,
    memory_mb: float,
    queue_length: int,                      // number of pending jobs
    active_job: str | null                  // currently processing job_id
  }

GET /api/hirise-landforms/classify/{product_id}
  Response: cached result if previously classified, else 404
```

### Existing Code to Wrap

| Source File | What to Extract |
|---|---|
| `scripts/marslandform_v2/models/dinov2_lora.py` | DinoV2LoRA class → backbone |
| `scripts/marslandform_v2/models/mil_classifier.py` | AttentionMILClassifier → classifier |
| `scripts/marslandform_v2/models/embedder.py` | Tile embedding pipeline |
| `scripts/marslandform_v2/config.py` | All hyperparameters, class names, feature configs |
| `scripts/landform_pipeline/predict.py` | Reference for inference flow, DEM feature extraction, overlay generation |
| `scripts/landform_pipeline/fusion.py` | Bayesian CNN+DEM logit fusion (reference pattern) |

### New Files

```
backend/analysis/hirise_landforms/
├── __init__.py
├── models.py              # Pydantic: ClassifyRequest, ClassifyResponse, TileResult, JobStatus
├── pipeline.py            # Inference: load model, tile image, run MIL, return results
├── preprocessing.py       # Image fetch, tiling, DEM feature extraction
├── heatmap.py             # Attention map → overlay PNG generation
└── job_queue.py           # Async job queue: submit, poll, cleanup

backend/api/hirise_landforms_router.py   # FastAPI router

backend/data/hirise_landforms/
├── weights/               # V2 checkpoint + Mars-Bench checkpoint
└── cache/                 # Cached classification results (JSON per product)
```

### Edge Cases
- **No model weights found**: Return 503 with message indicating weights need to be placed in `data/hirise_landforms/weights/`
- **Image fetch failure**: Return 502 with upstream error. Cache failures to avoid repeated retries.
- **GPU not available**: Fall back to CPU inference (slower but functional). Log warning at startup.
- **Image too small** (<224 px in either dimension): Return 422 with minimum size requirement.
- **Concurrent requests**: Async job queue with single-worker processing. DINOv2 uses ~2GB VRAM — only 1 inference runs at a time. Additional requests are queued (FIFO). Max queue depth: 10 — return 429 if exceeded.
- **Job expiry**: Completed job results cached for 24h (keyed by product_id + model_version). Stale jobs cleaned up on startup.
- **Long-running inference**: Frontend polls `GET /jobs/{job_id}` every 2s. Backend updates `progress` as tiles are processed (progress = tiles_done / total_tiles).

---

## 6. Module B: Neutron Spectroscopy Ice Detection

### Overview
Serve Mars Odyssey GRS/MONS epithermal neutron data as a queryable ice-indicator layer. Epithermal neutron suppression indicates hydrogen (water ice) in the top ~1 m of regolith. This is the coarsest SWIM input (~300–500 km native resolution) but provides unique depth-to-ice constraints.

### Data Source
- **Primary**: SWIM neutron consistency GeoTIFF from swim.psi.edu (already processed to ~3 km/pixel)
- **Secondary** (for point-query detail): MONS Improved Derived Neutron Data (IDND), PDS4 bundle `urn:nasa:pds:mesick_pdart19_odyssey_idnd`
- **Existing MarsLab data**: `mars_science_context.json` already contains per-region `neutron_h2o_wt_pct` values

### Algorithm
The SWIM neutron layer uses Feldman et al. (2004) epithermal neutron suppression:

```
1. Load SWIM neutron GeoTIFF (pre-computed consistency scores)
2. For point queries at (lat, lon):
   a. Sample the neutron consistency grid at nearest pixel
   b. Return: consistency score (-1 to +1), water-equivalent hydrogen (wt%)
3. For region queries:
   a. Extract bounding-box subgrid
   b. Return: statistics (mean, std, min, max) + raster tile for overlay
```

**Interpretation thresholds** (from Feldman et al. 2004, Pathare et al. 2018):
- WEH > 10 wt%: strong ice indicator (C_neutron = +1)
- WEH 3–10 wt%: ambiguous (C_neutron = 0)
- WEH < 3 wt%: inconsistent with near-surface ice (C_neutron = −1)

### API Endpoints

```
GET /api/swim-ice/neutron/point?lat=X&lon=Y
  Response: {
    lat: float, lon: float,
    consistency_score: float,       # -1 to +1
    water_equivalent_h: float,      # wt%
    depth_range: "0-1m",
    data_quality: "nominal" | "interpolated" | "no_data"
  }

GET /api/swim-ice/neutron/region?north=N&south=S&east=E&west=W
  Response: {
    bounds: { north, south, east, west },
    stats: { mean, std, min, max, coverage_pct },
    tile_url: str                    # PNG overlay tile
  }

GET /api/swim-ice/neutron/tile/{z}/{x}/{y}.png
  Response: colored PNG tile for map overlay
```

### New Files

```
backend/analysis/swim_neutron/
├── __init__.py
├── models.py              # NeutronPointResponse, NeutronRegionResponse
├── pipeline.py            # Load GeoTIFF, point sample, region extract, tile render

backend/data/swim/
├── neutron_consistency.tif    # SWIM neutron GeoTIFF (~3 km/px)
└── neutron_metadata.json      # Source, version, no-data value, CRS info
```

### Edge Cases
- **Query outside 60°S–60°N**: Return `data_quality: "no_data"` — SWIM coverage limit
- **Query above +1 km elevation**: Return `data_quality: "no_data"` — SWIM excludes high elevations
- **No-data pixels** (value = −30 in SWIM products): Return `consistency_score: null`
- **Low native resolution**: Warn client that spatial precision is ~300 km despite ~3 km grid

---

## 7. Module C: TES Thermal Inertia Ice Indicator

### Overview
Extend the existing `thermal_inertia.py` module to serve SWIM-specific ice consistency scores derived from TES seasonal apparent thermal inertia (ATI) patterns. The existing module provides raw TI point queries at 20 ppd. The extension adds ice-interpretation logic from the SWIM framework.

### Data Source
- **Existing**: TES 20 ppd thermal inertia grid already loaded in `thermal_inertia.py` as numpy array
- **New**: SWIM thermal consistency GeoTIFF from swim.psi.edu (combines 3 independent thermal ice maps: Bandfield & Feldman 2008, SWIM TES analysis, Piqueux et al. 2019 MCS)
- **Existing context**: `mars_science_context.json` has per-region `thermal_inertia` values

### Algorithm
SWIM thermal ice detection uses a two-layer model (dry regolith over ice-cemented ground):

```
1. MARSTHERM forward model: finite-difference 1D thermal diffusion
   - Inputs: TI, albedo, dust opacity, surface pressure, latitude, Ls
   - Soil heat capacity: 627.9 J/kg/K, density: 1500 kg/m³
   - Time steps: 144/day, convergence: 3 Mars years
2. Seasonal ATI heterogeneity classification:
   - Extract ATI at multiple Ls (solar longitudes) from TES observations
   - Compare seasonal curve to lookup table of two-material models
   - Classify pixel as: uniform, layered-ice-consistent, dry
3. Combined score from 3 independent analyses → C_thermal ∈ {-1, 0, +1}
```

**For MarsLab**, we consume the pre-computed SWIM thermal GeoTIFF rather than running MARSTHERM:

```
1. Load SWIM thermal consistency GeoTIFF
2. For point queries:
   a. Sample SWIM thermal consistency at (lat, lon)
   b. Also sample raw TI from existing 20 ppd grid
   c. Return both: SWIM consistency score + raw TI value + ice interpretation
3. SWIM consistency values already encode ice interpretation:
   - C_thermal = +1: ice-cemented ground consistent
   - C_thermal = 0: ambiguous
   - C_thermal = -1: inconsistent with ice
4. Reference TI thresholds (for display context only, NOT for computing C_thermal):
   - TI > 600 TIU at mid-latitudes → possible ice-cemented
   - TI 200–600 TIU → ambiguous
   - TI < 200 TIU → fine dust, unlikely ice
```

### API Endpoints (extend existing)

```
GET /api/thermal-inertia/ice-score?lat=X&lon=Y
  Response: {
    lat: float, lon: float,
    thermal_inertia_tiu: float,        # raw TI from existing module
    swim_consistency: float,            # -1 to +1 from SWIM GeoTIFF
    interpretation: str,                # "ice_cemented" | "ambiguous" | "dry_fines"
    depth_range: "0-1m"
  }

GET /api/swim-ice/thermal/tile/{z}/{x}/{y}.png
  Response: colored PNG tile for SWIM thermal consistency overlay

GET /api/swim-ice/thermal/region?north=N&south=S&east=E&west=W
  Response: { bounds, stats, tile_url }
```

### Changes to Existing Files

| File | Change |
|---|---|
| `backend/api/thermal_inertia.py` | Add `/ice-score` endpoint, import SWIM thermal pipeline |
| `backend/data/swim/thermal_consistency.tif` | New: SWIM thermal GeoTIFF |

### New Files

```
backend/analysis/swim_thermal/
├── __init__.py
├── models.py              # ThermalIceResponse
├── pipeline.py            # Load SWIM thermal GeoTIFF, score interpretation
```

---

## 8. Module D: SHARAD Radar Surface Power

### Overview
Serve SHARAD surface reflectivity data as an ice-indicator layer. High surface reflectivity (relative to a smooth-surface Hagfors model) indicates a high-dielectric material at/near the surface — consistent with ice-rich regolith in the top ~1 m.

### Data Source
- **Primary**: SWIM radar surface power consistency GeoTIFF from swim.psi.edu
- **Existing**: `sharad_highres_router.py` already parses SHARAD RDR radargrams and can extract surface power from individual tracks
- **Existing**: `mars_science_context.json` has per-region `radar_dielectric` values

### Algorithm
SWIM surface power extraction (reference: SOPA `surfPow` module):

```
1. For each SHARAD track:
   a. Range compression of Italian EDR (pulse compression in frequency domain)
   b. Detect nadir first return (surface echo)
   c. Extract surface power at nadir within Fresnel zone window
   d. Correct for Fresnel zone spreading
   e. Compare to Hagfors smooth-surface theoretical model
2. Excess power = observed - modeled → map to dielectric anomaly
3. Gridded product at ~3 km/pixel: SWIM integrates thousands of tracks
4. Consistency: C_radar_surface = +1 if excess power exceeds threshold
```

**For MarsLab**, we serve the pre-computed SWIM radar surface power GeoTIFF. For individual track inspection, we extend the existing SHARAD router.

### API Endpoints

```
GET /api/swim-ice/radar-surface/point?lat=X&lon=Y
  Response: {
    lat: float, lon: float,
    consistency_score: float,          # -1 to +1
    surface_power_excess_db: float,    # dB above Hagfors model
    depth_range: "0-1m",
    nearest_track_id: str | null       # closest SHARAD track for detail view
  }

GET /api/swim-ice/radar-surface/tile/{z}/{x}/{y}.png
  Response: colored PNG tile

GET /api/swim-ice/radar-surface/region?north=N&south=S&east=E&west=W
  Response: { bounds, stats, tile_url }
```

### New Files

```
backend/analysis/swim_sharad_surface/
├── __init__.py
├── models.py              # RadarSurfacePointResponse
├── pipeline.py            # Load SWIM radar surface GeoTIFF, sample, tile

backend/data/swim/
├── radar_surface_consistency.tif
```

---

## 9. Module E: SHARAD Radar Dielectric (Subsurface)

### Overview
Serve SHARAD subsurface dielectric constant estimates as ice indicators for 1–5 m and >5 m depth ranges. The dielectric constant ε' estimated from radar two-way travel time and geometric depth constraints distinguishes ice (ε' ≈ 3.15) from rock (ε' ≈ 7.5–9).

### Data Source
- **Primary**: SWIM radar dielectric consistency GeoTIFF from swim.psi.edu
- **Existing**: `analysis/ice_evidence/sharad_reflectors.py` already detects subsurface reflectors and estimates signal-to-noise
- **Existing**: `analysis/epsilon_terrace/` implements dielectric inversion from SHARAD
- **Existing**: `analysis/radar_attenuation/` computes SHARAD attenuation maps

### Algorithm
SHARAD dielectric estimation (Bramson & Petersen, SWIM team):

```
ε' = (c · δt / 2 · δx)²

where:
  c     = 299,792,458 m/s (speed of light)
  δt    = two-way travel time to subsurface reflector (seconds)
  δx    = geometric depth to reflector (meters)

Depth estimation methods (3, context-dependent):
  1. Mantle edges: interpolate basement elevation between exposed edges
  2. Apron/scarp geometry: linear regression of bedrock slope under feature
  3. Point estimates: from layer exposures in terraced craters/fossae

Consistency thresholds:
  ε' < 4.5   → ice-consistent (C_radar_dielectric = +1)
  ε' 4.5–6.0 → ambiguous (C_radar_dielectric = 0)
  ε' > 6.0   → rock/regolith (C_radar_dielectric = −1)
  Median for confirmed LDA ice: ε' = 3.0 (nearly pure ice)
```

**For MarsLab**, we serve the pre-computed SWIM dielectric GeoTIFF and cross-reference with existing MarsLab SHARAD analysis modules for track-level detail.

### API Endpoints

```
GET /api/swim-ice/radar-dielectric/point?lat=X&lon=Y
  Response: {
    lat: float, lon: float,
    consistency_score_1_5m: float,     # -1 to +1 for 1-5m depth
    consistency_score_5m_plus: float,  # -1 to +1 for >5m depth
    estimated_epsilon: float | null,   # dielectric constant if available
    depth_ranges: ["1-5m", ">5m"],
    nearest_track_id: str | null
  }

GET /api/swim-ice/radar-dielectric/tile/{z}/{x}/{y}.png?depth=1-5m|5m-plus
  Response: colored PNG tile (depth-range specific)

GET /api/swim-ice/radar-dielectric/region?north=N&south=S&east=E&west=W&depth=1-5m|5m-plus
  Response: { bounds, stats, tile_url }
```

### Integration with Existing Modules

| Existing Module | How to Integrate |
|---|---|
| `analysis/ice_evidence/sharad_reflectors.py` | Cross-reference SWIM dielectric with track-level reflector SNR |
| `analysis/epsilon_terrace/` | Use existing ε' inversion for individual track detail views |
| `analysis/radar_attenuation/` | Supplement SWIM with attenuation-based ice estimates |

### New Files

```
backend/analysis/swim_sharad_dielectric/
├── __init__.py
├── models.py              # DielectricPointResponse
├── pipeline.py            # Load SWIM dielectric GeoTIFFs (2 depth ranges), sample, tile
```

---

## 10. Module F: Geomorphic Ice Mapping

### Overview
Serve SWIM geomorphic mapping data as ice-indicator layers. Periglacial and glacial landforms (polygons, scalloped terrain, LDAs, CCFs, etc.) indicate present or recent ice at various depths. SWIM maps 10 landform types for shallow ice (0–5 m) and 8 for deep ice (>5 m).

This module has a **dual role**:
1. **Serve SWIM geomorphic GeoTIFFs** (pre-computed grid-mapping scores)
2. **Link to HiRISE landform classification** (Module A) for automated detection of LDA/LVF/CCF/GLF — the deep-ice geomorphic indicators

### Data Source
- **Primary**: SWIM geomorphology GeoTIFFs from swim.psi.edu:
  - `geomorphology_0_1m.tif` (shallow, 10 landform types weighted)
  - `geomorphology_1_5m.tif` (intermediate)
  - `geomorphology_5m_plus.tif` (deep, 8 landform types weighted)
- **Supplementary**: Individual landform maps from SWIM (per-landform ancillary GeoTIFFs)
- **Cross-link**: MarsLandformNet V2 classifications (Module A) for automated LDA/LVF/CCF/GLF detection

### Landform Types (SWIM4MIM)

| Landform | Depth Indicator | Detection Method |
|---|---|---|
| Thermal contraction crack polygons | 0–1 m | CTX + HiRISE manual mapping / ML |
| Sublimation pits | 0–1 m | CTX manual mapping |
| Smooth mantling material | 0–1 m | CTX texture analysis |
| Dissected/pitted mantle | 0–1 m | CTX manual mapping |
| Scalloped terrain | 1–5 m | CTX thermokarst mapping |
| Viscous flow features (VFF) | 1–5 m | CTX morphological mapping |
| Lobate Debris Aprons (LDA) | >5 m | CTX/HiRISE + SHARAD ε' |
| Lineated Valley Fill (LVF) | >5 m | CTX/HiRISE + SHARAD ε' |
| Concentric Crater Fill (CCF) | >5 m | CTX/HiRISE + SHARAD ε' |
| Glacier-like forms (GLF) | >5 m | CTX/HiRISE morphological |

### Algorithm

```
SWIM Grid-Mapping (Ramsdale et al. 2017):
1. Survey area divided into cells (~20×20 km or ~40×40 km)
2. Each cell scored for presence/absence of each landform type
3. Weighted score per cell per depth range:
   C_geomorph[depth] = Σ(w_i × presence_i) / Σ(w_i)
   where w_i = weight for landform i, presence_i ∈ {0, 1}
4. Normalize to [-1, +1] consistency range

Cross-link with Module A:
- When HiRISE classification (Module A) identifies LDA/LVF/CCF/GLF in an image:
  → Feed result into geomorphic layer as supplementary evidence
  → Flag the cell as having ML-detected glacial landforms
```

### API Endpoints

```
GET /api/swim-ice/geomorphic/point?lat=X&lon=Y
  Response: {
    lat: float, lon: float,
    consistency_shallow: float,         # 0-1m
    consistency_intermediate: float,    # 1-5m
    consistency_deep: float,            # >5m
    landforms_detected: [str],          # list of landform types in this cell
    hirise_classification: {            # from Module A, if available
      nearest_product: str | null,
      class: str | null,
      confidence: float | null
    }
  }

GET /api/swim-ice/geomorphic/tile/{z}/{x}/{y}.png?depth=0-1m|1-5m|5m-plus
  Response: colored PNG tile

GET /api/swim-ice/geomorphic/landforms?lat=X&lon=Y&radius_km=R
  Response: {
    landforms: [
      { type: str, count: int, example_coords: [lat, lon] }
    ]
  }
```

### New Files

```
backend/analysis/swim_geomorphic/
├── __init__.py
├── models.py              # GeomorphicPointResponse, LandformInfo
├── pipeline.py            # Load 3 depth-range GeoTIFFs, sample, cross-link with Module A

backend/data/swim/
├── geomorphology_0_1m.tif
├── geomorphology_1_5m.tif
├── geomorphology_5m_plus.tif
└── landform_ancillary/        # individual per-landform GeoTIFFs (optional)
```

---

## 11. Module G: SWIM Consistency Fusion

### Overview
The integration layer that combines all individual method scores into composite ice consistency maps at three depth ranges. This is the "glue" module — it reads outputs from Modules B–F and computes the weighted SWIM C_I equation.

### Algorithm
SWIM Consistency Integration (Morgan & Putzig 2025, Eq. 6):

```
C_I[depth] = (Σ w_i × C_i) / (Σ w_i)

where:
  C_i = consistency score from method i ∈ {-1, 0, +1}
  w_i = weight for method i (from SWIM Table 1, listed below)
  Sum is over methods applicable to that depth range

Depth range → contributing methods:
  0–1 m:  neutron + thermal + geomorphic_shallow + radar_surface
  1–5 m:  geomorphic_shallow + radar_surface + radar_dielectric
  >5 m:   geomorphic_deep + radar_dielectric

Output: C_I ∈ [-1, +1]
  +1 = all evidence consistent with ice
   0 = mixed/ambiguous evidence
  -1 = all evidence inconsistent with ice
```

**SWIM Method Weights (from Morgan & Putzig 2025, Table 1):**

| Method | 0–1 m | 1–5 m | >5 m |
|---|---|---|---|
| Neutron spectroscopy | 1.0 | — | — |
| Thermal inertia | 1.0 | — | — |
| Radar surface power | 1.0 | 1.0 | — |
| Radar dielectric | — | 1.0 | 1.0 |
| Geomorphic (shallow) | 1.0 | 1.0 | — |
| Geomorphic (deep) | — | — | 1.0 |

**Note**: Weights shown are equal (1.0) per the SWIM framework. Custom weighting via the POST endpoint allows users to adjust. "—" means the method does not contribute to that depth range.

### Two Operating Modes

**Mode 1: Pre-computed SWIM products** (default)
- Load the official SWIM combined consistency GeoTIFFs from swim.psi.edu
- Fastest: simple grid sampling, no computation
- Matches published SWIM results exactly

**Mode 2: Live fusion from individual MarsLab modules** (advanced)
- Query each method module (B–F) at the requested location
- Apply SWIM weighting equation
- Allows toggling individual methods on/off
- Allows incorporating MarsLab-specific data (e.g., Module A landform detections)
- Results may differ slightly from published SWIM due to data version differences

### API Endpoints

```
GET /api/swim-ice/consistency/point?lat=X&lon=Y&mode=precomputed|live
  Response: {
    lat: float, lon: float,
    consistency_0_1m: float,
    consistency_1_5m: float,
    consistency_5m_plus: float,
    method_scores: {                   # individual method contributions
      neutron: float | null,
      thermal: float | null,
      radar_surface: float | null,
      radar_dielectric: float | null,
      geomorphic_shallow: float | null,
      geomorphic_deep: float | null
    },
    mode: "precomputed" | "live",
    depth_to_ice_estimate_m: float | null   # derived estimate
  }

GET /api/swim-ice/consistency/tile/{z}/{x}/{y}.png?depth=0-1m|1-5m|5m-plus
  Response: colored PNG tile for map overlay

GET /api/swim-ice/consistency/region?north=N&south=S&east=E&west=W
  Response: {
    bounds: { north, south, east, west },
    stats_0_1m: { mean, std, min, max, coverage_pct },
    stats_1_5m: { mean, std, min, max, coverage_pct },
    stats_5m_plus: { mean, std, min, max, coverage_pct },
    tile_urls: { "0-1m": str, "1-5m": str, "5m-plus": str }
  }

POST /api/swim-ice/consistency/custom
  Body: {
    lat: float, lon: float,
    enabled_methods: [str],            # subset of methods to include
    custom_weights: { method: float }  # optional custom weights
  }
  Response: same as point query but with custom fusion
```

### New Files

```
backend/analysis/swim_fusion/
├── __init__.py
├── models.py              # ConsistencyPointResponse, ConsistencyRegionResponse, CustomFusionRequest
├── pipeline.py            # Pre-computed loader + live fusion engine
├── weights.py             # SWIM method weights per depth range (from Table 1)

backend/data/swim/
├── consistency_0_1m.tif
├── consistency_1_5m.tif
├── consistency_5m_plus.tif
```

---

## 12. Frontend Components

### New Components

| Component | Purpose |
|---|---|
| `SwimIcePanel.tsx` | Main SWIM control panel: depth range selector, method toggles, legend, stats |
| `SwimMethodLayer.tsx` | Reusable map layer component for each SWIM method tile overlay |
| `IceConsistencyLegend.tsx` | Color legend for -1 to +1 consistency scale |
| `HiriseLandformPanel.tsx` | HiRISE classification control: model selector, results display, attention heatmap |
| `LandformClassCard.tsx` | Card showing classification result with class icon, probability bar, confidence |
| `DepthRangeSelector.tsx` | Toggle between 0–1 m, 1–5 m, >5 m depth views |

### Modified Components

| Component | Change |
|---|---|
| `MainPage.tsx` | Add SWIM layer group to layer controls, add HiRISE Landform analysis mode |
| `Inspector.tsx` | Add "Ice Evidence" tab showing SWIM scores at clicked point |
| `LayerPanel.tsx` | Add SWIM method layers as toggleable group |

### API Client Files

| File | Purpose |
|---|---|
| `frontend/src/api/swim_ice.ts` | All SWIM ice endpoints (neutron, thermal, radar, geomorphic, fusion) |
| `frontend/src/api/hirise_landforms.ts` | HiRISE classification endpoints |

### Layer Overlay Approach

All SWIM method layers use the same tile-based overlay pattern:
```typescript
// CesiumJS UrlTemplateImageryProvider
const swimLayer = new Cesium.UrlTemplateImageryProvider({
  url: '/api/swim-ice/{method}/tile/{z}/{x}/{y}.png',
  minimumLevel: 2,
  maximumLevel: 8,    // ~3 km native resolution limits useful zoom
  rectangle: Cesium.Rectangle.fromDegrees(-180, -60, 180, 60)  // post-CRS-conversion bounds
});
```

### Color Scheme

| Score Range | Color | Meaning |
|---|---|---|
| +0.7 to +1.0 | Deep blue (#1a237e) | Strong ice evidence |
| +0.3 to +0.7 | Medium blue (#42a5f5) | Moderate ice evidence |
| -0.3 to +0.3 | Gray (#9e9e9e) | Ambiguous / no data |
| -0.7 to -0.3 | Light red (#ef9a9a) | Moderate against ice |
| -1.0 to -0.7 | Deep red (#b71c1c) | Strong against ice |

---

## 13. File Structure

### New Files (complete list)

```
backend/
├── analysis/
│   ├── hirise_landforms/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── preprocessing.py
│   │   └── heatmap.py
│   ├── swim_common/
│   │   ├── __init__.py
│   │   ├── geotiff_loader.py    # Shared GeoTIFF loading, CRS conversion, no-data handling
│   │   ├── tile_renderer.py     # Shared PNG tile generation with color mapping
│   │   └── coord_utils.py       # 0-360°E ↔ -180–180°E conversion, bounds validation
│   ├── swim_neutron/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── pipeline.py
│   ├── swim_thermal/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── pipeline.py
│   ├── swim_sharad_surface/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── pipeline.py
│   ├── swim_sharad_dielectric/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── pipeline.py
│   ├── swim_geomorphic/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── pipeline.py
│   └── swim_fusion/
│       ├── __init__.py
│       ├── models.py
│       ├── pipeline.py
│       └── weights.py
├── api/
│   ├── hirise_landforms_router.py
│   └── swim_ice_router.py          # unified router for all SWIM methods
├── data/
│   ├── hirise_landforms/
│   │   ├── weights/
│   │   └── cache/
│   └── swim/
│       ├── neutron_consistency.tif
│       ├── thermal_consistency.tif
│       ├── radar_surface_consistency.tif
│       ├── radar_dielectric_1_5m.tif
│       ├── radar_dielectric_5m_plus.tif
│       ├── geomorphology_0_1m.tif
│       ├── geomorphology_1_5m.tif
│       ├── geomorphology_5m_plus.tif
│       ├── consistency_0_1m.tif
│       ├── consistency_1_5m.tif
│       ├── consistency_5m_plus.tif
│       └── metadata.json

frontend/src/
├── api/
│   ├── swim_ice.ts
│   └── hirise_landforms.ts
├── components/
│   ├── SwimIcePanel.tsx
│   ├── SwimMethodLayer.tsx
│   ├── IceConsistencyLegend.tsx
│   ├── HiriseLandformPanel.tsx
│   ├── LandformClassCard.tsx
│   └── DepthRangeSelector.tsx
```

### Modified Files

```
backend/app.py                              # Register 2 new routers
backend/api/thermal_inertia.py              # Add /ice-score endpoint
frontend/src/pages/MainPage.tsx             # Add SWIM layers + HiRISE analysis mode
frontend/src/components/Inspector.tsx       # Add Ice Evidence tab
frontend/src/components/LayerPanel.tsx      # Add SWIM method layer group
```

---

## 14. Data Dependencies

### SWIM GeoTIFF Products (download from swim.psi.edu)

| File | URL | Size (est.) | Resolution |
|---|---|---|---|
| Combined consistency 0–1 m | swim.psi.edu/SWIM4MIMProducts.php | ~50 MB | ~3 km/px |
| Combined consistency 1–5 m | " | ~50 MB | ~3 km/px |
| Combined consistency >5 m | " | ~50 MB | ~3 km/px |
| Geomorphology (3 depth ranges) | " | ~150 MB total | ~3 km/px |
| Neutron consistency | " | ~50 MB | ~3 km/px |
| Thermal consistency | " | ~50 MB | ~3 km/px |
| Radar surface power | " | ~50 MB | ~3 km/px |
| Radar dielectric (2 depths) | " | ~100 MB | ~3 km/px |
| Combined consistency SWIM 1.0 | " | ~50 MB | ~3 km/px |
| SWIM Reconnaissance Zones | " | ~5 MB | Shapefile |

**Total estimated**: ~650 MB of GeoTIFF data + ~5 MB Shapefile

**Projection**: Simple cylindrical, Mars sphere, 0–360°E, ±60° latitude
**No-data value**: −30
**CRS**: IAU Mars 2000 (EPSG-like, not standard EPSG)

### Data Ingestion Script

A one-time setup script to download and verify SWIM products:

```
scripts/download_swim_data.py
  - Downloads all GeoTIFFs from swim.psi.edu
  - Verifies checksums
  - Converts CRS if needed (0-360°E → -180 to 180°E for Cesium compatibility)
  - Generates tile pyramids for map overlay
  - Writes metadata.json with version info
```

### HiRISE Model Weights

| File | Source | Size |
|---|---|---|
| MarsLandformNet V2 checkpoint | Local (from `scripts/marslandform_v2/` training) | ~500 MB |
| Mars-Bench ViT checkpoint | HuggingFace `Mirali33/mars-bench-landmark-classification` | ~350 MB |
| DINOv2-ViT-B/14 backbone | HuggingFace `facebook/dinov2-base` | ~350 MB |

---

## 15. Implementation Phases

### Phase 1: Data Ingestion & Infrastructure (Week 1)
1. Create `scripts/download_swim_data.py` — download all SWIM GeoTIFFs
2. Create shared `analysis/swim_common/` with GeoTIFF loader, tile renderer, coordinate utils
3. Create `backend/data/swim/` directory structure
4. Verify GeoTIFF loading and coordinate alignment with existing MarsLab CRS

### Phase 2: SWIM Tile Layers (Week 2)
5. Create `swim_ice_router.py` with tile endpoints for all methods
6. Create analysis modules for neutron, thermal, radar surface, radar dielectric, geomorphic
7. Create `SwimIcePanel.tsx`, `DepthRangeSelector.tsx`, `IceConsistencyLegend.tsx`
8. Add SWIM layers to `LayerPanel.tsx` and `MainPage.tsx`

### Phase 3: SWIM Point Queries & Fusion (Week 3)
9. Add point query endpoints for each SWIM method
10. Create `swim_fusion/` module with pre-computed and live fusion modes
11. Add "Ice Evidence" tab to `Inspector.tsx`
12. Extend `thermal_inertia.py` with `/ice-score` endpoint

### Phase 4: HiRISE Landform Classification (Week 4)
13. Create `hirise_landforms/` analysis module wrapping V2 inference pipeline
14. Create `hirise_landforms_router.py` with `/classify` endpoint
15. Create `HiriseLandformPanel.tsx` and `LandformClassCard.tsx`
16. Implement attention heatmap overlay generation

### Phase 5: Integration & Cross-Linking (Week 5)
17. Link Module A (landform classification) results into Module F (geomorphic mapping)
18. Add custom fusion endpoint (POST `/consistency/custom`)
19. End-to-end testing: point queries, tile overlays, classification, fusion
20. Performance optimization: tile caching, concurrent request limits

---

## 16. Edge Cases & Failure Modes

### Global
- **SWIM data not downloaded**: All SWIM endpoints return 503 with `{"error": "SWIM data not available. Run scripts/download_swim_data.py"}`
- **Coordinate system mismatch**: SWIM uses 0–360°E; MarsLab/Cesium uses -180–180°E. The ingestion script MUST convert.
- **No-data pixels**: SWIM uses −30 as no-data. Never return −30 as a score — return `null` with `data_quality: "no_data"`.

### HiRISE Landform
- **Model OOM on GPU**: Limit max tiles per image. If image > 20×20 tile grid, subsample or reject.
- **No DEM coverage at location**: Run classifier without topographic features (visual-only mode). Log warning.
- **Stale cache**: Cache results keyed by (product_id, model_version). Invalidate on model update.

### SWIM Layers
- **Query at poles (>60° latitude)**: Return no-data — SWIM study region is ±60°.
- **Query at high elevation (>1 km)**: Return no-data — SWIM excludes these areas.
- **Overlapping radar tracks**: For track-level detail, return all tracks within search radius, sorted by distance.
- **Very large region queries**: Limit to 10°×10° bounding box. Return 422 if exceeded.

### Fusion
- **Missing method data**: If a method returns no-data at a location, exclude it from fusion (reduce denominator). If ALL methods return no-data, return null consistency.
- **Custom weights sum to zero**: Return 422 validation error.
- **Live vs pre-computed mismatch**: Log discrepancies >0.2 for investigation. Expected due to data version differences.

---

## 17. References

### Papers
1. Morgan, G.A., Putzig, N.E., et al. (2025). "The SWIM Project: Updated Maps of Martian Subsurface Water Ice." *Planetary Science Journal*, 6:29. DOI: 10.3847/PSJ/ad9b24
2. Feldman, W.C., et al. (2004). "Global distribution of near-surface hydrogen on Mars." *JGR Planets*, 109.
3. Pathare, A.V., et al. (2018). "Ice grain size and the rheology of the Martian polar deposits." *Icarus*, 315.
4. Bramson, A.M., et al. (2015). "Widespread excess ice in Arcadia Planitia, Mars." *GRL*, 42.
5. Purohit, M., et al. (2025). "Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks." *NeurIPS 2025*. arXiv:2510.24010
6. Wagstaff, K.L., et al. (2018). "Deep Mars: CNN Classification of Mars Imagery for the PDS Imaging Atlas." *AAAI 2018*.

### Data Sources
- SWIM Products: https://swim.psi.edu/SWIM4MIMProducts.php
- Mars-Bench Models: https://huggingface.co/collections/Mirali33/mars-bench-models
- MONS IDND: PDS4 `urn:nasa:pds:mesick_pdart19_odyssey_idnd`
- TES Thermal Inertia: `MGS-M-TES-5-TIMAP-V1.0` at PDS Geosciences Node
- SHARAD RDR: `MRO-M-SHARAD-4-RDR-V2.0` at PDS Geosciences Node

### Open-Source Tools Referenced
- SOPA (SHARAD processing): https://github.com/adamoferro/sopa
- MSIM (Mars thermal model): https://github.com/nschorgh/MSIM
- pdr (PDS reader): https://github.com/MillionConcepts/pdr
- Mars-Bench: https://github.com/kerner-lab/Mars-Bench
