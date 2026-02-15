# MarsLab — Patch Notes v2.0

**Date:** 2025-02-10

---

## 1. Responsive UI

### Mobile-First Layout System

MarsLab now adapts from desktop to mobile with a responsive layout engine.

- **Desktop (768px+):** 3-column layout — resizable left sidebar (LayerPanel), center map (Cesium), and right sidebar (Inspector/Analysis panels)
- **Mobile (<768px):** Full-screen map with bottom navigation bar and swipe-to-dismiss bottom sheets for panel content

### Resizable Panels (Desktop)

All sidebar panels support drag-to-resize via edge handles:

| Panel | Default Width | Min | Max |
|-------|--------------|-----|-----|
| LayerPanel (left) | 320px | 200px | 50% viewport |
| Inspector (right) | 384px | 280px | 60% viewport |
| AI Analysis Panel | 420px | 320px | 60% viewport |
| Agentic AI Panel | 560px | 360px | 65% viewport |
| Report Panel | 620px | 400px | 70% viewport |

- Resize handles appear on panel edges with `col-resize` cursor
- Panels collapse to a 48px icon bar with an expand button (LayerPanel)
- Collapse state persists to `localStorage`

### Mobile Bottom Sheet

- Touch-friendly modal overlays for Layers and Inspector
- Drag-down-to-dismiss gesture (threshold: 100px)
- Max height: 70vh with overflow scroll
- Bottom navigation bar with "Layers" and "Inspector" buttons
- Auto-opens Inspector when a product is selected on mobile

### Responsive Components

- **TopBar:** Search bar adapts to available width; mobile menu toggle
- **DataDownloadPage:** 2-panel layout collapses to stacked full-width panels on mobile
- **AppShell:** Orchestrates responsive rendering; conditionally removes sidebars on mobile

---

## 2. Agentic AI Mode

### Overview

A fully autonomous mission planning agent that takes a natural language objective and executes a multi-step analysis pipeline — searching, downloading, and analyzing Mars orbital data to produce a decision-oriented mission assessment report.

### Capabilities

- **Natural Language Planning:** Parses user objectives (e.g., *"Assess Jezero Crater for shallow ice accessibility"*) into executable step sequences
- **LLM Integration:** Llama 3.3 (70B) via local Ollama for plan generation and narrative synthesis
- **Graceful Fallback:** If Ollama is unavailable, switches to rule-based planning + template narrative — the agent always works
- **Multi-Instrument Search:** Queries ODE API (CRISM, HiRISE, SHARAD) and local GeoJSON indices (CTX, HiRISE DTM, SHARAD high-res)
- **Automated Downloads:** Fetches missing data products with progress tracking (completed/failed/skipped count + ETA)
- **Real SHARAD Radargram Analysis:** Binary RDR ingestion, auto surface picking, SNR-based subsurface reflector detection, depth estimation (assuming εr = 3.15 for water ice)
- **CRISM Ice Scoring:** Per-product ice/hydration pixel statistics, spatial hotspot clustering
- **5x5 Slope Grid Analysis:** Multi-point terrain safety assessment (FAVORABLE / MARGINAL / UNFAVORABLE)
- **Cross-Instrument Fusion:** Spatial consistency analysis between SHARAD (direct detection) and CRISM (spectral proxy), with haversine distance measurement
- **Site Recommendation:** Weighted scoring — slope safety (40%), ice proximity (35%), SHARAD proximity (25%) — outputs primary + secondary sites with trade-off reasoning

### Composite Scoring (0–95 scale)

| Component | Max Points | Method |
|-----------|-----------|--------|
| SHARAD Subsurface | 20 | Detection count + depth estimation |
| CRISM Ice Signatures | 20 | High-ice product count + spatial clustering |
| Cross-Instrument Consistency | 15 | SHARAD–CRISM spatial agreement (<50 km = full score) |
| Engineering Feasibility | 25 | Slope grid safety rating (final filter, not primary driver) |
| Data Coverage | 12 | Total product count as confidence proxy |

Score never reaches 100 — capped at 95 to acknowledge inherent uncertainty.

**Recommendation tiers:** STRONG_CANDIDATE (75+), PROMISING_WITH_CAVEATS (55–74), REQUIRES_FURTHER_INVESTIGATION (35–54), LOW_PRIORITY (<35)

### Streaming & Resume

- **Server-Sent Events (SSE):** All progress streams in real time — planning tokens, step status, download counts, narrative chunks
- **Resumable Sessions:** Multi-consumer event buffer with `asyncio.Condition`. Disconnect and reconnect at any checkpoint via `/api/agent/resume/{id}?from_index=N`
- **Background Execution:** Agent continues running even if the browser tab closes. Reconnect later to see results.

### Report Generation

- **Markdown Report:** Structured mission assessment with sections for subsurface potential, surface composition, cross-instrument consistency, engineering feasibility, and landing site decision
- **PDF Export:** Professionally styled via weasyprint (falls back to Markdown if unavailable)
- **Figures:** Embedded SHARAD radargram, CRISM ice map, and slope grid heatmap (base64 PNGs)

### Session Persistence

- Completed sessions are saved to `backend/data/agent_sessions.json`
- Survive backend restarts — load automatically on startup
- Session history panel: browse past analyses, click to reload, view scores and reports
- Max 50 sessions retained with oldest-first eviction

### Knowledge Injection

- **Science context:** Per-region facts (ice confidence, known minerals, key findings) injected into LLM prompts
- **Instrument context:** Interpretation notes, detection methods, clutter warnings per instrument
- **Knowledge files:** Reusable methodology documents in `knowledge/*.md`, auto-loaded and tag-matched (e.g., dielectric constant estimation for terraced craters)

### UI

- Resizable panel (560px default, up to 65% viewport)
- Live reasoning display: streaming Llama tokens during planning and synthesis phases
- Step progress bar with per-step icons, status badges, and ETA
- Download progress sub-bar with file count and estimated time remaining
- Assessment score visualization: overall bar (0–100), recommendation badge, sub-score grid
- Rendered markdown narrative with prose styling
- Report download buttons (MD / PDF)
- Session history overlay with past session cards
- Ollama status badge (LLAMA / RULES indicator)

---

## 3. AI Landing Site Report

### Overview

A multi-region comparison engine that filters, analyzes, and ranks candidate Mars landing sites across ~55 predefined regions, producing a structured comparison report with executive summary and per-region breakdowns.

### Configuration (Ground Rules)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Latitude bounds | -50° to +50° | Filter regions by latitude range |
| Longitude bounds | Disabled | Optional east-west constraint |
| Include/Exclude regions | — | Explicitly include or exclude named regions |
| Include/Exclude tags | Exclude: "polar" | Tag-based filtering (crater, ice, volcanic, etc.) |
| Min slope safety | — | Post-analysis hard filter: FAVORABLE or MARGINAL required |
| Max regions | 5 | Cap number of regions analyzed (1–10) |
| Analyses | slope, subsurface, mineral | Select which analyses to run |
| Auto-download | true | Automatically fetch missing data products |
| Custom notes | — | Free-text guidance passed to LLM for narrative |

### 5-Phase Pipeline

1. **Filter:** Apply ground rules to ~55 regions → candidate list
2. **Per-Region Analysis:** For each candidate, execute search → check local data → download → slope analysis → subsurface scan → mineral analysis → synthesize
3. **Post-Filter:** Apply `min_slope_safety` constraint; flag violations
4. **Compare:** Rank regions by composite score, identify category winners (best engineering, subsurface, ice, coverage)
5. **Generate Report:** Executive summary (LLM or template fallback) + full Markdown report with rankings table and per-region detail sections

### Report Structure

```
Landing Site Comparison Report
├── Ground Rules Applied
├── Executive Summary (LLM-generated narrative)
├── Rankings Table (score breakdown per region)
├── Category Winners
├── Per-Region Details
│   ├── Engineering Feasibility (slope grid, safety rating)
│   ├── Subsurface Radar (SHARAD tracks, detections, depth)
│   ├── CRISM Mineral Signatures (ice/hydration, hotspot)
│   └── Score Breakdown
└── Recommended Landing Site (coordinates, rationale, trade-offs)
```

### Output Formats

- **Markdown** — always available
- **PDF** — styled HTML via weasyprint with table formatting, color-coded headers

### Session History & Persistence

- Completed reports saved to `backend/data/report_sessions.json` (max 20)
- History panel: browse past reports, view region count, recommended site, score
- Click to reload — instant display from cached JSON (no re-analysis)
- Resumable SSE streaming for in-progress reports

### UI (ReportPanel — 3-View Design)

1. **Config Wizard:** Ground rule sliders, tag toggles, region picker, analysis checkboxes, matching region preview
2. **Progress Monitor:** Per-region progress cards (collapsible), step status, elapsed time, live reasoning text, stop button
3. **Report Viewer:** Executive summary, rankings table with highlights, category winners grid, recommended site card, download buttons (MD/PDF)

---

## 4. CRISM 1D CNN-Attention Mineral Classification

### Overview

Per-pixel mineral identification on CRISM TRR3 hyperspectral images using a multi-branch 1D CNN with attention-based spectral fusion. Classifies each pixel into one of 23 mineral classes with strict confidence filtering.

### Model Architecture: MultiBranchAttnCNN

```
Input: 350 IR bands (1.021–3.477 µm)
  ↓
7 Spectral Branches (one per wavelength group)
  Each: Conv1d(1→16, k=7) → Conv1d(16→32, k=5) → Conv1d(32→64, k=3) → AvgPool
  ↓
Attention Fusion
  FC(896→64) → FC(64→7) → Softmax(τ=0.5)
  → Weighted sum of 7 branch feature vectors
  ↓
Classification Head
  FC(128→64) → FC(64→23)
  → Softmax → Confidence threshold at 95%
```

- **7 spectral branches** spanning the full CRISM IR range
- **Attention mechanism** with temperature τ=0.5 for sharp branch weighting
- **23 mineral classes** including CO₂ ice, H₂O ice, phyllosilicates, sulfates, and more
- **Confidence threshold: 95%** — pixels below threshold marked as unclassified

### 23 Supported Mineral Classes

CO₂ Ice, H₂O Ice, Gypsum, Ferric Hydroxysulfate, Fe Smectite, Mg Smectite, Prehnite, Jarosite, Serpentine, Alunite, Akaganeite, Al Smectite (×2), Kaolinite, Bassanite, Epidote, Polyhydrated Sulfate, Illite, Analcime, Monohydrated Sulfate, Hydrated Silica, Ferricopiapite, Chlorite, Chlorite-Smectite

### Input Processing

| Input | Format | Source |
|-------|--------|--------|
| TRR3 cube | PDS3 binary, 438 bands × rows × cols | CRISM observation |
| DDR | PDS3 binary, incidence angle extraction | Geometry data |
| VS ADR | Volcano Scan reference spectra | Atmospheric standard |

### JCAT Atmospheric Correction

Per-pixel spectral correction using the JCAT (CRISM Analysis Toolkit) pipeline:

1. Extract 350 IR bands from TRR3 cube
2. Load VS ADR reference spectra (vstrans + vsart) for the correct time epoch
3. Per-pixel loop: apply `jcat_correction_pipeline()` with incidence angle
4. Progress callback every 500 pixels → real-time SSE updates

**ADR Time-Based Selection:** 3 calibration epochs (1980, 2009-04, 2010-01). Automatically selects the latest epoch ≤ observation acquisition time.

### Pipeline (7 Steps, SSE-Streamed)

1. **Resolve Files** — Locate TRR3 + DDR in `mineral_cnn_data/{obs_id}_07/`
2. **Load TRR3 Cube** — PDS3 binary → float32 array
3. **Load DDR** — Extract incidence angle → radians
4. **Select ADR** — Time-based epoch matching
5. **JCAT Correction** — Per-pixel atmospheric correction (progress-streamed)
6. **CNN Inference** — Batch processing (1024 pixels/batch), 7-branch attention fusion, softmax filtering
7. **Save Results** — Disk cache: `.npy` arrays + `metadata.json` + PNG maps

### Output Products

| File | Content |
|------|---------|
| `mineral_map.npy` | Per-pixel mineral ID (int32), -1 = unclassified |
| `confidence_map.npy` | Per-pixel softmax confidence (float32, 0–1) |
| `attention_map.npy` | Per-pixel 7-dim attention weights (float32) |
| `valid_mask.npy` | Boolean mask (≥250 valid bands per pixel) |
| `metadata.json` | Dimensions, class distribution, elapsed time, ADR used |
| `mineral_map.png` | Color-coded mineral map (fixed palette per class) |
| `confidence_map.png` | Grayscale confidence visualization |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mineral-cnn/classify/{obs_id}` | POST | Run classification (SSE stream with progress) |
| `/api/mineral-cnn/result/{obs_id}/stats` | GET | Classification statistics + mineral distribution |
| `/api/mineral-cnn/result/{obs_id}/mineral-map.png` | GET | Color mineral map image |
| `/api/mineral-cnn/result/{obs_id}/confidence-map.png` | GET | Confidence map image |
| `/api/mineral-cnn/result/{obs_id}/pixel?line=N&sample=M` | GET | Per-pixel: mineral name, confidence, attention weights |
| `/api/mineral-cnn/result/{obs_id}/legend` | GET | Mineral legend (names, colors, pixel counts) |
| `/api/mineral-cnn/status` | GET | Model status, device, available cached results |

### Design Notes

- **Thread-safe model loading:** Double-checked locking singleton — model loads once even under concurrent requests
- **Async executor pattern:** JCAT correction and inference run in thread pool to avoid blocking the FastAPI event loop
- **Cache-first:** If results exist on disk, returns instantly without re-running inference
- **Input validation:** Observation ID validated with `^[a-zA-Z0-9_]+$` regex to prevent path traversal
- **Memory management:** Original TRR3 cube freed after correction to release ~1–2 GB before inference

---

## 5. Slope 3D Analysis

### Overview

Interactive terrain analysis combining statistical slope assessment with a full 3D terrain viewer. Uses the HRSC/MOLA Blend DEM (200 m/pixel global coverage) for slope computation and elevation visualization.

### Slope Statistics Panel

- Displays mean slope, standard deviation, and maximum slope for a selected region
- Slope distribution histogram in three bins: 0–3° (safe), 3–5° (marginal), 5°+ (hazardous)
- Safety assessment with clear ratings:

| Rating | Condition | Interpretation |
|--------|-----------|----------------|
| FAVORABLE | All slopes < 5° | Safe for landing |
| MARGINAL | Mean < 5°, few steep areas | Acceptable with caution |
| UNFAVORABLE | ≥10% pixels exceed 5° | Not recommended |
| UNKNOWN | No valid terrain data | Cannot assess |

### 3D Terrain Viewer

- Three.js-based interactive 3D mesh rendering via React Three Fiber
- Orbit controls: rotate, zoom, pan with mouse/touch
- Elevation-based coloring: HSL gradient from brown/tan (low) to white (high)
- Red center marker sphere at the analysis point

**Controls:**
- Patch size: 2 km, 5 km, 10 km, 50 km
- Vertical exaggeration: 1–20x slider
- Wireframe toggle
- Bounding box toggle
- Debug mode

### Backend

- **Data source:** `Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif` (GeoTIFF)
- **Slope computation:** `slope = arctan(√(dz/dx² + dz/dy²))` with latitude-aware meter-per-degree conversion
- **Distance masking:** Vectorized haversine to exclude pixels beyond the analysis radius
- **DEM patch extraction:** Bilinear resampling to target grid (default 128×128), handles NaN fill values
- **Mars ellipsoid:** Equatorial radius 3,396,190 m, polar radius 3,376,200 m (IAU 2000)

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /terrain/slope_stats` | Slope statistics for a point + radius |
| `GET /terrain/dem_patch` | Elevation grid for 3D rendering |

---

## 6. Multi-Instrument Overlap Filter

### Overview

Spatial intersection filter that shows only products overlapping with at least one product from another instrument. Enables cross-instrument co-location studies across all six supported instruments.

### Supported Instruments

CRISM, HiRISE, SHARAD, SHARAD High-Res, CTX, HiRISE DTM

### Algorithm

1. **Spatial grid index** — 5° cells (72×36 grid for full Mars) for O(k) candidate lookup instead of O(n²) brute force
2. **Per-product test** — For each product in instrument X, check overlap with any product in instruments Y, Z, ...
3. **Geometry-aware intersection:**
   - **Polygon–Polygon:** Bounding box overlap test
   - **LineString–Polygon:** Liang-Barsky line clipping per segment (precise — avoids false positives from bbox-only tests)
   - **LineString–LineString:** Both lines must intersect each other's bounding box
4. **Antimeridian handling:** Correctly detects and indexes bounding boxes crossing 180°/−180°

### Filter Composition

Overlap filter composes with other active filters via AND logic:
- CRISM: overlap AND ice score filter (if active)
- Other instruments: overlap is the sole visibility determinant

### UI

- Toggle ON/OFF in LayerPanel under "Overlap Filter" section
- Summary statistics: "X/Y products passing" (total and per-instrument)
- Requires at least 2 instruments loaded to compute
- Warning message if no overlaps found

---

## 7. Field Notes

### Overview

Annotation system for attaching research memos and tags to any Mars observation product. Notes are organized by tag, displayed on the Cesium globe as pin markers, and persisted to JSON.

### Capabilities

- **Full CRUD:** Create, read, update, delete notes via modal dialog
- **Tag system:** Multiple tags per note, autocomplete with existing tags, create-new-tag inline
- **Free-text memo:** Multi-line annotation field
- **All instruments supported:** CRISM, HiRISE, SHARAD, SHARAD High-Res, CTX, HiRISE DTM
- **Unicode support:** Tag names support any character set (e.g., Korean, Japanese)

### Map Visualization

- **Pin markers** on the Cesium globe with instrument-specific colors:
  - CRISM: cyan, HiRISE: yellow, SHARAD: orange, CTX: pink, HiRISE DTM: amber
- **Coordinate resolution:** If a note lacks explicit coordinates, the system auto-fetches the product footprint centroid (polygon centroid or LineString midpoint)
- **Tag filtering:** Select a tag in the LayerPanel to display only notes with that tag on the map
- **Click interaction:** Click a map marker to jump to the associated product in the Inspector

### LayerPanel Integration

- Collapsible "Field Notes" section with count badge
- **Grouped view:** Notes organized by tag (with "Untagged" group)
- **Tag filter mode:** Click a tag pill to filter both the list and map markers
- Per-note display: instrument badge, product ID, memo preview
- "Show on Map" toggle checkbox

### Persistence

- Stored in `backend/data/field_notes.json`
- Endpoints: `GET/POST /api/fieldnotes`, `PUT/DELETE /api/fieldnotes/{id}`, `GET /api/fieldnotes/tags`, `GET /api/fieldnotes/product/{id}`

---

## 8. HiRISE DTM 3D Viewer

### Overview

High-resolution 3D terrain visualization using HiRISE Digital Terrain Models (~1–2 m/pixel). Click any DTM footprint on the globe to open an interactive 3D viewer with overlaid instrument footprints.

### Features

- **Three.js 3D mesh** with orbit controls (rotate, zoom, pan)
- **Elevation coloring:** Brown/tan at low elevation → white at high elevation (HSL gradient)
- **Vertical exaggeration:** 1–20x slider to emphasize terrain features
- **Patch size selection:** 1 km, 5 km, 10 km, full DTM extent
- **Hover elevation probe:** Move cursor over terrain to read elevation at any point
- **Instrument footprint overlays:** Wire-frame boundaries of overlapping CRISM, HiRISE, SHARAD, CTX products projected onto the 3D surface
- **Wireframe and bounding box** debug toggles
- **Slope statistics** in footer (mean/max slope for the visible patch)

### Data

- DTM products stored in `backend/hirise_dtm_data/` as PDS `.IMG` binary elevation grids and `.JP2` orthoimages
- Spatial index: `index.geojson` with footprint polygons and resolution metadata
- Product IDs: `DTEEC_*`, `DTEED_*`, etc. — matched via `startsWith("DTE")`

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /terrain/hirise_dtm_patch` | Extract DEM patch at center point with radius |
| `GET /terrain/hirise_dtm_elevation_grid` | Compact elevation grid for hover lookups |

---

## 9. SHARAD High-Res Radargram Viewer + Cluttergram

### Overview

Full-featured radar subsurface viewer for SHARAD high-resolution products. Includes interactive radargram display, automatic surface picking, cluttergram overlay for clutter identification, and depth conversion with adjustable dielectric parameters.

### Radargram Viewer

- **Zoomable canvas** with scroll-to-zoom and drag-to-pan
- **Display modes:** Log/linear scale, amplitude/power representation
- **Normalization:** Per-trace or global
- **Contrast control:** Adjustable percentile sliders (1–99%)
- **Cursor info bar:** Trace index, range bin, lat/lon coordinates, zoom level

### Surface Line

- **Auto-picked** from peak detection algorithm (coarse anchor + refinement band)
- **Manual adjustment mode:** Click and drag surface vertices to correct picks locally
- **Gaussian smoothing** applied to edits with triangular weighting kernel
- **Freeze option** to lock manual edits while panning

### Cluttergram Overlay

- **Toggle checkbox** to composite the US SHARAD team's surface clutter simulation on top of the radargram
- **Opacity slider** (0–100%) for blending
- Cluttergrams loaded from NetCDF `.nc` files, auto-aligned to RDR traces via lat/lon interpolation
- Helps distinguish real subsurface reflectors from surface echo artifacts

### Depth Conversion

- **Piecewise dielectric model:** Two layers with adjustable εr₁ (surface layer), εr₂ (deep layer), and boundary depth
- Click below the surface line to compute depth in meters at that point
- Uses two-way travel time: `depth = c × Δt / (2 × √εr)`

### MOLA Elevation Profile

- Draggable MOLA elevation profile aligned alongside the radargram
- Horizontal offset adjustment for correlation analysis between surface topography and subsurface features

### Data Format

- Binary PDS3 RDR: 5,822 bytes/row, 667 range bins per trace
- Cluttergram: NetCDF `.nc` files from US SHARAD team
- Stored in `backend/sharad_highres_data/` with `index.geojson` spatial index

---

## 10. Coordinate Grid Overlay

### Overview

Toggleable latitude/longitude grid overlay on the Cesium globe. Grid spacing automatically adapts to the camera zoom level — no manual configuration needed.

### Auto-Adaptive Spacing

| Zoom Level | Grid Spacing | Label Behavior |
|------------|-------------|----------------|
| Global view | 10°+ | No labels (too dense) |
| Continental | 5° | Labels on every line |
| Regional | 1° | Labels on every line |
| Local | 0.5° | Labels on every 2nd line |
| Close-up | 0.1° | Labels on every 2nd line |

### Visual Style

- **Grid lines:** White, 20% opacity, 1 pixel width — subtle enough to not obstruct data
- **Labels:** White, 45% opacity, 9px monospace font
- **Format:** 1 decimal place for fine spacing, integers for coarse

### Performance

- **Viewport clipping:** Fine grids only draw lines within the camera frustum (not full-planet)
- **Tier system:** Grid rebuilds only when the spacing tier changes or the fine grid needs to follow a pan
- **Batch entity management:** All grid entities prefixed `GRID_` for efficient bulk add/remove

### Toggle

- Checkbox in LayerPanel: "Coordinate Grid" with grid icon
- State managed in MainPage, passed to MapView

---

*MarsLab — Mars Orbital Data Analysis Platform*
