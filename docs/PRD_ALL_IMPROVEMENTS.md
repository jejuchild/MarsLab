# MarsLab Improvements — Product Requirements Documents

## PRD 1: Empty States & Suggested Actions

### Overview
Add informative empty states to panels that currently show blank/nothing when no data is selected.

### Goals
- Reduce confusion for new users encountering empty panels
- Provide contextual suggested actions to guide discovery

### Scope
- Inspector panel (no product selected)
- LayerPanel (no instruments loaded)
- Right panel area (no analysis active)

### Technical Design
- Add `EmptyState` reusable component with icon, title, description, and optional action button
- Integrate into Inspector, LayerPanel, and the right panel fallback in MainPage

### Files
- `frontend/src/components/EmptyState.tsx` (new)
- `frontend/src/pages/MainPage.tsx` (right panel fallback)

---

## PRD 2: Keyboard Shortcuts

### Overview
Extend keyboard shortcuts beyond Cmd+K command palette for power users.

### Goals
- Enable keyboard-driven navigation through products and analysis modes
- Display shortcut hints in the UI

### Scope
- `N/P` — next/previous product in visible list
- `1-7` — toggle instrument layers
- `S` — toggle slope analysis mode
- `G` — toggle coordinate grid
- `Escape` — close current panel/modal
- `?` — show keyboard shortcut help overlay

### Technical Design
- Add global keydown listener in MainPage
- Show `KeyboardShortcutsHelp` modal with shortcut reference
- Display shortcut hints as small badges on buttons

### Files
- `frontend/src/components/KeyboardShortcuts.tsx` (new — help overlay)
- `frontend/src/pages/MainPage.tsx` (keydown handler)

---

## PRD 3: Onboarding Tour

### Overview
Lightweight step-by-step tour for first-time users highlighting key workflows.

### Goals
- Walk users through: search → load instruments → inspect → analyze
- Show only once (localStorage flag)
- Skippable at any time

### Technical Design
- Array of tour steps with target element selector, title, description, placement
- Highlight target element with spotlight overlay
- Next/Skip/Done buttons
- Store `marslab-tour-completed` in localStorage

### Files
- `frontend/src/components/OnboardingTour.tsx` (new)
- `frontend/src/pages/MainPage.tsx` (mount tour)

---

## PRD 4: Bookmarkable/Shareable State via URL

### Overview
Reflect key app state in URL query parameters so views can be shared.

### Goals
- Share exact map positions and selected instruments with collaborators
- Deep-link to specific products and analysis modes

### Scope
- URL params: `lat`, `lon`, `zoom`, `instruments`, `product`, `mode`
- Read on mount, update on state change (debounced)

### Technical Design
- Extend existing `useSearchParams` in MainPage
- Sync map position, instrument visibility, selected product, analysis mode
- Debounce URL updates (500ms) to avoid history spam

### Files
- `frontend/src/hooks/useUrlState.ts` (new)
- `frontend/src/pages/MainPage.tsx` (integrate hook)

---

## PRD 5: Spectral Profile Comparison Tool

### Overview
Allow users to pin CRISM spectra from multiple locations/products and overlay them on a single plot for comparison.

### Goals
- Compare mineral signatures across different locations
- Reference against known mineral spectra

### Scope
- Pin up to 5 spectra with distinct colors
- Overlay plot with shared wavelength axis
- Label each spectrum with product ID + coordinates
- Clear individual or all pinned spectra

### Technical Design
- `SpectralComparison` component with Recharts line chart
- Store pinned spectra in MainPage state
- "Pin Spectrum" button in Inspector CRISM Spectrum tab
- Each pinned spectrum: `{ productId, lat, lon, wavelengths[], reflectance[], color }`

### Files
- `frontend/src/components/SpectralComparison.tsx` (new)
- `frontend/src/components/Inspector.tsx` (add Pin button)
- `frontend/src/pages/MainPage.tsx` (state + panel rendering)

---

## PRD 6: CRISM Band Ratio Calculator

### Overview
Interactive calculator for custom spectral band ratios rendered as map overlays.

### Goals
- Enable exploratory mineralogy with custom formulas
- Presets for common indices (BD1900, OLINDEX, SINDEX, BD2100)

### Scope
- Formula input: `(R[band1] - R[band2]) / R[band3]`
- Preset dropdown with common indices
- Compute on backend, return as image overlay
- Color-mapped result with adjustable min/max stretch

### Technical Design
- Backend endpoint: `POST /api/crism/band-ratio` with formula + product_id
- Frontend: `BandRatioCalculator` component in Inspector CRISM tab
- Result rendered as Cesium overlay

### Files
- `backend/api/crism_band_ratio.py` (new router)
- `frontend/src/components/BandRatioCalculator.tsx` (new)
- `frontend/src/components/Inspector.tsx` (add tab/button)
- `backend/app.py` (register router)

---

## PRD 7: Thermal Inertia Query

### Overview
Expose TES thermal inertia data as a point-queryable layer.

### Goals
- Allow users to query thermal inertia at any point on Mars
- Display value in slope analysis and Inspector metadata

### Scope
- Point query endpoint returning thermal inertia value
- Display in SlopeAnalysis panel alongside slope stats
- Show in terrain click info

### Technical Design
- Backend endpoint: `GET /terrain/thermal_inertia?lat=X&lon=Y`
- Read from existing climate model or TES GeoTIFF
- Display thermal inertia value with interpretation (fine dust < 100, rock > 1200)

### Files
- `backend/api/terrain_router.py` (add endpoint)
- `frontend/src/components/SlopeAnalysis.tsx` (display TI value)

---

## PRD 8: GIS Export

### Overview
Export analysis results (slope maps, mineral maps) as GeoTIFF files with proper CRS.

### Goals
- Enable researchers to bring MarsLab results into QGIS/ArcGIS
- Export slope grids, mineral classification maps, DEM patches

### Scope
- Export slope analysis as GeoTIFF
- Export mineral CNN results as GeoTIFF
- Export DEM patches as GeoTIFF
- Download button in analysis panels

### Technical Design
- Backend endpoint: `GET /terrain/export_geotiff?lat=X&lon=Y&radius=R`
- Use rasterio to write GeoTIFF with Mars CRS (IAU:49900)
- Frontend: download button in SlopeAnalysis panel

### Files
- `backend/api/terrain_router.py` (add export endpoint)
- `frontend/src/components/SlopeAnalysis.tsx` (add download button)

---

## PRD 9: Cross-Section Transect Tool

### Overview
Draw a line on the map and get combined elevation + instrument data along the path.

### Goals
- Geological cross-section analysis along arbitrary transects
- Combined MOLA elevation profile + slope variation

### Scope
- Extend existing LineProfile with slope overlay
- Add elevation + slope dual-axis chart
- Show distance markers along transect

### Technical Design
- Backend endpoint already exists for elevation profile
- Add slope computation along transect to backend
- Frontend: enhanced LineProfile with dual Y-axis (elevation + slope)

### Files
- `backend/api/terrain_router.py` (add transect_profile endpoint)
- `frontend/src/components/LineProfile.tsx` (enhance with slope)

---

## PRD 10: Statistical Region Analysis

### Overview
Draw a polygon on the map and get aggregate statistics for the enclosed area.

### Goals
- Area-based analysis complementing point-based approach
- Slope distribution, elevation range, area calculation

### Scope
- Draw polygon tool on map
- Backend computes: slope distribution, elevation stats, area in km²
- Display results in a panel

### Technical Design
- New analysis mode: "region_stats"
- Polygon drawing via Cesium click handler (collect vertices, close on double-click)
- Backend endpoint: `POST /terrain/region_stats` with polygon vertices
- Frontend: `RegionStatsPanel` component

### Files
- `backend/api/terrain_router.py` (add region_stats endpoint)
- `frontend/src/components/RegionStatsPanel.tsx` (new)
- `frontend/src/pages/MainPage.tsx` (add analysis mode + panel)

---

## PRD 11: Temporal Change Detection

### Overview
Compare co-registered images from different time periods to detect surface changes.

### Goals
- Identify RSL, frost changes, new impacts
- Side-by-side and difference view

### Scope
- Select two HiRISE/CTX products covering the same area
- Compute normalized difference image
- Display side-by-side with synchronized pan/zoom
- Highlight change regions

### Technical Design
- Backend endpoint: `POST /api/temporal/difference` with two product IDs
- Image alignment via georeferenced bounds
- Difference computation: normalized pixel-wise subtraction
- Frontend: `TemporalComparison` modal with split view

### Files
- `backend/api/temporal_router.py` (new)
- `frontend/src/components/TemporalComparison.tsx` (new)
- `backend/app.py` (register router)

---

## PRD 12: Unified Search Experience

### Overview
Smart search bar that auto-detects query type and routes to the appropriate search mode.

### Goals
- Single search input handles all query types
- Reduce cognitive load from multiple search modes

### Scope
- Auto-detect: coordinates → point search, product ID pattern → ID search, text → AI search
- Show search type indicator badge
- Unified results dropdown with mixed instrument results

### Technical Design
- Pattern matching in TopBar to detect query type:
  - Regex for lat/lon: `-?\d+\.?\d*\s*,\s*-?\d+\.?\d*`
  - Regex for product IDs: `^(frt|hrl|esp|psp|dteec|s_)` etc.
  - Fallback: text search
- Route to appropriate backend endpoint
- Show detected mode badge in search bar

### Files
- `frontend/src/components/TopBar.tsx` (enhance search logic)

---

## PRD 13: Undo/Redo System

### Overview
Lightweight action stack for reversible operations.

### Goals
- Undo/redo overlay changes, field note deletions, analysis actions
- Ctrl+Z / Ctrl+Shift+Z keyboard shortcuts

### Scope
- Track: overlay add/remove, field note create/delete, instrument toggle
- Max 20 actions in stack
- Show undo/redo buttons in TopBar

### Technical Design
- `useUndoRedo` hook with action stack
- Actions: `{ type, payload, undo() }` — each action knows how to reverse itself
- Keyboard shortcuts: Ctrl+Z (undo), Ctrl+Shift+Z (redo)

### Files
- `frontend/src/hooks/useUndoRedo.ts` (new)
- `frontend/src/pages/MainPage.tsx` (integrate hook)
- `frontend/src/components/TopBar.tsx` (undo/redo buttons)
