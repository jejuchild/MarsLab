# PRD: SHARAD RDR → Radar Consistency Pipeline

## Overview

Replace pre-computed SWIM2 GeoTIFFs (radar_surface_consistency.tif, radar_dielectric_*.tif)
with consistency grids computed from real SHARAD RDR data on disk. Produces traceable,
reproducible radar consistency scores from 102 high-resolution RDR tracks (116 GB).

## Goals

1. Compute **Radar Surface (RS) consistency** from SHARAD surface echo power
2. Compute **Radar Dielectric (RD) consistency** from subsurface reflector analysis
3. Output GeoTIFFs that drop-in replace existing SWIM2 files
4. Reuse existing RDR parser, surface picker, reflector detector, and epsilon calculator

## Non-Goals

- Downloading additional tracks from PDS (Phase 2)
- Modifying the SWIM fusion pipeline weights
- Changing the frontend Science Layers UI
- Full SHARAD calibration (Campbell 2013 absolute radiometric calibration)

## Technical Design

### Architecture

```
backend/analysis/sharad_rdr_pipeline/
├── __init__.py
├── PRD.md                     ← this file
├── rdr_loader.py              ← reuse sharad_highres_router parser, extract per-track data
├── surface_power.py           ← RS: surface echo extraction + geometric correction + scoring
├── dielectric.py              ← RD: reflector-based epsilon estimation + scoring
├── gridder.py                 ← aggregate trace-level measurements onto 0.5° grid → GeoTIFF
└── batch_process.py           ← CLI script: process all tracks → output GeoTIFFs
```

### Pipeline 1: Radar Surface Consistency (RS)

**Algorithm per track:**

1. Load power array (n_traces × 667) and geometry (lat, lon, alt) from cache
2. Load surface picks from cache (int32 array, -1 = no detection)
3. For each trace with valid surface pick:
   a. Extract surface power: `P_surf = max(power[trace, surf_bin-2 : surf_bin+3])`
      (5-bin window around pick to capture peak)
   b. Estimate noise floor: `P_noise = median(power[trace, surf_bin+100 : surf_bin+200])`
      (deep subsurface region below any expected reflectors)
   c. Compute SNR: `snr = P_surf / (P_noise + 1e-20)`
   d. Skip trace if snr < 5.0 (weak/unreliable surface)
   e. Convert to dB: `P_dB = 10 * log10(P_surf + 1e-20)`
   f. Geometric spreading correction: `P_corr_dB = P_dB + 20 * log10(alt_km / h_ref)`
      where `h_ref = 300 km` (nominal SHARAD altitude)
4. Return arrays: lat[], lon[], P_corr_dB[], snr[]

**Reference model (self-calibrating):**

Since we lack absolute radiometric calibration, use a relative approach:
- Compute the median corrected power per 5° latitude band across ALL tracks
- This median represents "typical" (dry basaltic) surface at each latitude
- Excess: `ΔP_dB = P_corr_dB - P_ref(lat)`
- This self-calibrating approach is robust to instrument gain variations

**Consistency scoring:**

```
if ΔP_dB ≥ +3.0:  consistency = +1.0  (anomalously reflective → ice-like interface)
if ΔP_dB ≤ -3.0:  consistency = -1.0  (anomalously absorptive → dry/porous)
else:              consistency = ΔP_dB / 3.0  (linear interpolation)
```

Clamp final score to [-1.0, +1.0].

### Pipeline 2: Radar Dielectric Consistency (RD)

**Algorithm per track:**

1. Load power array and geometry from cache
2. Load surface picks from cache
3. Detect subsurface reflectors using existing `_detect_reflectors_in_track()`
   (from ice_evidence/sharad_reflectors.py)
4. For each reflector segment:
   a. Compute two-way travel time delay below surface:
      `Δt_s = (reflector_bin - surface_bin) × 0.0375e-6`
   b. Compute apparent depth assuming vacuum propagation:
      `d_apparent = c × Δt_s / 2`
   c. Compute apparent depth assuming pure ice (εr=3.15):
      `d_ice = d_apparent / sqrt(3.15)`
   d. Assign depth bin:
      - d_ice < 5m:     skip (below SHARAD vertical resolution ~15m in free space)
      - 5m ≤ d_ice < 50m:  depth_bin = "1-5m" (SWIM shallow subsurface category)
      - d_ice ≥ 50m:       depth_bin = "5m-plus"
   e. Estimate εr from reflection coefficient ratio:
      `R = sqrt(P_reflector / P_surface)` (amplitude ratio)
      `ε_subsurface ≈ ((1 + R) / (1 - R))²` (Fresnel at normal incidence, simplified)
      Note: this is a rough upper bound; attenuation and transmission losses
      make the true εr lower.
   f. Apply existing thresholds:
      - εr < 4.5 → consistency = +1.0 (low dielectric → ice)
      - 4.5 ≤ εr ≤ 6.0 → consistency = 0.0 (ambiguous)
      - εr > 6.0 → consistency = -1.0 (high dielectric → rock)
5. Return arrays: lat[], lon[], consistency[], depth_bin[], snr[]

**Note on SHARAD vertical resolution:**
- Free-space range resolution: c/(2×BW) ≈ 15m (BW=10MHz)
- In ice (εr≈3.15): ~8.5m
- Depths < ~15m free-space are not resolvable → no "0-1m" bin for RD

### Pipeline 3: Gridding

**Grid specification:**
- Resolution: 0.5° × 0.5° (matches SWIM standard)
- Extent: -180° to +180° lon, -60° to +60° lat
- Shape: 720 cols × 240 rows
- CRS: Mars equicylindrical (IAU 2000, same as SWIM2 files)

**Aggregation per cell:**
1. Collect all trace-level measurements falling in cell
2. If ≥ 3 measurements: consistency = SNR-weighted median
3. If 1-2 measurements: consistency = SNR-weighted mean (mark as low-confidence)
4. If 0 measurements: consistency = 0.0 (no data / inconclusive)

**Output files:**
```
backend/data/swim/
├── radar_surface_rdr.tif          ← RS from real RDR (replaces radar_surface_consistency.tif)
├── radar_dielectric_rdr_1_5m.tif  ← RD shallow (replaces radar_dielectric_1_5m.tif)
├── radar_dielectric_rdr_5m_plus.tif ← RD deep (replaces radar_dielectric_5m_plus.tif)
```

Symlinks updated to point to new files.

## File Structure

### New files:
- `backend/analysis/sharad_rdr_pipeline/__init__.py`
- `backend/analysis/sharad_rdr_pipeline/rdr_loader.py`
- `backend/analysis/sharad_rdr_pipeline/surface_power.py`
- `backend/analysis/sharad_rdr_pipeline/dielectric.py`
- `backend/analysis/sharad_rdr_pipeline/gridder.py`
- `backend/scripts/build_sharad_consistency.py` (batch CLI)

### Modified files:
- `backend/data/swim/` symlinks (point to new GeoTIFFs)

### Unchanged:
- `swim_sharad_surface/pipeline.py` (still loads GeoTIFF via swim_common)
- `swim_sharad_dielectric/pipeline.py` (still loads GeoTIFF via swim_common)
- `swim_router.py` method-tile endpoint (serves whatever GeoTIFF symlinks point to)
- Frontend (no changes)

## Edge Cases

1. **Tracks with no valid surface picks**: Skip entirely (some tracks have instrument noise)
2. **Tracks crossing -180/+180 longitude**: Normalize all lon to -180..180 before gridding
3. **Overlapping tracks in same cell**: Weighted median prevents outlier domination
4. **Very high altitude passes**: Geometric correction handles via h² normalization
5. **NaN/Inf in power data**: Already handled by existing parser (nan_to_num)
6. **Empty reflector results**: Cell gets no dielectric measurement (stays 0.0)

## Success Criteria

1. `build_sharad_consistency.py` processes all 102 tracks without errors
2. Output GeoTIFFs have correct shape (240×720), CRS, and dtype (float32)
3. Non-zero coverage > 0% for both RS and RD (any real computed data)
4. Tiles render correctly on the map via existing method-tile endpoint
5. Science layers show colored regions where our tracks have data
