# Mars SWIM Project
## Subsurface Water Ice Mapping & MarsLab Implementation

---

# SECTION 1 — What is SWIM?

---

## Overview

- NASA-funded project managed by JPL, hosted at Planetary Science Institute (PSI)
- Goal: Map accessible subsurface water ice on Mars using multiple remote-sensing datasets
- Synthesizes five independent ice-detection techniques into a single composite score
- Designed to support future human mission planning and ISRU (In-Situ Resource Utilization)
- Portal: https://swim.psi.edu

Key publications:
- Morgan et al. 2021, Nature Astronomy 5:230–236 (SWIM 1.0)
- Putzig et al. 2023, Handbook of Space Resources, pp. 583–616 (SWIM 2.0)
- Morgan et al. 2025, PSJ 6:29 (SWIM4MIM — latest)

---

## The SWIM Equation (SWIM 1.0)

    Ci = (CN + CT + CG + CRS + CRD) / 5

Each term ranges from -1 to +1:
- +1 = wholly consistent with ice
-  0 = inconclusive or missing data
- -1 = wholly inconsistent with ice

Interpretation thresholds:
- Ci > 0.2 → at least one technique supports ice
- Ci > 0.5 → majority of techniques (≥3 of 5) support ice

---

## SWIM 2.0 — Depth-Weighted Equations

SWIM 2.0 splits the composite into three depth zones, applying shallowness weighting factors:

    Ci(d) = Σ [sMn(d) × CMn] / Σ sMn(d)

Three depth equations:
- Ci [0–1 m]  — uses CN, CT, CG_shallow, CRS
- Ci [1–5 m]  — uses CN, CT, CG, CRS, CRD
- Ci [>5 m]   — uses CN, CT, CG_deep, CRD

Geomorphology is split into "shallow" (polygons, mantle) and "deep" (LDA/LVF/CCF, glacial features).

---

## Five Ice-Detection Techniques

| Term | Technique                 | Instrument      | Depth Sensitivity |
|------|---------------------------|-----------------|-------------------|
| CN   | Neutron detection         | MONS (Odyssey)  | < 1 m             |
| CT   | Thermal analysis          | TES + THEMIS    | < 1 m             |
| CG   | Geomorphic mapping        | CTX (MRO)       | All depths         |
| CRS  | Radar surface analysis    | SHARAD (MRO)    | < 5 m             |
| CRD  | Radar dielectric analysis | SHARAD (MRO)    | > 15 m            |

Each technique measures a different physical property that serves as a proxy for ice.
No single technique can directly detect ice — SWIM's power is in their agreement.

---

# SECTION 2 — Technique Details & Formulas

---

## CN — Neutron Detection

Instrument: Mars Odyssey Neutron Spectrometer (MONS)
Physical basis: Hydrogen in subsurface water attenuates neutrons; measures water-equivalent hydrogen (WEH = Wdn)

Scoring:
    CN = +1           where Wdn ≥ 25%
    CN =  0 to +1     where 10% ≤ Wdn < 25%  (linear interpolation)
    CN = -1 to  0     where  5% ≤ Wdn < 10%  (linear interpolation)
    CN = -1           where Wdn < 5%

Depth: Sensitive to upper ~1 m only.
Caveat: A dry overburden of ~50 cm hides even pure ice deposits beneath.

Source: Pathare et al. 2018

---

## CT — Thermal Analysis

Instruments: MGS TES (3 km footprint) + MRO THEMIS (higher resolution)
Physical basis: Apparent Thermal Inertia (ATI) varies with subsurface layering. Ice and rock have similar high TI, so layering pattern is the diagnostic.

Scoring:
    CT = +1    where low-ATI material over high-ATI material
               (dust/sand over rock/ice — consistent with buried ice)
    CT = -1    where high-ATI over low-ATI
               (duricrust/rock over sand — inconsistent with ice)
    CT =  0    where no model match, OR top layer < 1 diurnal skin depth

Method: Forward modeling of seasonal ATI variation using four material types (dust, sand, duricrust, rock/ice). TES global map + THEMIS at 44 targeted ROIs.

Final CT = average of TES and THEMIS results, limiting CT to range [0, +0.5].

Source: Sizemore et al. 2020 (LPSC 2529)

---

## CG — Geomorphic Mapping

Instrument: CTX 5 m/pixel imagery
Physical basis: Presence of periglacial/glacial landforms indicates ice.

10 landform types surveyed:
1. Mantle                    6. 100-m scale polygons
2. Sublimation-type pits     7. Expanded craters
3. Textured terrain          8. Ring-mold craters
4. Scalloped terrain         9. Pedestal craters
5. LDA/LVF/CCF (glacial)   10. Terraced craters

Scoring:
    CG = max(g_survey, g_glacial, g_pedestal, g_scallop)

    where:
    g_survey   = (count of landforms present) / 10
    g_glacial  = 1.0   where LDA/LVF/CCF individually mapped
    g_pedestal = 0.75  where pedestal craters individually mapped
    g_scallop  = 0.75  where scalloped terrain individually mapped

Latitude taper: Gaussian taper applied from 30°N to 27°N (decline from 0.1 to 0).
Below 27°N with no features: CG = -1.

Source: Levy et al. 2014 (glacial mapping), Kadish et al. 2009 (pedestals)

---

## CRS — Radar Surface Analysis

Instrument: SHARAD (SHAllow RADar), MRO
Physical basis: Surface echo power ∝ Fresnel reflectivity ∝ near-surface density.
Ice is low-density → low power → consistent with ice.
Sensing depth: upper ~5 m (defined by SHARAD wavelength, 15 m in free space).

### Processing Pipeline (4 steps)

Step 1 — Ionosphere filtering
    Exclude all daytime tracks to limit ionospheric distortion.

Step 2 — Roughness normalization
    Normalize surface power by the SHARAD roughness parameter (Campbell et al. 2013).
    Roughness parameter derived from echo pulse broadening:
    - Smooth surface → energy concentrated in peak bin
    - Rough surface → energy spread across multiple bins
    Normalization isolates Fresnel reflectivity from roughness effects.

Step 3 — Topographic slope correction
    Correct for power loss due to regional slope using median MOLA slope
    over a Fresnel zone (~3 km baseline).

Step 4 — Spatial binning
    Take median of corrected returns in 1/12° lon × 1/12° lat bins
    to average out MRO roll, solar-panel configuration effects.

### Scoring Rubric (from global power distribution)

    z = (corrected_power - μ_global) / σ_global

    z < -1σ            →  CRS = +1.0   (very low power, ice-consistent)
    -1σ  ≤ z < -0.5σ   →  CRS = +0.5
    -0.5σ ≤ z < +0.5σ  →  CRS =  0.0   (inconclusive)
    +0.5σ ≤ z < +1σ    →  CRS = -0.5
    z ≥ +1σ            →  CRS = -1.0   (very high power, ice-inconsistent)

Output: 4 discrete non-zero values {-1, -0.5, +0.5, +1}, each ~25% of data.

Source: Campbell et al. 2013, JGR 118:436–450 (roughness parameter)
        Morgan et al. 2021 Supplementary Table 1 (scoring)

---

## CRD — Radar Dielectric Analysis

Instrument: SHARAD, MRO
Physical basis: Subsurface radar reflections reveal dielectric interfaces.
The dielectric constant constrains composition:
    Pure water ice:   ε' ≈ 3.15
    Porous regolith:  ε' ≈ 4–6
    Dense basalt:     ε' ≈ 7–9

### Dielectric Constant Estimation

    ε' = (c · Δt / 2h)²

    where:
    c  = speed of light in vacuum (3 × 10⁸ m/s)
    Δt = two-way time delay between surface and subsurface echoes (seconds)
    h  = physical depth to subsurface reflector (meters)

### Depth Estimation Methods (all manual/semi-manual)

Method A — Crater terraces (Bramson et al. 2015)
    Use HiRISE/CTX stereo DTMs to measure terrace depth in simple craters.
    Terrace = strength contrast at dielectric interface.
    Applied in Arcadia Planitia → ε' = 2.5 ± 0.3

Method B — MOLA interpolation at LDA margins (Petersen et al. 2018)
    Measure bedrock elevation at glacier (LDA) edges.
    Interpolate bedrock surface beneath the LDA.
    Depth = (LDA surface elevation) - (interpolated bedrock elevation).
    Applied in Deuteronilus/Protonilus → ε' ≈ 3.0–3.5

Method C — Mantle edge interpolation (Morgan et al. 2021, SWIM)
    Find both edges of a surface mantle deposit.
    Interpolate a sloped line between basement elevations.
    Applied broadly across northern mid-latitudes.

Method D — Point estimates
    Layer exposures in terraced craters and fossae.
    Direct depth measurement where stratigraphy is visible.

### Scoring Formula

    CRD = ½ (5 - ε')       clamped to [-1, +1]
        = -0.5 · ε' + 2.5

    ε' = 3  →  CRD = +1.0   (pure ice)
    ε' = 5  →  CRD =  0.0   (ambiguous: could be porous rock or icy regolith)
    ε' = 7  →  CRD = -1.0   (dense rock, no ice)
    No reflector detected → CRD = 0 (inconclusive)

Source: Bramson et al. 2015, GRL 42:6566–6574
        Petersen et al. 2018, GRL 45:11595–11604
        Morgan et al. 2021 Supplementary Table 1

---

## Summary — All Scoring Formulas

| Term | Input                | Formula                                      | Range        |
|------|----------------------|----------------------------------------------|--------------|
| CN   | Wdn (WEH %)         | Linear mapping at 5/10/25% thresholds        | [-1, +1]     |
| CT   | ATI layering         | Binary: low-over-high = +1, high-over-low = -1 | {-1, 0, +1} |
| CG   | Landform count       | max(survey/10, glacial, pedestal, scallop)    | [-1, +1]     |
| CRS  | Corrected power (dB) | 5-level σ rubric from global distribution     | {-1, -0.5, 0, +0.5, +1} |
| CRD  | ε' (dielectric)      | -0.5·ε' + 2.5, clamped                       | [-1, +1]     |

---

# SECTION 3 — Data Availability

---

## SWIM Products

Portal: https://swim.psi.edu
Products page: https://swim.psi.edu/SWIM2Products.php

Available formats:
- GeoTIFF (loads directly into QGIS, ArcMap)
- Browse PNG (with labels and color bars)
- Ancillary data

Projection: Simple cylindrical
Pixel scale: 3000 m (0.05° at equator)
Longitude: 0–360°E
Latitude: 60°S–60°N
No-data value: -30

### Product List

Composite products:
- Combined Ice Consistency Ci [0–1 m]
- Combined Ice Consistency Ci [1–5 m]
- Combined Ice Consistency Ci [>5 m]
- Combined Ice Consistency Ci (SWIM 1.0, unweighted)

Individual technique maps:
- Geomorphology (CG)
- Neutron Dataset (CN)
- Estimated Dielectric Properties (CRD)
- Radar Surface Power Return (CRS)
- Thermal Analysis (CT)

---

## Source Data

SHARAD RDR (Reduced Data Records):
- PDS Geosciences Node: https://pds-geosciences.wustl.edu
- Format: Binary tables (32-bit float, 3600 samples × N traces)
- ~32,000 orbital tracks globally, ~102 tracks over Arcadia Planitia

MOLA (Mars Orbiter Laser Altimeter):
- Topographic data used for slope correction (CRS) and depth estimation (CRD)
- 463 m/pixel global grid (MEGDR)

HiRISE DTMs:
- High-resolution stereo terrain models for crater terrace measurements
- 1–2 m/pixel, limited spatial coverage

SHARAD Super-Resolution Products:
- SAR-focused SHARAD data with ~300–500 m along-track resolution
- Reduces surface clutter, improves subsurface reflector detection
- Foss et al. 2024, Icarus 419:115793

---

## Key References

| Paper | DOI | Role in SWIM |
|-------|-----|--------------|
| Morgan et al. 2021, Nat. Astron. | 10.1038/s41550-020-01290-z | SWIM 1.0 equation, all scoring formulas (Suppl. Table 1) |
| Putzig et al. 2023, Handbook    | 10.1007/978-3-030-97913-3_16 | SWIM 2.0 depth-weighted equations |
| Morgan et al. 2025, PSJ         | 10.3847/PSJ/ad9b24 | SWIM4MIM, latest geomorphic update |
| Campbell et al. 2013, JGR       | 10.1002/jgre.20050 | SHARAD roughness parameter for RS |
| Bramson et al. 2015, GRL        | 10.1002/2015GL064844 | Dielectric method, Arcadia ε'=2.5±0.3 |
| Petersen et al. 2018, GRL       | 10.1029/2018GL079759 | LDA dielectric survey, ε'≈3.0–3.5 |

---

# SECTION 4 — MarsLab Implementation

---

## What We Built

A Python pipeline that computes CRS and CRD from 102 local SHARAD RDR tracks
over Arcadia Planitia (35–55°N, 160–230°E), then grids them to 0.05° GeoTIFFs
matching the SWIM2 product format.

Pipeline modules:
    backend/analysis/sharad_rdr_pipeline/
    ├── rdr_loader.py       # Binary SHARAD RDR parser + surface pick cache
    ├── surface_power.py    # CRS: integrated echo energy + σ-scoring
    ├── dielectric.py       # CRD: automatic reflector detection + Fresnel ε'
    └── gridder.py          # 0.05° GeoTIFF output, matching SWIM2 grid

Entry point:
    backend/scripts/build_sharad_consistency.py

Output files:
    backend/data/swim/radar_surface_rdr.tif          (CRS, 66 MB)
    backend/data/swim/radar_dielectric_rdr_1_5m.tif   (CRD 1–5m, 66 MB)
    backend/data/swim/radar_dielectric_rdr_5m_plus.tif (CRD >5m, 66 MB)

---

## CRS Implementation — Differences from SWIM

### What matches SWIM exactly:
- 5-level σ rubric: same thresholds at ±0.5σ, ±1σ
- Scoring formula: z < -1σ → +1, ..., z ≥ +1σ → -1
- Grid resolution: 0.05° (SWIM uses 1/12° ≈ 0.083°; ours is finer)
- Geometry correction: 20·log10(alt/300 km) spreading-loss term

### What differs:

| Step | SWIM | MarsLab | Impact |
|------|------|---------|--------|
| Track filtering | Nighttime only | All tracks | Minor |
| Roughness normalization | Campbell roughness parameter | Integrated echo energy (±10 bin window) | Moderate |
| Slope correction | MOLA slope over 3 km Fresnel zone | Not implemented | Moderate |
| σ reference | Global (32,000+ tracks) | Local (102 Arcadia tracks) | Major |

Integrated echo energy rationale:
    Peak power is biased by roughness — smooth surfaces concentrate energy
    in one range bin, rough surfaces spread it across many bins.
    Summing power in a ±10-bin window (~118 m) around the surface pick
    captures total reflected energy regardless of roughness.
    This approximates Campbell's roughness normalization without requiring
    the separate roughness parameter dataset.

### CRS Code (scoring function)

    z = (corrected_power_dB - μ) / σ

    CRS = +1.0   where z < -1.0
    CRS = +0.5   where -1.0 ≤ z < -0.5
    CRS =  0.0   where -0.5 ≤ z < +0.5
    CRS = -0.5   where +0.5 ≤ z < +1.0
    CRS = -1.0   where z ≥ +1.0

---

## CRD Implementation — Differences from SWIM

### What matches SWIM exactly:
- Scoring formula: CRD = -0.5 · ε' + 2.5, clamped to [-1, +1]
- Depth zones: 1–5 m and >5 m bins
- Output format: 0.05° GeoTIFF

### What fundamentally differs:

| Aspect | SWIM | MarsLab |
|--------|------|---------|
| ε' method | ε' = (c·Δt/2h)² with known h | ε' from Fresnel power ratio |
| Depth source | Manual MOLA topographic interpolation | Automatic: assume ε'=3.15 (ice) |
| Physical meaning | Bulk dielectric of layer above reflector | Interface dielectric at reflecting boundary |
| Reflector ID | Manual, clutter-checked | Automatic peak-picking (SNR ≥ 3) |

MarsLab CRD formula:
    R = √(P_reflector / P_surface)        # amplitude reflection coefficient
    R capped at 0.95
    ε' = ((1 + R) / (1 - R))²             # Fresnel equation
    CRD = -0.5 · ε' + 2.5                 # same SWIM scoring

These measure different physical quantities:
    SWIM: "What is the dielectric constant of the MATERIAL ABOVE the reflector?"
          (bulk property of traversed layer — answers "is this layer icy?")
    MarsLab: "What is the dielectric contrast AT the reflecting interface?"
          (interface property — answers "is there a strong dielectric boundary?")

---

## Comparison Results — CRS

102 SHARAD tracks processed, 8.87 million traces, 0 errors.

Against SWIM2 official product (SWIM2_RS.tif):

    Pearson correlation:   r = 0.519
    Sign agreement:        72.6%
    Exact value match:     39.2%
    Mean absolute error:   0.535
    Our mean:             -0.43  (vs SWIM -0.08)

The systematic negative bias (-0.43 vs -0.08) is caused by the local σ reference:
our 102 Arcadia tracks have a narrower power distribution than the global
32,000+ track distribution SWIM uses. This shifts our z-scores, producing
more negative (ice-consistent) scores than SWIM.

### Iteration History

| Version | Change | Pearson r | Sign Match |
|---------|--------|-----------|------------|
| v0 | Continuous clip(excess/3) | -0.326 | 40.1% |
| v1 | σ-based 5-level discrete | +0.435 | 66.8% |
| v2 | + roughness subtraction | +0.365 | 62.2% |
| v3 (final) | Integrated echo energy | +0.519 | 72.6% |

---

## Comparison Results — CRD

Against SWIM2 official product (SWIM2_RD.tif):

    Pearson correlation:   r ≈ 0.04
    Sign agreement:        53.0%

Poor correlation is expected — the two methods measure different physics.
SWIM uses topographic depth (h from MOLA), we use Fresnel power ratios.
These are fundamentally different approaches that cannot converge without
access to the same manual topographic depth estimates.

---

## Achievable vs. Not Achievable

### Achievable with more data/effort:
- Better CRS by using nighttime-only filtering
- Better CRS by adding MOLA slope correction
- Better CRS by using global σ (requires processing all 32,000+ tracks)

### Not achievable automatically:
- Matching CRD — requires manual reflector-by-reflector topographic analysis
  with human interpretation of geologic context
- Campbell roughness parameter — requires the separate derived dataset
  (not included in standard SHARAD RDR products)

### Fundamental insight:
CRS is algorithmically reproducible (σ-rubric on corrected power).
CRD is not — it requires human geologic interpretation for depth estimation.

---

## Pipeline Architecture

    SHARAD RDR binary files (102 tracks)
            │
            ▼
    rdr_loader.py
    - Parse 32-bit float power arrays (3600 bins × N traces)
    - Extract lat, lon, alt, surface picks from label files
    - Cache surface bin indices for fast re-processing
            │
            ├────────────────┐
            ▼                ▼
    surface_power.py    dielectric.py
    - ±10-bin integrated   - Automatic peak detection
      echo energy            (SNR ≥ 3, continuity filter)
    - Geometry correction  - Fresnel ε' estimation
    - Global μ/σ stats     - SWIM scoring formula
    - 5-level σ scoring
            │                │
            └────────┬───────┘
                     ▼
              gridder.py
    - 0.05° resolution grid (2400 × 7200 global)
    - Median aggregation per cell
    - GeoTIFF output (simple cylindrical, 0–360°E)
            │
            ▼
    radar_surface_rdr.tif
    radar_dielectric_rdr_1_5m.tif
    radar_dielectric_rdr_5m_plus.tif
