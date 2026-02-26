# THEMIS — Thermal Emission Imaging System

## Overview

THEMIS (Thermal Emission Imaging System) is a multispectral thermal infrared and visible imager aboard NASA's Mars Odyssey spacecraft. Developed by Arizona State University (ASU) under Philip Christensen, it launched April 7, 2001, and entered Mars orbit October 24, 2001. Mars Odyssey remains operational as of 2026, making THEMIS one of the longest-running Mars orbital instruments. THEMIS observes Mars in two modes: thermal infrared (IR) for surface temperature and composition, and visible (VIS) for morphology. It operates in a near-polar, sun-synchronous orbit at ~400 km altitude, crossing the equator at ~2:00 AM and ~2:00 PM local solar time — enabling consistent day/night thermal comparisons.

## Technical Specifications

| Parameter | IR Channel | VIS Channel |
|-----------|-----------|-------------|
| Spatial resolution | 100 m/pixel | 18 m/pixel |
| Spectral bands | 10 bands | 5 bands |
| Wavelength range | 6.78–14.88 μm | 425–860 nm |
| Swath width | 32 km | 18 km |
| Detector | Uncooled microbolometer array | CCD array |
| Array size | 320 × 240 pixels | 1024 × 192 pixels |
| Digitization | 12-bit | 8-bit |
| Noise equivalent delta T | ~0.3 K | — |

### IR Band Center Wavelengths (μm)
6.78, 7.93, 8.56, 9.35, 10.21, 11.04, 11.79, 12.57, 14.88 (Band 10 used for atmospheric correction)

## Data Products

- **Brightness temperature maps**: Calibrated radiance converted to apparent surface temperature (K); daytime and nighttime mosaics at 100 m/pixel
- **Thermal inertia maps**: Derived from day/night temperature difference; global coverage at ~100 m/pixel (THEMIS-derived TI) and ~3 km/pixel (TES-constrained)
- **Decorrelation stretch (DCS) composites**: False-color IR composites (bands 8-7-5 or 9-6-4) highlighting surface mineralogy; olivine, pyroxene, feldspar, and carbonate have distinct spectral signatures
- **Albedo maps**: Visible-channel reflectance mosaics
- **Daytime IR mosaics**: Global coverage at 100 m/pixel; primary product for surface temperature mapping
- **Nighttime IR mosaics**: Reveal thermophysical properties independent of solar heating geometry

## Scientific Applications

### Surface Composition Mapping
- IR spectral bands span the fundamental silicate absorption features (8–12 μm reststrahlen bands)
- DCS composites distinguish mafic minerals: olivine (band 7 bright), high-Ca pyroxene (band 6 bright), plagioclase (band 5 bright)
- Identified olivine-rich exposures in Nili Fossae, Ganges Chasma, and crater floors
- Carbonate detection at Nili Fossae (Ehlmann et al. 2008) — first orbital carbonate identification

### Thermal Inertia Analysis
- Thermal inertia (TI) quantifies resistance to temperature change: TI = k × ρ × c (J m⁻² K⁻¹ s⁻½, SI units)
- Derived from amplitude of diurnal temperature cycle; requires atmospheric correction
- Enables grain size and rock abundance mapping at 100 m/pixel — far superior to TES (~3 km/pixel)

### Night vs. Day Imaging
- Nighttime IR images suppress albedo effects; thermophysical properties dominate
- Warm nighttime surfaces = high thermal inertia (rock, ice, indurated material)
- Cold nighttime surfaces = low thermal inertia (fine dust, loose regolith)
- Nighttime imaging reveals buried structures, lava flow boundaries, and ejecta blankets invisible in daytime

### Volcanic and Aqueous Mineral Detection
- Identifies volcanic units by composition and thermophysical contrast with surrounding terrain
- Detects hydrated silica, phyllosilicates, and sulfates in conjunction with CRISM data
- Mapped hydrothermal alteration zones in Valles Marineris interior layered deposits

## Thermal Inertia Significance

| TI Value (J m⁻² K⁻¹ s⁻½) | Interpretation |
|---------------------------|----------------|
| < 50 | Very fine dust (< 40 μm); unconsolidated |
| 50–200 | Fine sand (40–500 μm); typical plains |
| 200–400 | Coarse sand/gravel (0.5–10 mm) |
| 400–800 | Rocky surface; indurated material |
| > 800 | Bedrock exposure or water ice |

- **Low TI regions**: Tharsis dust mantle, Arabia Terra — thick unconsolidated dust deposits
- **High TI regions**: Syrtis Major (basalt), polar layered deposits (ice), crater central peaks (bedrock)
- **Intermediate TI**: Most of the Martian plains; sand dune fields (TI ~200–350)

## Global Coverage

- Near-complete daytime IR coverage achieved by 2004; updated mosaics released through 2020
- Nighttime IR global mosaic at 100 m/pixel — unique capability not replicated by other instruments
- VIS coverage: ~30% of surface at 18 m/pixel (targeted observations, not systematic global)
- Data archive: NASA PDS Geosciences Node; ISIS3-compatible format (EDR, RDR, BTR, ABR, PBT)

## Synergy with MarsLab

- **Ice detection**: High thermal inertia anomalies in mid-latitudes flag candidate ice-rich terrain for MarsLab analysis; cross-reference with SHARAD and GRS hydrogen maps
- **Mineral mapping complement to CRISM**: THEMIS provides broader spatial context at 100 m/pixel; CRISM provides higher spectral resolution at 18 m/pixel — combined analysis resolves composition at outcrop scale
- **Landing site assessment**: Thermal inertia maps constrain surface bearing strength and dust hazard; low TI = soft, dusty terrain; high TI = rocky, firm surface
- **Nighttime anomalies**: Geothermal or subsurface ice signatures identifiable in nighttime IR; relevant for habitability and resource assessment

## Tags

themis, thermal-inertia, infrared, mars-odyssey, surface-composition, mineralogy, asu, thermal-emission, daytime-nighttime, grain-size, rock-abundance, decorrelation-stretch, olivine, pyroxene
