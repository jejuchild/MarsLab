# CTX — Context Camera

## Overview

- **Mission**: Mars Reconnaissance Orbiter (MRO), launched August 2005
- **PI Institution**: Malin Space Science Systems (MSSS), San Diego, CA
- **Type**: Grayscale pushbroom line scanner (panchromatic)
- **Primary goal**: Provide broad context imagery for HiRISE and CRISM targeted observations; global geomorphic mapping
- **Wavelength**: 350 nm bandpass centered at ~600 nm (550–850 nm effective range)
- **Status**: Operational as of 2026; >120,000 images acquired

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Spatial resolution | ~6 m/pixel (from 290 km altitude) |
| Swath width | ~30 km (cross-track) |
| Detector | 5,064 pixels cross-track (single CCD line array) |
| Pixel size | 7 µm |
| Focal length | 350 mm |
| F-number | f/3.25 |
| Spectral bandpass | 350 nm centered at 600 nm (panchromatic) |
| Bit depth | 8-bit (after onboard compression) |
| Data rate | ~10 Mbit/s |
| Image length | Variable; up to ~160 km along-track |
| Instrument mass | 9.1 kg |
| Power | 5.8 W |

## Coverage

- **Global coverage**: >99% of Mars surface imaged at least once by 2024
- **Stereo coverage**: ~40% of surface covered by stereo pairs suitable for DTM generation
- **Total images**: >120,000 as of 2024 (PDS archive, MSSS)
- **Repeat coverage**: Many regions imaged 5–20+ times enabling change detection
- **Coordinate system**: Mars 2000 IAU sphere; images georeferenced to MOLA datum
- **Archive**: NASA PDS Imaging Node; also accessible via JMARS, CTX mosaic (Murray Lab global mosaic at 5 m/pixel)

## Data Products

| Product | Description | Format |
|---------|-------------|--------|
| EDR | Experimental Data Record; raw DN values, minimal processing | IMG + LBL |
| RDR | Reduced Data Record; radiometrically calibrated, map-projected | IMG + LBL |
| Stereo pairs | Two overlapping images at different emission angles for DTM | Paired EDR/RDR |
| DTMs | Digital Terrain Models from stereo pairs; ~20 m/post typical | GeoTIFF |
| Global mosaic | Murray Lab 5 m/pixel global mosaic (non-PDS, community product) | GeoTIFF tiles |

Calibration converts raw DN to **I/F** (radiance factor); absolute radiometric accuracy ~10%.

## Scientific Applications

### Geomorphology
- Mapping of fluvial channels, alluvial fans, delta deposits, lava flows, dunes, polygonal terrain
- Crater morphology classification (fresh, degraded, filled) at regional scale
- Identification of tectonic features: graben, wrinkle ridges, fault scarps

### Crater Counting & Age Dating
- 6 m/pixel resolution resolves craters ≥50 m diameter reliably
- Crater size-frequency distributions (CSFD) used with Neukum/Hartmann production functions for surface age estimates
- CTX provides statistically robust crater populations over large areas (vs. HiRISE's small footprint)

### Change Detection
- Temporal baseline of 15+ years enables detection of:
  - New impact craters (dark blast zones visible at 6 m/pixel)
  - Slope streak formation and fading
  - Dune migration rates (cm/yr to m/yr)
  - Recurring Slope Lineae (RSL) seasonal appearance/disappearance
  - CO₂ frost deposition and sublimation patterns

### Landing Site Characterization
- Primary tool for landing ellipse hazard assessment (rock abundance, slopes, roughness)
- Used for all post-2006 landing site selections: Phoenix, Curiosity (Gale), InSight (Elysium), Perseverance (Jezero)
- Stereo-derived DTMs provide topographic context for entry, descent, landing (EDL) modeling

## Synergy with Other Instruments

### CTX + HiRISE
- CTX provides **context** for HiRISE targeting: ~30 km swath vs. HiRISE ~6 km RED swath
- HiRISE observations are planned using CTX images to identify high-priority sub-regions
- CTX stereo DTMs used to orthorectify HiRISE images and plan stereo acquisitions

### CTX + CRISM
- CRISM targeted observations (~10 km × 10 km) are planned using CTX geomorphic context
- CTX identifies mineralogically interesting landforms (fans, layered outcrops) for CRISM follow-up
- Combined CTX + CRISM analysis links morphology to mineralogy at regional scale

### CTX + MOLA/HRSC
- CTX stereo DTMs fill resolution gap between MOLA (~460 m/pixel) and HiRISE DTMs (1 m/post)
- HRSC (ESA Mars Express) provides complementary color context at 12–25 m/pixel

## Tags

CTX, MRO, context-camera, MSSS, panchromatic, pushbroom, 6m-resolution, geomorphology, crater-counting, change-detection, landing-site, stereo, DTM, EDR, RDR, Murray-Lab-mosaic, HiRISE-context, CRISM-context, RSL, dune-migration, impact-craters, Jezero, Gale, Elysium
