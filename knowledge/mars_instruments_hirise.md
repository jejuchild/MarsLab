# HiRISE — High Resolution Imaging Science Experiment

## Overview

- **Mission**: Mars Reconnaissance Orbiter (MRO), launched August 2005
- **PI Institution**: University of Arizona, Lunar and Planetary Laboratory (LPL)
- **Type**: Pushbroom line scanner; largest telescope ever flown to another planet
- **Primary goal**: Sub-meter imaging of Mars surface for geologic, geomorphic, and hazard assessment
- **Telescope**: 0.5 m aperture, 12 m focal length (f/24 Cassegrain)
- **Status**: Operational as of 2026; >80,000 observations acquired

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Spatial resolution | 25–30 cm/pixel (from 300 km altitude) |
| CCD detectors | 14 CCDs (10 RED, 2 BG, 2 NIR) |
| Pixels per CCD | 2,048 cross-track × variable along-track |
| RED swath width | ~6 km (10 CCDs × 2,048 pixels × 30 cm) |
| Color swath width | ~1.2 km (2 CCDs each for BG and NIR) |
| Along-track length | Up to ~20 km (limited by data volume) |
| Bit depth | 14-bit (onboard compression to 8-bit optional) |
| Data volume per image | Up to ~28 Gbit (uncompressed) |
| Instrument mass | 65 kg |
| Power | 90 W (peak) |

### Color Channels

| Channel | Wavelength Range | CCDs | Coverage |
|---------|-----------------|------|----------|
| BG (Blue-Green) | 400–600 nm | 2 | ~1.2 km central swath |
| RED | 550–850 nm | 10 | ~6 km full swath |
| NIR (Near-Infrared) | 800–1,000 nm | 2 | ~1.2 km central swath |

## Image Products

### Standard Products
- **RED**: Full-swath grayscale at 25–30 cm/pixel; primary science product
- **COLOR**: BG + RED + NIR composite over central 1.2 km; false-color IRB (NIR-RED-BG) standard display
- **IRB composite**: NIR assigned to red channel, RED to green, BG to blue; enhances spectral contrast

### Stereo & Elevation Products
- **Stereo pairs**: Two overlapping HiRISE images at convergence angles of 15–30°; requires separate targeted acquisitions
- **DTMs**: Digital Terrain Models at ~1 m/post from stereo pairs; produced by USGS Astrogeology and community (ASP pipeline)
- **Orthorectified images**: HiRISE images projected onto DTM surface; removes topographic distortion

### Product ID Format
- Pattern: `ESP_XXXXXX_XXXX` (Extended Science Phase)
  - `ESP`: Mission phase (Extended Science Phase, post-2008)
  - `XXXXXX`: Orbit number (6 digits)
  - `XXXX`: Target latitude code (center latitude × 10 + 9000, zero-padded)
- Example: `ESP_011350_1755` = orbit 11350, centered near 17.5°N
- Earlier phase: `PSP_XXXXXX_XXXX` (Primary Science Phase, 2006–2008)
- Full product: `ESP_011350_1755_RED` (RED channel), `_COLOR` (color), `_BG0`, `_RED4`, `_NIR0` (individual CCDs)

## Scientific Highlights

### Recurring Slope Lineae (RSL)
- Dark, narrow (0.5–5 m wide) features that grow downslope during warm seasons and fade in winter
- First identified in HiRISE images (McEwen et al. 2011, Science)
- Candidate mechanisms: briny water seeps, dry granular flows, CO₂ sublimation
- Detected at >50 sites; concentrated at steep equatorial and mid-latitude slopes

### Gullies
- Alcove-channel-apron morphology; formed in Amazonian (geologically recent)
- Seasonal CO₂ frost-driven activity observed in HiRISE time series (Dundas et al. 2012)
- Distinct from RSL; form in winter/spring, not summer

### Polar Processes
- Seasonal CO₂ ice sublimation: "spiders" (araneiform terrain), gas jets, dark fans imaged in real time
- Annual monitoring of polar cap edge retreat rates
- Tracking of new impact craters exposing subsurface ice

### Rover Monitoring
- Curiosity (Gale Crater): imaged from orbit at 25 cm/pixel; tracks, drill holes, wheel damage visible
- Perseverance (Jezero Crater): imaged during EDL (parachute deployment captured); ongoing traverse monitoring
- InSight lander: solar panel dust accumulation tracked over mission lifetime

### Boulder Tracking & Mass Wasting
- Individual boulders ≥1 m detectable; tracking of boulder trails from rockfalls
- Landslide deposits, debris flows, and slope failure scars mapped at cm-scale

## Applications for MarsLab

- **Landing site hazard assessment**: Rock abundance, slope angles, surface roughness at 25–30 cm/pixel; direct input to EDL models
- **Slope analysis**: DTMs at 1 m/post enable slope maps at scales relevant to rover trafficability
- **Rock abundance estimation**: Direct counting of rocks ≥1 m; extrapolation to smaller sizes using power-law size-frequency distributions
- **Change detection**: Multi-temporal HiRISE pairs detect meter-scale surface changes (new craters, RSL, frost patterns)
- **Geologic mapping**: Stratigraphic contacts, fault traces, fracture networks mapped at sub-meter precision
- **Stereo DTM generation**: ASP (Ames Stereo Pipeline) workflow: stereo pair → point cloud → DEM → orthoimage

## Tags

HiRISE, MRO, high-resolution, pushbroom, University-of-Arizona, 25cm-resolution, RED-channel, BG-channel, NIR-channel, IRB-composite, stereo, DTM, ESP-product-ID, PSP, RSL, gullies, polar-processes, boulder-tracking, Curiosity, Perseverance, InSight, landing-site, rock-abundance, slope-analysis, change-detection, ASP, Ames-Stereo-Pipeline
