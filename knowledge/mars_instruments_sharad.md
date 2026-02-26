# SHARAD — Shallow Radar

## Overview

- **Mission**: Mars Reconnaissance Orbiter (MRO), launched August 2005
- **PI Institution**: Istituto Nazionale di Astrofisica (INAF), Italy; funded by Italian Space Agency (ASI)
- **Type**: Orbital subsurface sounding radar (HF band)
- **Primary goal**: Map subsurface dielectric interfaces to depths of ~1 km with ~15 m vertical resolution
- **Complementary instrument**: MARSIS on Mars Express (lower frequency, deeper penetration)
- **Operations**: Continuous since 2006; >700,000 km of ground tracks acquired

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Center frequency | 20 MHz |
| Bandwidth | 10 MHz (15–25 MHz) |
| Free-space vertical resolution | ~15 m |
| Vertical resolution in water ice | ~8.4 m (assuming ε = 3.15) |
| Along-track horizontal resolution (unfocused) | ~3–6 km |
| Along-track resolution (SAR-focused) | ~0.3–1 km |
| Cross-track footprint | ~3–5 km |
| Transmit power | 10 W |
| Pulse repetition frequency | 700 Hz |
| Antenna length | 10 m (dipole, deployed after MOI) |
| Maximum penetration depth | ~1 km (in low-loss materials) |
| Data rate | ~75 kbit/s |

## Operating Principle

### Radar Sounding
- Transmits chirped HF pulses; records time delay and amplitude of reflected echoes
- **Two-way travel time** converted to depth using assumed dielectric constant (ε) of subsurface material
- Depth (m) = (c × t) / (2 × √ε), where c = speed of light, t = two-way travel time

### Dielectric Properties
| Material | Dielectric Constant (ε) | SHARAD Penetration |
|----------|------------------------|-------------------|
| Water ice | 3.15 | ~1 km |
| Dry basalt | 7–9 | ~200–400 m |
| Regolith (dry) | 2.5–4 | ~500–800 m |
| Liquid water | 80 | Very shallow |
| CO₂ ice | 2.1 | ~1 km |

### SAR Processing
- Synthetic Aperture Radar (SAR) focusing applied along-track to improve horizontal resolution from ~3–6 km to ~0.3–1 km
- Clutter simulation using MOLA topography required to distinguish surface returns from subsurface echoes

## Data Products

| Product | Description |
|---------|-------------|
| EDR | Raw telemetry, unprocessed |
| RDR | Range-compressed radargrams (power vs. time delay vs. along-track position) |
| USRDR | US Team Reduced Data Record (SAR-focused radargrams) |
| USGEOM | Geometry files (latitude, longitude, altitude per range line) |
| Clutter simulations | Synthetic radargrams from MOLA DEM to identify surface clutter |
| 3D volumes | Stacked radargrams interpolated to 3D subsurface grids (regional studies) |

Radargrams displayed as 2D images: x-axis = along-track distance, y-axis = two-way travel time (depth proxy).

## Key Discoveries

### Polar Ice Deposits
- **South Polar Layered Deposits (SPLD)**: Mapped internal stratigraphy of ~3.7 km thick CO₂/H₂O ice stack; basal unit identified as nearly pure water ice (Plaut et al. 2007, Science)
- **Bright basal reflector (SPLD)**: High-reflectivity interface at base of SPLD; debated as liquid water vs. CO₂ ice vs. saline ice

### Mid-Latitude Ice
- **Arcadia Planitia**: Subsurface ice deposits at 40–50°N; ice table at ~1–10 m depth (Bramson et al. 2015)
- **Utopia Planitia**: Massive ice deposit ~130–170 m thick, volume ~14,300 km³ (Stuurman et al. 2016, GRL)
- **Deuteronilus Mensae**: Lobate debris aprons confirmed as debris-covered glaciers; ice fraction >80%

### Polar Layered Deposits
- North Polar Layered Deposits (NPLD): Internal layers traced over hundreds of km; record of obliquity-driven climate cycles
- Basal unit of NPLD: Sand-rich layer interpreted as ancient erg (Putzig et al. 2009)

## Comparison with MARSIS

| Parameter | SHARAD | MARSIS |
|-----------|--------|--------|
| Instrument | MRO | Mars Express |
| Frequency | 20 MHz (10 MHz BW) | 1.8–5 MHz (1 MHz BW) |
| Vertical resolution | ~15 m (free space) | ~50–150 m |
| Max penetration | ~1 km | ~3–5 km |
| Best for | Shallow stratigraphy, ice layers | Deep structure, large-scale features |
| Ionospheric effect | Moderate | Severe at low frequencies |
| Horizontal resolution | ~0.3–1 km (SAR) | ~5–10 km |

## Applications for MarsLab

- **Subsurface interface detection**: Identify dielectric boundaries (ice/regolith, rock/ice) from radargram reflectors
- **Ice table depth mapping**: Estimate depth to ice table at mid-latitudes for ISRU site selection
- **Dielectric property estimation**: Invert reflection coefficients for permittivity of subsurface units
- **ISRU site selection**: Identify accessible near-surface ice deposits for water extraction
- **Stratigraphic correlation**: Trace internal layers across polar regions to reconstruct climate history
- **Clutter discrimination**: Combine SHARAD radargrams with MOLA-derived clutter simulations to confirm subsurface origin of reflectors

## Limitations

- **Ionospheric distortion**: Ionosphere disperses HF signals; correction applied but residual phase errors remain, especially during solar events
- **Surface clutter**: Off-nadir surface topography produces spurious echoes that can mimic subsurface reflectors; requires clutter simulation for disambiguation
- **Nadir ambiguity**: Cannot distinguish left/right off-nadir returns without additional processing
- **Penetration depth limit**: High-loss materials (wet regolith, saline ice) severely attenuate signal; effective depth may be <100 m
- **Horizontal resolution**: Even SAR-focused resolution (~300 m) limits detection of small-scale features
- **No direct composition**: Dielectric constant constrains material class but cannot uniquely identify composition without ground truth

## Tags

SHARAD, MRO, radar-sounding, subsurface, HF-radar, dielectric-constant, radargram, SAR-processing, water-ice, polar-layered-deposits, SPLD, NPLD, Arcadia-Planitia, Utopia-Planitia, Deuteronilus-Mensae, MARSIS, ice-table, ISRU, clutter-simulation, stratigraphy, permittivity
