# MOLA — Mars Orbiter Laser Altimeter

## Overview

MOLA (Mars Orbiter Laser Altimeter) was a laser altimeter aboard NASA's Mars Global Surveyor (MGS) spacecraft. Developed by NASA Goddard Space Flight Center under David Smith, it operated from September 1999 through June 2001 (primary mapping phase), with limited operations continuing until MGS contact was lost in November 2006. MOLA produced the definitive global topographic map of Mars, transforming understanding of Martian geology, hydrology, and climate. It fired ~671 million laser pulses during the mapping mission, measuring surface elevation by timing the round-trip travel of 1064 nm laser pulses. MGS flew in a near-circular, near-polar orbit at ~378 km altitude.

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Laser wavelength | 1064 nm (Nd:YAG) |
| Pulse rate | 10 Hz |
| Pulse energy | ~45 mJ |
| Pulse width | ~8 ns |
| Beam divergence | 0.4 mrad |
| Footprint diameter | ~160 m at 378 km altitude |
| Along-track shot spacing | ~300 m |
| Cross-track spacing (equator) | ~4 km |
| Vertical accuracy | ~1 m (relative); ~30 m (absolute) |
| Range precision | ~37.5 cm |
| Receiver aperture | 50 cm |
| Detector | Silicon avalanche photodiode |

## MOLA MEGDR — Mission Experiment Gridded Data Record

The MEGDR is the primary global topographic product derived from MOLA data:

- **Resolution**: 128 pixels/degree (~463 m/pixel at equator); also available at 4, 16, 32, 64 px/degree
- **Vertical datum**: Mars areoid (equipotential surface defined by mean equatorial radius of 3,396,000 m)
- **Format**: Binary raster, 16-bit signed integer; available via NASA PDS Geosciences Node
- **Coverage**: Near-complete global coverage; data gaps filled by interpolation (primarily in polar regions and early orbit tracks)
- **Coordinate system**: Planetocentric latitude, east-positive longitude (IAU 2000)
- **Accuracy**: Absolute vertical accuracy ~30 m; relative accuracy ~1 m over baselines < 100 km

## Key Data Products

- **Global DEM (MEGDR)**: Primary topographic reference for all Mars mission planning and science
- **Areoid/geoid**: Mars gravitational equipotential surface; essential for hydrological flow modeling
- **Slope maps**: First and second derivative of elevation; identifies steep terrain, fault scarps, channel walls
- **Roughness maps**: RMS height variation at 0.6 km baseline; distinguishes smooth plains from rugged highlands
- **1064 nm reflectivity**: Passive measurement of surface reflectance at laser wavelength; correlates with albedo and dust cover
- **Polar layer profiles**: Detailed cross-sections of north and south polar layered deposits; ice volume estimates

## Scientific Applications

### Hydrological Flow Modeling
- MOLA DEM enables watershed delineation, flow accumulation, and drainage network extraction
- Identified ancient valley network outlets, alluvial fans, and delta deposits (e.g., Jezero Crater delta)
- Paleolake basin identification: closed topographic depressions with inlet/outlet channels (e.g., Eberswalde, Gale, Jezero)
- Hellas Basin floor at −8.2 km: highest atmospheric pressure on Mars; candidate for liquid water stability

### Crater Morphometry
- Depth/diameter ratios distinguish fresh craters from degraded ones; quantifies erosion rates
- Rim height, ejecta volume, and floor flatness measurable at 100 m scale
- Identified anomalously shallow craters in northern lowlands — consistent with sediment infill or volatile-rich targets

### Volcanic Volume Estimation
- Tharsis rise volume: ~3 × 10⁸ km³ (enough to depress lithosphere and tilt Mars' spin axis)
- Olympus Mons volume: ~2.4 × 10⁶ km³ (largest volcanic edifice in solar system)
- Lava flow thickness and extent measurable from topographic steps

### Landing Site Elevation Constraints
- All Mars landers require elevation below −1.3 km (parachute deployment requires sufficient atmospheric column)
- MOLA elevation is primary constraint for entry, descent, and landing (EDL) trajectory design
- Slope and roughness maps identify hazardous terrain at 100–500 m scale

## Role in MarsLab

- **Baseline topography**: All MarsLab spatial analyses reference MOLA MEGDR as the elevation foundation
- **Slope calculations**: Terrain slope derived from MOLA DEM for ice stability modeling (slope affects insolation and drainage)
- **Landing site elevation assessment**: MOLA elevation confirms EDL viability for candidate sites
- **Hydrological context**: Watershed and flow routing analyses use MOLA DEM to identify ancient water pathways
- **Integration with HiRISE/CTX DTMs**: MOLA provides absolute elevation reference; photogrammetric DTMs provide higher-resolution relative topography

## Limitations

- **No new data**: MGS contact lost November 2, 2006; MOLA dataset is static
- **Spatial resolution**: 160 m footprint and ~300 m along-track spacing insufficient for small-scale features; HiRISE stereo DTMs (1 m/pixel) and CTX DTMs (20 m/pixel) provide superior local resolution
- **Cross-track gaps**: ~4 km spacing at equator; interpolation artifacts visible in flat terrain
- **Polar data gaps**: Oblique illumination geometry and atmospheric scattering reduce accuracy near poles
- **Absolute accuracy**: ~30 m absolute vertical uncertainty limits precise elevation comparisons across large distances
- **No stereo**: Single-beam altimeter cannot resolve sub-footprint topographic variation

## Tags

mola, topography, dem, laser-altimeter, mars-global-surveyor, mgs, megdr, elevation, slope, roughness, hydrology, crater-morphometry, landing-site, areoid, polar-layers
