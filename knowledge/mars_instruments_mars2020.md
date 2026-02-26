# Mars 2020 / Perseverance Rover Instruments

## Mission Overview

Mars 2020 launched July 30, 2020, and landed in Jezero Crater (18.4°N, 77.7°E) on February 18, 2021, using the sky crane EDL system. Jezero Crater (~49 km diameter) was selected for its ancient delta deposit and evidence of a paleolake — a high-priority astrobiology target. The mission's primary objectives are: (1) seek signs of ancient microbial life, (2) characterize geology and climate, (3) cache samples for Mars Sample Return (MSR), and (4) demonstrate ISRU and flight technologies. Perseverance carries 7 science instruments plus the Ingenuity helicopter technology demonstrator.

## Instrument Suite Summary

| Instrument | Type | Key Capability |
|------------|------|----------------|
| Mastcam-Z | Stereo zoom camera | Multispectral imaging, 110 mm focal length |
| SuperCam | LIBS + Raman + VNIR | Remote rock characterization at 7 m |
| PIXL | X-ray fluorescence | Elemental mapping at 120 μm spot size |
| SHERLOC | UV Raman + fluorescence | Organic detection, mineral ID |
| RIMFAX | Ground-penetrating radar | Subsurface stratigraphy to ~10 m depth |
| MEDA | Meteorological station | Atmospheric T, P, wind, humidity, dust |
| MOXIE | Electrolysis reactor | O₂ production from CO₂ (ISRU demo) |

## Mastcam-Z

- **Type**: Stereo zoom multispectral camera pair
- **Focal length**: 26–110 mm zoom (each camera); 24.4 cm stereo baseline
- **Resolution**: 7.4 μrad/pixel; ~0.28 mm/pixel at 1 m distance
- **Spectral filters**: 11 narrowband filters per camera (400–1000 nm); RGB + geology-targeted bands
- **Key bands**: 800 nm (iron oxides), 754 nm (ferric minerals), 527 nm (green), 442 nm (blue)
- **Stereo capability**: 3D terrain models at cm-scale; enables precise sample targeting
- **Data volume**: Primary imaging workhorse; >1 million images returned as of 2025

## SuperCam

- **Techniques**: Laser-Induced Breakdown Spectroscopy (LIBS), Time-Resolved Raman, Passive VNIR reflectance, acoustic microphone
- **LIBS range**: Up to 7 m; 1064 nm laser; elemental composition (C, H, N, O, Na, Mg, Al, Si, K, Ca, Ti, Fe, etc.)
- **Raman range**: Up to 7 m; 532 nm laser; mineral identification (sulfates, carbonates, silicates, organics)
- **VNIR**: 0.4–0.85 μm passive reflectance; iron mineralogy
- **Microphone**: First acoustic recordings on Mars; wind noise, laser plasma pops, dust devil sounds
- **Spot size**: ~0.4 mm at 3 m (LIBS); ~0.7 mm at 3 m (Raman)

## PIXL — Planetary Instrument for X-ray Lithochemistry

- **Type**: Micro X-ray fluorescence (μXRF) spectrometer
- **Spot size**: 120 μm; enables sub-mm elemental mapping
- **Elements detected**: Na through U (Z = 11–92); quantitative elemental abundances
- **Standoff distance**: Contact instrument; mounted on robotic arm
- **Scan area**: Up to 25 × 25 mm per session; produces 2D elemental maps
- **Key findings**: Identified sulfate-rich veins, phosphate minerals, and potential biosignature-relevant textures in Jezero delta rocks
- **Hexagonal scan pattern**: 4,000+ individual spectra per target; ~8 hours per full scan

## SHERLOC — Scanning Habitable Environments with Raman & Luminescence for Organics & Chemicals

- **Type**: Deep-UV Raman spectroscopy + fluorescence imager
- **Laser**: 248.6 nm (deep UV); excites aromatic organics and minerals
- **Raman capability**: Identifies carbonates, sulfates, perchlorates, silicates, organics
- **Fluorescence**: Detects aromatic organic compounds at ppb sensitivity
- **WATSON camera**: Context imager co-boresighted with SHERLOC; 6.9 μm/pixel at 48 mm standoff
- **Organic detection**: First in-situ UV Raman on Mars; detected organic molecules in Jezero delta sediments (Farley et al. 2023, Science)
- **Standoff**: Contact instrument on robotic arm; 48 mm working distance

## RIMFAX — Radar Imager for Mars' Subsurface Experiment

- **Type**: Ground-penetrating radar (GPR)
- **Frequency range**: 150–1,200 MHz (stepped frequency)
- **Depth penetration**: Up to 10 m in dry regolith; ~20 m in ice-rich material
- **Vertical resolution**: ~15 cm in regolith
- **Along-track resolution**: ~30 cm (at rover driving speed)
- **Key findings**: Imaged subsurface stratigraphy beneath Jezero delta; identified tilted sedimentary layers and erosional unconformities (Hamran et al. 2022, Science Advances)
- **Antenna**: Downward-looking; mounted on rear of rover

## MEDA — Mars Environmental Dynamics Analyzer

- **Sensors**: Air temperature (5 sensors), ground temperature (IR), pressure, wind speed/direction (2D), relative humidity, dust particle size/shape (RDLS)
- **Sampling rate**: 1 Hz (meteorology); continuous pressure at 2 Hz
- **Key findings**: Documented convective vortices, dust devil passages, boundary layer structure; first humidity measurements in Jezero
- **Relevance**: Atmospheric data for EDL planning of future missions; climate model validation

## MOXIE — Mars Oxygen In-Situ Resource Utilization Experiment

- **Type**: Solid oxide electrolysis cell (SOEC)
- **Process**: CO₂ → CO + O (at 800°C); O ions transported through zirconia electrolyte → O₂
- **Production rate**: ~6–10 g O₂/hour at full operation
- **Total O₂ produced**: ~122 g across 16 runs (April 2021 – August 2023)
- **Significance**: First production of O₂ on another planet; demonstrated ISRU viability
- **Scale-up requirement**: Human Mars mission needs ~2 kg O₂/hour (200× MOXIE scale)
- **Status**: Experiment concluded August 2023; all objectives met

## Ingenuity Helicopter

- **Type**: Autonomous rotorcraft technology demonstrator
- **Mass**: 1.8 kg; rotor diameter 1.2 m (coaxial counter-rotating)
- **Power**: Solar-charged lithium-ion batteries; 350 W peak motor power
- **First flight**: April 19, 2021 — first powered controlled flight on another planet
- **Total flights**: 72 flights completed; final flight January 18, 2024 (rotor blade damage on landing)
- **Max altitude**: 24 m; max speed: 10 m/s; max range per flight: ~704 m
- **Cameras**: Color nadir camera (13 MP); B&W navigation camera
- **Legacy**: Demonstrated aerial reconnaissance viability; informed Mars Science Helicopter concept

## Sample Caching Status

- **Samples cached**: 23 rock core samples + 1 atmospheric sample tube sealed as of early 2025
- **Depot**: 10 samples deposited at "Three Forks" depot in Jezero Crater floor (Jan 2023) as MSR backup cache
- **Sample types**: Igneous (olivine-bearing floor rocks), sedimentary (delta carbonates, sulfates), regolith
- **Tube design**: Titanium tubes, hermetically sealed; designed for 10+ year storage on Mars surface
- **MSR timeline**: Sample retrieval mission target: early 2030s (pending funding and mission design)

## Relevance to MarsLab

- **Ground-truth for orbital data**: PIXL and SHERLOC elemental/mineralogical data validate CRISM and OMEGA orbital detections at outcrop scale
- **Mineral identification validation**: SuperCam Raman confirms mineral phases inferred from orbital spectroscopy
- **In-situ composition data**: Elemental abundances from PIXL provide ground-truth for geochemical models
- **Subsurface stratigraphy**: RIMFAX data constrains sedimentary architecture beneath orbital-resolution imagery
- **Atmospheric baseline**: MEDA meteorological data supports atmospheric modeling used in MarsLab climate analyses

## Tags

mars2020, perseverance, jezero, mastcam-z, supercam, pixl, sherloc, rimfax, meda, moxie, ingenuity, sample-caching, mars-sample-return, isru, astrobiology, in-situ, raman, xrf, gpr
