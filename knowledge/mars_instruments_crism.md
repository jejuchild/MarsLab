# CRISM — Compact Reconnaissance Imaging Spectrometer for Mars

## Overview

- **Mission**: Mars Reconnaissance Orbiter (MRO), launched August 2005, Mars orbit insertion March 2006
- **PI Institution**: Johns Hopkins University Applied Physics Laboratory (JHUAPL)
- **Type**: Visible/Near-Infrared (VNIR) + Infrared (IR) imaging spectrometer
- **Wavelength range**: 362–3,920 nm (0.362–3.92 µm) across two detectors
- **Primary goal**: Map surface mineralogy at high spatial and spectral resolution to constrain aqueous history
- **Detector cooling**: Cryogenic cooler maintained IR detector at ~100 K; cooler failed 2011, degrading long-wavelength IR sensitivity

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Spatial resolution (targeted) | 15–19 m/pixel (from ~300 km altitude) |
| Spatial resolution (survey) | 100–200 m/pixel |
| Spectral channels | 544 channels (VNIR: 362–1,053 nm; IR: 1,002–3,920 nm) |
| Spectral sampling | ~6.55 nm/channel |
| Field of view | 2.12° (targeted); 11.46° (survey) |
| Swath width (targeted) | ~10 km |
| Swath width (survey) | ~300 km |
| Signal-to-noise ratio | >400:1 (VNIR); >100:1 (IR, pre-cooler failure) |
| Data rate | Up to 6 Mbit/s |
| Instrument mass | 33 kg |

## Operating Modes

### Targeted Mode (FRT / HRL)
- **Full Resolution Targeted (FRT)**: 15–19 m/pixel, full 544-channel hyperspectral, ~10 km × 10 km scene
- **Half Resolution Long (HRL)**: 36 m/pixel, full spectral, ~10 km × 10 km; used for larger coverage
- Gimbal-mounted: instrument rotates to track ground target during MRO flyover, enabling longer integration

### Survey / Mapping Mode
- **Multispectral Survey (MSP)**: 72 selected wavelength channels, 100–200 m/pixel, broad swath
- **Hyperspectral Survey (HSP)**: Full 544 channels at reduced spatial resolution
- After 2017 (cooler failure + detector degradation): operations shifted predominantly to MSP mode

### Emission Phase Function (EPF)
- Multi-angle observations of same target at different emission angles
- Used for atmospheric aerosol characterization and surface photometric modeling

## Data Products

| Product Code | Description | Resolution | Spectral |
|-------------|-------------|------------|---------|
| EDR | Experimental Data Record (raw DN) | Native | Full |
| TRR3 | Targeted Reduced Data Record v3 (calibrated I/F) | 15–36 m/pixel | Full 544 ch |
| MTRDR | Map-projected Targeted Reduced Data Record | 15–36 m/pixel | Full 544 ch |
| FRT | Full Resolution Targeted observation | ~18 m/pixel | 544 ch |
| HRL | Half Resolution Long observation | ~36 m/pixel | 544 ch |
| MSP | Multispectral Survey | 100–200 m/pixel | 72 ch |
| HSP | Hyperspectral Survey | 100–200 m/pixel | 544 ch |
| CAT | CRISM Analysis Toolkit summary products | Variable | Derived |

All calibrated products expressed as **I/F** (radiance factor = observed radiance / solar irradiance at Mars).

## Key Mineral Detections

### Phyllosilicates (Noachian-age aqueous alteration)
- **Fe/Mg smectites** (nontronite, saponite): widespread in Noachian highlands, Mawrth Vallis, Jezero delta
- **Al-phyllosilicates** (montmorillonite, kaolinite): upper stratigraphic positions, leaching environments
- **Chlorite, prehnite, serpentine**: deep crustal exposures, hydrothermal settings

### Sulfates (Hesperian evaporite/acid weathering)
- **Polyhydrated sulfates** (kieserite, epsomite): Valles Marineris interior layered deposits
- **Monohydrated sulfates**: Meridiani Planum, Aram Chaos
- **Jarosite**: Meridiani (confirmed by Opportunity); acid-sulfate environment indicator

### Oxides & Carbonates
- **Ferric oxides** (hematite, goethite): Meridiani, widespread dust
- **Carbonates** (magnesite, siderite): Nili Fossae (Ehlmann et al. 2008); rare, implies near-neutral pH water
- **Hydrated silica (opaline silica)**: Jezero delta, Valles Marineris; high-temperature hydrothermal or low-T diagenesis

### Mafic Minerals
- **Olivine**: Nili Fossae, Argyre rim, fresh impact ejecta; Fo60–Fo90 compositions
- **Low-Ca pyroxene (LCP)**: ancient Noachian crust, orthopyroxene-bearing terrains
- **High-Ca pyroxene (HCP)**: Hesperian volcanic plains, Syrtis Major

## Spectral Parameter Summary

| Parameter | Wavelength Region | Mineral Target |
|-----------|------------------|----------------|
| BD1900 | 1,900 nm band depth | H₂O in hydrated minerals |
| BD2100 | 2,100 nm band depth | Monohydrated sulfates (kieserite) |
| BD2210 | 2,210 nm band depth | Al-OH (kaolinite, montmorillonite) |
| BD2350 | 2,350 nm band depth | Fe/Mg-OH (smectites, chlorite) |
| D2300 | 2,300 nm drop | Fe/Mg phyllosilicates |
| BD2500 | 2,500 nm band depth | Carbonates |
| OLINDEX3 | 1,000–1,300 nm | Olivine |
| LCPINDEX2 | 1,800–2,000 nm | Low-Ca pyroxene |
| HCPINDEX2 | 2,200–2,400 nm | High-Ca pyroxene |
| SINDEX2 | 2,100–2,400 nm | Sulfates (broad) |

Parameters computed per-pixel from TRR3/MTRDR cubes; combined into **summary product** RGB composites for rapid mapping.

## Machine Learning Applications

- **CNN mineral classification**: Pixel-wise classification of CRISM cubes using convolutional networks trained on spectral libraries (e.g., USGS, RELAB); achieves >85% accuracy on major mineral classes
- **Spectral unmixing**: Linear/nonlinear unmixing to estimate sub-pixel mineral abundances; endmember extraction via VCA, NFINDR
- **Automated mapping**: Random forest and SVM classifiers applied to summary parameter stacks for regional mineral maps
- **Dimensionality reduction**: PCA, MNF (Minimum Noise Fraction) transforms to isolate spectral variance from noise
- **Anomaly detection**: Autoencoder-based detection of spectrally unusual pixels for discovery of rare minerals
- **Transfer learning**: Models pre-trained on terrestrial hyperspectral data (AVIRIS) fine-tuned on CRISM

## Limitations

- **Atmospheric correction**: CO₂ and H₂O gas absorptions must be removed; residual errors affect 1,400 nm, 1,900 nm, 2,700 nm regions
- **Dust contamination**: Airborne dust and surface dust coatings suppress mineral absorption features; dust index (DI) used to flag affected spectra
- **Calibration drift**: Detector response changed over mission lifetime; TRR3 v3 calibration partially corrects but residual artifacts remain
- **Cooler failure (2011)**: Loss of cryogenic cooling degraded IR detector sensitivity beyond ~2,600 nm; long-wavelength carbonate/sulfate bands compromised
- **Spatial mixing**: 18 m/pixel footprint mixes multiple lithologies; spectral signatures represent area-weighted averages
- **Nadir-only geometry**: Single viewing angle limits photometric correction; EPF mode partially addresses this

## Tags

CRISM, MRO, imaging-spectrometer, VNIR, infrared, mineralogy, phyllosilicates, sulfates, carbonates, olivine, pyroxene, hydrated-silica, TRR3, MTRDR, FRT, HRL, MSP, HSP, spectral-parameters, BD1900, OLINDEX, LCPINDEX, HCPINDEX, machine-learning, spectral-unmixing, atmospheric-correction, Jezero, Nili-Fossae, Mawrth-Vallis
