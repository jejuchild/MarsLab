# Arcadia Planitia — Landing Site Analysis & Ice Resources

## Geographic Overview

| Parameter | Value |
|-----------|-------|
| Location | Northern lowlands, western hemisphere |
| Latitude range | ~40°N – 60°N |
| Longitude range | ~150°E – 210°E |
| Elevation (MOLA) | −3 to −5 km below datum |
| Geologic age | Late Amazonian (relatively young surface) |
| Terrain type | Smooth volcanic plains with periglacial modification |

- Arcadia Planitia is a broad, low-relief plain bounded by Amazonis Planitia to the south and the northern polar region to the north
- The southern portion (~40–50°N) is the primary zone of human landing site interest due to the combination of accessible ice, adequate solar power, and favorable EDL conditions

## Why Arcadia Matters

Arcadia Planitia is considered a **prime candidate for the first human Mars landing** based on three converging factors:

1. **Abundant shallow subsurface ice**: Multiple independent datasets confirm ice within 1 m of the surface across large areas — the most critical ISRU resource for crew water and propellant production
2. **Low elevation for EDL**: At −3 to −5 km MOLA, the atmospheric column is thick enough to decelerate a Starship-class vehicle during entry, descent, and landing
3. **Moderate latitude for solar power**: 40–50°N provides reasonable solar insolation year-round, unlike higher-latitude ice deposits that suffer extreme seasonal darkness

## Subsurface Ice Evidence

### SHARAD Radar Detections
- **Instrument**: SHARAD (SHAllow RADar) on Mars Reconnaissance Orbiter; 15–25 MHz, penetrates 1–2 km
- Subsurface reflectors detected at **10–30 m depth** across southern Arcadia
- Dielectric constant of reflective layer: **~3.0–3.5**, consistent with ice-rich material (pure water ice ~3.15)
- Bramson et al. (2015) identified a laterally continuous buried ice slab extending hundreds of km
- Reflector geometry suggests a **massive ice deposit** rather than ice-cemented regolith alone

### GRS Neutron Spectroscopy
- **Instrument**: Gamma Ray Spectrometer on Mars Odyssey; sensitive to hydrogen within top ~1 m
- High hydrogen abundance in Arcadia indicates **ice within 1 m of surface** across broad areas
- Epithermal neutron suppression consistent with water-equivalent hydrogen >10% by mass
- Spatial resolution ~300 km; confirms regional-scale ice presence

### SWIM Ice Consistency Map
- **SWIM** (Subsurface Water Ice Mapping): Multi-dataset synthesis (Morgan et al. 2021, JGR Planets)
- Arcadia Planitia scores **high consistency** (multiple independent datasets agree) in southern region
- SWIM integrates: SHARAD, GRS, thermal inertia, geomorphology, and CRISM data
- High-score zones in Arcadia overlap with proposed SpaceX landing site corridors

### Thermal Inertia (THEMIS)
- Low thermal inertia in surface layer overlying high thermal inertia substrate
- Pattern consistent with **ice-cemented regolith** beneath a dry, unconsolidated lag deposit
- Seasonal temperature modeling constrains ice table depth to 0.3–1.5 m in southern Arcadia

### Morphological Indicators
- **Expanded secondary craters**: Ejecta excavation into ice causes excess volume; diagnostic of shallow ice
- **Polygonal terrain**: Thermal contraction cracking of ice-rich ground; widespread at 45–60°N
- **Sublimation pits and scalloped depressions**: Active or recent ice loss at surface
- **Lobate debris aprons**: Glacier-like features indicating ice-rich flow; preserved under lag deposits

## Terrain Characterization

### Surface Smoothness
- **CTX (Context Camera)** and **HiRISE** imagery: Low rock abundance (<5% in candidate zones)
- Slopes < 2° over km-scale baselines in southern Arcadia — favorable for large vehicle landing
- Minimal boulder fields compared to equatorial volcanic terrains
- Dust devil tracks visible; surface is mobile fine-grained material over competent substrate

### Dust Cover
- **TES albedo**: Moderate (~0.15–0.20); higher than dust-mantled regions but not pristine
- Dust accumulation manageable for solar panels with periodic cleaning or tilt strategies
- No persistent thick dust mantles that would impede drilling or thermal extraction

### Crater Density
- Low crater density consistent with **Late Amazonian resurfacing** (~100–500 Ma surface age in places)
- Younger surface age implies less secondary ejecta hazard and more intact ice deposits
- Primary craters >1 km spaced widely enough to avoid landing hazard in flat inter-crater plains

## ISRU Potential

| Parameter | Estimate | Notes |
|-----------|----------|-------|
| Ice depth to table | 0.3–1.5 m | Southern Arcadia 40–50°N |
| Ice volume (regional) | >10⁴ km³ | Bramson 2015 lower bound for slab |
| Ice purity | 50–90% by volume | Mixed with regolith; varies by depth |
| Water extraction energy | ~3–5 MJ/kg | Thermal heating of ice + sublimation/melt |
| O₂ production (MOXIE-scale) | ~2 kg/hr (scaled) | Required for crew life support |
| Propellant production | ~30 t CH₄ + 90 t LOX | Estimated for single Starship return |

- Ice extraction methods under study: **resistive heating rods**, **microwave heating**, **steam injection**
- Extracted water electrolyzed for O₂ (crew breathing) and H₂ (Sabatier feedstock for CH₄)
- Regolith overburden (0.3–1.5 m) must be excavated or thermally penetrated before ice access

## SpaceX Landing Site Interest

- SpaceX has publicly identified **southern Arcadia Planitia (~40–50°N)** as a primary candidate for early Starship landings
- Rationale: shallow ice for ISRU + low elevation for EDL + relatively flat terrain for large vehicle
- Candidate corridors evaluated using SWIM maps, SHARAD profiles, and HiRISE terrain analysis
- Landing site selection requires: rock hazard maps at <1 m resolution (HiRISE), slope maps, ice depth confirmation
- **Terrain Relative Navigation** (demonstrated on Perseverance) required for precision landing within safe zones

## MarsLab Relevance

Arcadia Planitia is the **primary study region** for MarsLab research and MARVIS analysis:

- **Ice resource mapping**: SHARAD radargram processing to constrain ice slab geometry and depth
- **Landing site scoring**: Multi-criteria analysis integrating elevation, ice depth, slope, dust, and solar power
- **SHARAD profile analysis**: Identifying subsurface reflectors, computing dielectric constants, mapping lateral extent
- **SWIM integration**: Reproducing and extending Morgan et al. (2021) consistency scores with updated datasets
- **Thermal modeling**: Constraining ice table depth from THEMIS thermal inertia + seasonal temperature data
- MarsLab tools are designed to ingest MRO/SHARAD data and produce ice probability maps centered on Arcadia

## Key References

| Reference | Finding |
|-----------|---------|
| Bramson et al. (2015), GRL | SHARAD detection of massive buried ice slab in Arcadia; dielectric constant ~3.0 |
| Bramson et al. (2017), JGR | Extended mapping of ice slab; lateral continuity over >400 km |
| Morgan et al. (2021), JGR | SWIM multi-dataset ice consistency map; Arcadia scores highest in northern plains |
| Dundas et al. (2018), Science | Exposed ice scarps at mid-latitudes; confirms near-surface ice composition |
| Stuurman et al. (2016), GRL | SHARAD ice detection methodology applied to northern plains |
| Mellon & Jakosky (1995) | Theoretical ice stability model; predicts ice table depth vs. latitude |

## Tags

arcadia-planitia, landing-site, subsurface-ice, sharad, isru, swim, grs, thermal-inertia, human-exploration, marslab, starship, ice-resources, northern-lowlands, bramson, morgan
