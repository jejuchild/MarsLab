# Dielectric Constant Estimation Using Terraced Craters (SHARAD)

## Concept

- Terraced craters are interpreted as forming due to contrasts in subsurface material properties.
- Terrace levels are assumed to correspond to subsurface material interfaces.
- Terrace depth is treated as the depth to a subsurface interface.
- When a SHARAD radar track crosses or passes near a terraced crater, the observed subsurface reflector can be correlated with the terrace depth to estimate the dielectric constant of the intervening material.

## Physical Assumptions

- The terraced crater exposes a real subsurface interface (not a structural or impact-related artifact alone).
- The SHARAD subsurface reflector corresponds to the same interface exposed by the terrace.
- Radar propagation through the subsurface is approximated as vertical (nadir geometry).
- SHARAD time delay is two-way travel time (signal descends to interface, reflects, and returns).
- The material between the surface and the interface is treated as homogeneous for the purpose of dielectric estimation.

## Required Inputs

| Parameter | Symbol | Units | Source |
|-----------|--------|-------|--------|
| Terrace depth | depth | meters | DTM or topographic profile across the crater |
| SHARAD two-way travel time | t | seconds | Radargram time-pick of subsurface reflector |
| Speed of light in vacuum | c | m/s | Constant: 299,792,458 m/s |

## Method

### Electromagnetic wave velocity in a dielectric medium

```
v = c / sqrt(εr)
```

### Two-way travel time relationship

The SHARAD signal travels down to the interface and back:

```
t = 2 * depth / v
```

Substituting the velocity:

```
t = 2 * depth * sqrt(εr) / c
```

### Dielectric constant inversion

Solving for the real part of the dielectric constant:

```
εr = (c * t / (2 * depth))^2
```

Where:
- **c** = 299,792,458 m/s (speed of light in vacuum)
- **t** = SHARAD two-way travel time to the subsurface reflector (seconds)
- **depth** = terrace depth measured from surface to the terrace level (meters)
- **εr** = real part of the relative dielectric constant (dimensionless)

### Computational example

For a terrace depth of 50 m and SHARAD two-way travel time of 0.59 μs:

```
εr = (299792458 * 0.59e-6 / (2 * 50))^2
   = (176877.55 / 100)^2
   = (1768.78)^2
   ≈ 3.13
```

This is consistent with water ice.

## Interpretation Guidelines

| εr Range | Interpretation | Notes |
|----------|---------------|-------|
| ≈ 3.0–3.2 | Water ice | Clean, solid H₂O ice |
| ≈ 2.5–3.0 | Porous or ice-rich regolith | High-porosity material with significant ice content |
| ≈ 4–6 | Dry basaltic regolith | Typical Mars surface material without ice |
| ≈ 6–9 | Dense basalt / consolidated rock | Intact volcanic rock |
| < 2.0 | Suspect — likely measurement error | Below physical minimum for geological materials |
| > 15 | Suspect — likely measurement error | Unusually high; check for clutter or misidentified reflector |

**Important**: These values are heuristic reference anchors derived from laboratory measurements and Mars orbital studies. They must be reported as assumptions and interpreted in context, not treated as absolute material classifications.

Reference dielectric constants:
- Pure water ice: εr ≈ 3.15 (Matsuoka et al., 1997)
- Ice-rich permafrost: εr ≈ 2.8–4.0 (depends on porosity and dust fraction)
- Dry Mars soil: εr ≈ 4–8 (Heggy et al., 2001)

## Uncertainty and Failure Modes

### Sources of uncertainty

1. **Terrace depth uncertainty**: DTM vertical accuracy is typically ±1–5 m for HiRISE DTMs. For a 50 m terrace, this introduces ~2–10% uncertainty in εr.
2. **SHARAD time-pick uncertainty**: Radargram time resolution is ~0.0375 μs (1/bandwidth). Manual picks may vary by ±1–2 samples, introducing ~5–15% uncertainty depending on depth.
3. **Non-vertical propagation**: Off-nadir clutter or dipping interfaces violate the vertical-propagation assumption, biasing εr estimates.
4. **Material inhomogeneity**: Layered or mixed materials produce an effective εr that does not correspond to a single material.

### Failure conditions

- **εr < 1**: Physically impossible (vacuum = 1). Indicates measurement error — likely incorrect time-pick or depth measurement.
- **εr > 15**: Unphysically large for near-surface Mars materials. Check for surface clutter misidentified as subsurface reflector.
- **Terrace–reflector mismatch**: The terrace may not correspond to the same interface producing the SHARAD reflector. This is especially likely when:
  - The SHARAD track does not cross the crater directly
  - Multiple subsurface interfaces are present
  - The crater has been significantly modified by erosion
- **Spatial mismatch**: SHARAD ground tracks have ~3–6 km cross-track footprint. If the nearest track is >5 km from the crater, the lateral heterogeneity of the subsurface makes correlation unreliable. The distance between crater center and SHARAD track must always be reported.

### Recommended validation checks

1. Compare εr with independent estimates from the same region (e.g., MARSIS, thermal inertia)
2. Check whether multiple SHARAD tracks crossing the same crater yield consistent εr
3. Verify that the terrace is not an impact-structural feature (compare with non-terraced craters nearby)
4. Report εr with explicit uncertainty bounds: εr ± δεr

## Application Context

This methodology is applicable when:
- Terraced craters are identified in HiRISE or CTX imagery
- SHARAD radargram data is available for tracks crossing or passing near the crater
- Subsurface ice characterization is required for landing site assessment or resource evaluation

The estimated εr value feeds directly into:
- **Ice confidence scoring**: εr ≈ 3.1 provides strong supporting evidence for subsurface ice
- **Depth-to-ice estimation**: Combined with SHARAD travel time, gives calibrated depth
- **Landing site ranking**: Regions with εr consistent with ice rank higher for ISRU potential

## Tags

subsurface, SHARAD, dielectric, ice-detection, terraced-crater, methodology, DTM, radar
