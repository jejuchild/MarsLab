# Mars Mid-Latitude Viscous Flow Feature Diagnostic Criteria

## Purpose
This synthesis captures practical, literature-grounded criteria for discriminating four Mars viscous flow landform classes used in mapping and machine-learning workflows: Lobate Debris Aprons (LDA), Lineated Valley Fill (LVF), Concentric Crater Fill (CCF), and Glacier-Like Forms (GLF).

## Topographic Context
### LDA: scarp- and massif-adjacent aprons
- Located at the base of mesas, massif walls, and fretted terrain scarps.
- Characterized by radial to downslope spreading away from source slopes.
- Often forms broad lobate margins with convex-downslope profiles.

### LVF: valley-confined flow systems
- Occupies linear or branching valleys, troughs, and fretted channels.
- Displays confinement by valley walls and integration of tributary lineations.
- Long-axis flow indicators align with valley orientation.

### CCF: crater-internal flow fill
- Resides inside impact craters, commonly at mid-latitudes.
- Presents basinward organization with concentric or arcuate ridges and troughs.
- Morphology expresses crater-wall control rather than external apron spreading.

### GLF: small alpine-style glaciers
- Found in alcoves and small valleys, typically with clear headwall source zones.
- Shows compact tongue-like forms and local moraine-like arcuate ridges.
- Commonly less extensive than regional LDA/LVF/CCF populations.

## Morphometric Tendencies
### Typical scale envelopes
- LDA: often broad aprons that can exceed tens of km in length.
- LVF: valley length can be large, but width is constrained by valley geometry.
- CCF: scale tied to crater diameter; fill patterns often annular or concentric.
- GLF: generally small, often less than approximately 10 km scale.

### Latitude and elevation context
- Most classes cluster in mid-latitudes where ice-assisted creep is favored.
- Regional setting should be interpreted with local topography and thermal context.
- Elevation alone is not diagnostic, but interacts with slope and basin geometry.

## Surface Texture Diagnostics
### Shared ice-related textures
- Brain terrain, polygonal/pitted textures, and lineations can occur across classes.
- Debris cover can mute diagnostic texture in lower-resolution data.

### Texture cues by class
- LDA: lobate fronts, lineations parallel to apparent flow, occasional pitted mantles.
- LVF: strong longitudinal lineations, tributary convergence, valley-axis banding.
- CCF: concentric ridges/troughs and crater-centric texture organization.
- GLF: small tongues, possible nested arcuate ridges interpreted as moraine-like.

## Slope Angle Heuristics
### First-order slope guides
- GLF source alcoves frequently include steep headwalls near approximately 30 degrees.
- LDA/LVF/CCF surfaces are usually low-gradient, often less than approximately 5 degrees.
- Interpret slope with caution: rough DEMs, mixed pixels, and local scarps can bias estimates.

## MOLA-Derivable Features for Classification Support
### Core raster-derived predictors
- Slope mean and slope variability.
- Roughness or TRI-style local relief metrics.
- TPI at multiple windows to quantify ridges, basins, and confinement.
- Curvature and profile shape descriptors to capture lobateness and basin fill structure.

### Useful interpretations
- Strongly confined negative TPI corridors support LVF hypotheses.
- Radially spreading low-slope aprons with convex profiles support LDA.
- Crater-internal concentric topography plus low gradients support CCF.
- Small steep headwall plus short tongue and local arcuate ridges supports GLF.

## Five-Step Diagnostic Flowchart
### Step 1: Determine geomorphic container
Is the feature in a crater interior, valley corridor, scarp base apron, or alcove-headwall setting?

### Step 2: Check planform organization
Assess whether geometry is crater-concentric, valley-parallel, radially spreading, or compact tongue-like.

### Step 3: Evaluate slope structure
Measure headwall and body slopes. Headwalls near approximately 30 degrees suggest GLF sources; broad body slopes less than approximately 5 degrees are common for LDA/LVF/CCF.

### Step 4: Verify texture and flow markers
Look for lineations, concentric ridges, pitted/brain terrain, tributary junctions, and terminus morphology.

### Step 5: Resolve class with confidence tags
Assign LDA, LVF, CCF, or GLF with a confidence level. If signatures are mixed, mark as ambiguous and preserve competing hypotheses.

## Ambiguity and Mixed Cases
### Frequent confusion pairs
- LDA vs LVF where confinement transitions into apron-like spreading.
- CCF vs LVF in crater breaches connected to valley systems.
- Small LDA vs GLF in limited-resolution imagery.

### Recommended handling
- Use multi-source evidence: context, slope, morphometry, and texture.
- Preserve uncertainty labels for borderline examples.
- Revisit class assignment with higher-resolution DEM/imaging when available.
