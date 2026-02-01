ZOOM LEVEL GATING (NON-NEGOTIABLE)

Implement strict zoom / camera-height based gating.
Do NOT render polygon footprints unless the camera is sufficiently zoomed in.

1) Define zoom proxy

Use camera height (distance from Mars ellipsoid surface) as zoom proxy.

Example thresholds (tune if needed, but must exist):

FAR VIEW

camera height > 2,000 km

Behavior:

DO NOT fetch footprints at all

lod = none

MID VIEW

2,000 km ≥ height > 500 km

Behavior:

Fetch centroid points only

lod = point

Max count cap (e.g. 3,000 points)

CLOSE VIEW

height ≤ 500 km

Behavior:

Fetch polygon footprints

lod = poly

Use geometry simplification based on height:

500–200 km → simplify = mid

<200 km → simplify = high

These exact numbers can be constants in a config file, but gating logic MUST exist.

2) Enforce gating on BOTH client and server

Client:

Must NOT request polygons when zoom is FAR or MID.

Must NOT render polygons if returned accidentally.

Server:

If lod=poly but zoom/height too large:

either downgrade to lod=point

or return empty FeatureCollection with reason

3) Rendering rules (hard constraints)

FAR:

zero footprint rendering (nothing)

MID:

PointPrimitiveCollection ONLY

no polygons, no outlines

CLOSE:

GroundPrimitive / Primitive polygons only

outline disabled

minimal material (no translucency)

4) UI feedback

When user is too zoomed out:

show hint: “Zoom in to see footprints”

If polygon data is truncated:

show: “Too many footprints — zoom in further”

5) Acceptance criteria (zoom gating)

At global view: 0 footprint draw calls

At mid zoom: points only

Polygons appear ONLY when zoomed in

No FPS drop when panning at global scale

This zoom gating is NOT optional.
Implement as a first-class system, not a heuristic.