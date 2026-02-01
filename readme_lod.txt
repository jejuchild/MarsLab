Use camera height above Mars ellipsoid surface as the ONLY zoom proxy.

Thresholds (MUST use these exact values unless justified)
FAR VIEW (Global)

camera height > 10,000 km

Behavior:

DO NOT fetch footprints

DO NOT render anything

lod = none

MID VIEW (Continental)

10,000 km ≥ height > 6,000 km

Behavior:

Fetch centroid points only

lod = point

Hard cap: max 3,000 points

No polygons, no outlines

MID VIEW (Regional)

6,000 km ≥ height > 3,000 km

Behavior:

Fetch centroid points only

lod = point

Cap can increase (e.g. 5,000)

Used for candidate-area discovery

CLOSE VIEW (Local / Analysis)

height ≤ 3,000 km

Behavior:

Fetch polygon footprints

lod = poly

Geometry simplification by height:

3,000–1,000 km → simplify = low

1,000–300 km → simplify = mid

<300 km → simplify = high

Enforcement (STRICT)

Client:

MUST NOT request polygons unless height ≤ 3,000 km

MUST NOT render polygons if accidentally returned

Server:

If lod=poly but height > 3,000 km:

downgrade to lod=point OR

return empty FeatureCollection with reason

Rendering constraints

FAR: zero draw calls

MID: PointPrimitiveCollection only

CLOSE: GroundPrimitive / Primitive polygons only

outline disabled

minimal material

no translucency

UX feedback

If height > 6,000 km:

show “Zoom in to see footprints”

If polygon response is truncated:

show “Too many footprints — zoom in further”

This gating is mandatory and must be enforced consistently on both client and server.