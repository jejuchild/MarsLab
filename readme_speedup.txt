We have a severe performance issue: rendering ALL footprints globally (GeoJSON) makes the app unusably slow. Refactor the footprint pipeline end-to-end to be viewport-based, LOD-aware, and cache-friendly. Do NOT ask me for permission; implement directly and commit.

0) Current problem (must fix)

We currently load/display too many footprints at once (global GeoJSON + client toggles).

This causes huge CPU/GPU load (draw calls, entity count, re-renders) and memory pressure.

1) Target behavior (requirements)

Implement these core behaviors:

(A) Viewport-based loading (MANDATORY)

Only fetch and render footprints that intersect the current camera view bounding box (bbox).

Fetch should happen on camera move end, not continuously:

Use moveEnd (Cesium) or equivalent.

Add debounce (e.g., 200–400ms) to avoid request spam.

API should support:

instrument=CRISM|HIRISE

bbox=minLon,minLat,maxLon,maxLat in degrees (handle antimeridian crossing properly)

zoom=<number> or camera height derived zoom proxy

optionally limit to cap results

(B) Zoom gating / LOD (MANDATORY)

Use zoom/height thresholds:

If zoom is far-out (low detail):

Do NOT render polygons.

Render either nothing or centroids as points (cheap).

Mid zoom:

Render points only (centroids).

Close zoom:

Render polygon footprints.

Define thresholds (adjust empirically):

Example:

zoom < Z1: none (or heatmap later)

Z1 ≤ zoom < Z2: centroids points only

zoom ≥ Z2: polygon footprints

(C) Geometry simplification (STRONGLY RECOMMENDED)

Footprint polygons have too many vertices.

Provide simplified geometries depending on zoom/LOD:

simplify=low|mid|high OR tolerance=<meters or degrees>

Use Douglas–Peucker (or similar) server-side.

Ensure topology doesn’t break (no self-intersections if possible; at least keep it valid GeoJSON).

(D) Client-side caching (MANDATORY)

Cache footprint responses by (instrument, lod, bboxTileKey) so revisiting an area doesn’t refetch.

Use an LRU cache with a max size (e.g., 50–200 tiles).

Support aborting in-flight requests when a new moveEnd happens (AbortController).

(E) Hard limits + progressive rendering (MANDATORY)

If bbox query returns too many features:

server should return truncated=true and count_returned, count_total_estimate

client shows warning like “Zoom in to see more footprints”

Progressive rendering:

render points first, then polygons if needed

avoid re-creating all primitives on every update; update only diff

2) Backend tasks (implement)

We are using a backend (FastAPI). Implement new endpoints (or refactor existing):

Endpoint 1: footprints in bbox

GET /api/footprints
Query params:

instrument (required)

bbox (required) format minLon,minLat,maxLon,maxLat

lod (optional) one of none|point|poly

simplify (optional) low|mid|high OR tolerance

limit (optional, default e.g. 2000)

Return:

GeoJSON FeatureCollection

Include metadata fields:

truncated: boolean

returned: number

total_estimate: number (if available)

lod: string

simplify: string|number

Implementation details:

Use spatial index:

If using PostGIS, ensure GIST index on geometry.

If file-based, build a tile index (quadkey or simple grid) once and query intersecting tiles quickly.

Handle antimeridian bbox:

If minLon > maxLon, split query into two bboxes and merge results.

Geometry simplification:

Use shapely (or equivalent) simplify with preserve_topology when possible.

Ensure output remains valid polygons.

Endpoint 2: centroid points only (optional)

If it’s easier, you can implement centroids as a separate endpoint:
GET /api/footprints/centroids
But prefer single endpoint with lod=point.

3) Frontend tasks (implement)

We are using Cesium in React.

(A) Compute bbox from camera view

On moveEnd:

compute visible rectangle in WGS84 degrees (or Mars ellipsoid equivalent already used)

produce bbox string

Handle cases where rectangle is undefined (space view / horizon):

fallback to a reasonable bbox around camera target or skip fetch

(B) Determine zoom/LOD

Derive zoom from camera height or Cesium’s computed “distance to surface”.

Map to LOD:

far => lod=none

mid => lod=point

close => lod=poly and pick simplify level

(C) Rendering strategy (Cesium performance)

Do NOT create thousands of Entities if avoidable.

Prefer:

Points: PointPrimitiveCollection

Polygons: GroundPrimitive / Primitive with GeometryInstances

Avoid per-feature React state; keep Cesium primitives managed outside React render loop.

Diff updates:

compare previous set of feature IDs vs new set

add/remove only changed instances

(D) UI controls

Layer panel toggles should not cause full re-fetch unless needed.

When user toggles CRISM footprints off:

hide primitives without clearing cache

Add “Footprints density” notice:

If truncated=true, show “Too many footprints. Zoom in.”

4) Deliverables

Code changes for backend + frontend

Update any config/constants (LOD thresholds, limits)

Add a short PERF_NOTES.md explaining:

why old approach was slow

new pipeline overview (viewport fetch + LOD + simplify + cache)

how to tune thresholds

5) Acceptance criteria (must meet)

Initial map load is fast (no global footprint load).

Panning/zooming does not freeze UI.

Footprints appear only when zoomed in enough.

Switching between CRISM/HIRISE layers is instantaneous due to caching.

No request spam while dragging (debounced moveEnd only).

Works across antimeridian.

Implement now.