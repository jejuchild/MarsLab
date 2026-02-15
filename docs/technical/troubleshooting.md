# Troubleshooting Guide

This document provides solutions for common issues, debugging techniques, and known problems in MarsLab.

---

## Symptom-Cause-Fix Reference

### Quick Reference Table

| Symptom | Cause | Fix |
|---------|-------|-----|
| Footprints not appearing | Camera too zoomed out (LOD) | Zoom in below 15,000 km |
| Footprints not appearing | Visibility toggle off | Enable in LayerPanel |
| Footprints not appearing | Haven't clicked "Load" | Click load button for instrument |
| Empty footprint response | Index file missing | Check `backend/*_data/index.geojson` |
| Overlay not showing | Product doesn't exist locally | Download via `/api/download` |
| Map blank/black | Base layer failing to load | Check NASA Trek connectivity |
| "Fly to" not working | Product not in footprints | Load footprints first |
| CRISM spectrum empty | ENVI files missing | Download full observation |
| HiRISE overlay slow | Large GeoTIFF processing | Check disk cache |
| UI freezes on pan | Too many entities | Zoom out or clear overlays |
| SHARAD tracks wrong path | Antimeridian crossing | Known issue - interpolation needed |
| Products missing from filter | Longitude normalization | Check -180 to 180 range |
| API 404 errors | Backend not running | Start backend server |
| API CORS errors | Wrong proxy config | Check vite.config.ts |

---

## Common Issues

### 1. Footprints Not Loading

**Symptoms:**
- Clicking "Load Footprints" does nothing
- Console shows errors
- Footprint count stays at 0

**Diagnostic Steps:**

```bash
# 1. Check backend is running
curl http://localhost:8000/health

# 2. Check index file exists
ls -la backend/crism_data/index.geojson
ls -la backend/hirise_data/index.geojson
ls -la backend/sharad_data/index.geojson

# 3. Test API directly
curl "http://localhost:8000/api/footprints?instrument=CRISM&bbox=-180,-90,180,90"
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Backend not running | Start: `uvicorn app:app --reload --port 8000` |
| Index file missing | Regenerate or restore from backup |
| Invalid bbox | Ensure coordinates are in valid range |
| LOD enforced | Zoom in to view footprints |

**Code Reference:** `backend/api/footprints_router.py:219-296`

---

### 2. Dateline Crossing Footprints

**Symptoms:**
- Footprints near ±180° longitude appear incorrectly
- SHARAD tracks render as straight lines across globe
- Bounding box queries miss products at dateline

**Root Cause:**

The antimeridian (International Date Line at ±180°) requires special handling. A bounding box from 170° to -170° actually spans 20° (crossing the dateline), not 340°.

**Code Handling:**

```python
# backend/api/footprints_router.py:80-138
def feature_intersects_bbox(feature, min_lon, min_lat, max_lon, max_lat, crosses_antimeridian):
    # Detect antimeridian crossing when min_lon > max_lon
    if crosses_antimeridian:
        # Feature intersects if in western OR eastern part
        in_western_part = feat_max_lon >= min_lon  # 170° to 180°
        in_eastern_part = feat_min_lon <= max_lon  # -180° to -170°
        return in_western_part or in_eastern_part
```

**Frontend Handling:**

```typescript
// frontend/src/utils/FootprintManager.ts:366-424
// SHARAD tracks crossing antimeridian need interpolation
if (lonSpan > 120) {
  // Interpolate through antimeridian instead of across globe
  ...
}
```

**Workaround:**

For extreme cases, split queries at the antimeridian:
```bash
# Instead of bbox=-170,-50,170,50 (crosses dateline)
# Use two queries:
curl ".../api/footprints?bbox=-170,-50,-180,50"
curl ".../api/footprints?bbox=180,-50,170,50"
```

---

### 3. Missing Products Due to Longitude Normalization

**Symptoms:**
- Products don't appear in filter results
- Spatial search returns fewer results than expected
- Footprints visible but not selectable

**Root Cause:**

Inconsistent longitude conventions:
- Some data uses 0° to 360° (east positive)
- Application expects -180° to 180°

**Normalization Function:**

```python
# backend/api/footprints_router.py:71-77
def normalize_lon(lon: float) -> float:
    """Normalize longitude to -180 to 180 range."""
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon
```

**Verification:**

```bash
# Check index file coordinates
cat backend/crism_data/index.geojson | jq '.features[0].geometry.coordinates'

# Should be in range [-180, 180], not [0, 360]
```

**Fix:**

Reprocess index file to normalize coordinates:

```python
import json

with open("index.geojson") as f:
    data = json.load(f)

def normalize_lon(lon):
    while lon > 180: lon -= 360
    while lon < -180: lon += 360
    return lon

for feature in data["features"]:
    coords = feature["geometry"]["coordinates"]
    if feature["geometry"]["type"] == "Polygon":
        for ring in coords:
            for coord in ring:
                coord[0] = normalize_lon(coord[0])

with open("index_normalized.geojson", "w") as f:
    json.dump(data, f)
```

---

### 4. Overlay Slowness Due to Layer Recreation

**Symptoms:**
- Toggling overlays takes several seconds
- UI becomes unresponsive when changing overlays
- Console shows multiple entity creation/destruction

**Root Cause:**

Each overlay toggle was recreating all overlay entities, not just the changed one.

**Current Implementation:**

```typescript
// frontend/src/pages/MainPage.tsx:351-383
const derivedOverlays = useMemo(() => {
  // Derive overlay lists from activeOverlays Map
  // Only recomputes when activeOverlays changes
}, [activeOverlays]);
```

**Performance Tips:**

1. **Batch entity operations:**
   ```typescript
   viewer.entities.suspendEvents();
   // ... make changes
   viewer.entities.resumeEvents();
   ```

2. **Reuse entities when possible:**
   ```typescript
   // Instead of remove + add, update existing
   const entity = viewer.entities.getById(id);
   if (entity) {
     entity.rectangle.material.alpha = newOpacity;
   }
   ```

3. **Limit active overlays:**
   - Warn users when > 10 overlays active
   - Auto-deactivate oldest overlays

---

### 5. Ice Score Filter Not Working

**Symptoms:**
- Filter returns empty results
- Filtered IDs don't match visible footprints
- Score statistics show 0 observations

**Diagnostic Steps:**

```bash
# 1. Check score stats file exists
ls -la backend/crism_score/score_stats.json

# 2. Verify file contents
cat backend/crism_score/score_stats.json | jq 'keys | length'

# 3. Test API directly
curl "http://localhost:8000/api/filter/ice?min_score=0.1&min_percent=1"

# 4. Check available thresholds
curl "http://localhost:8000/api/score/stats"
```

**Common Issues:**

| Issue | Solution |
|-------|----------|
| score_stats.json missing | Run `scripts/generate_score_maps.py` |
| Wrong threshold used | API snaps to nearest precomputed threshold |
| Observation IDs don't match | Ensure base_key matches (e.g., `frt00003156` not full product ID) |

**Code Reference:** `backend/app.py:461-521`

---

### 6. Backend Connection Errors

**Symptoms:**
- "Failed to fetch" errors in console
- 502 Bad Gateway
- Network requests hanging

**Diagnostic Steps:**

```bash
# 1. Check backend is running
ps aux | grep uvicorn

# 2. Check port is listening
netstat -tlnp | grep 8000
# or
lsof -i :8000

# 3. Test direct connection
curl http://localhost:8000/health

# 4. Check Vite proxy (dev mode)
cat frontend/vite.config.ts | grep proxy -A 20
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Backend not started | `cd backend && uvicorn app:app --reload --port 8000` |
| Wrong port | Check port in vite.config.ts matches backend |
| Firewall blocking | `sudo ufw allow 8000` |
| CORS issues | Check middleware in app.py |

---

### 7. Cesium Rendering Issues

**Symptoms:**
- Black screen
- Entities not visible
- Globe not rendering

**Diagnostic Steps:**

```typescript
// Add to MapView.tsx for debugging
console.log("Viewer:", viewerRef.current);
console.log("Entity count:", viewerRef.current?.entities.values.length);
console.log("Scene mode:", viewerRef.current?.scene.mode);
```

**Common Causes:**

| Issue | Solution |
|-------|----------|
| Missing Cesium assets | Check `vite-plugin-cesium` configured |
| Invalid ellipsoid | Verify Mars ellipsoid parameters |
| Camera underground | Reset camera position |
| WebGL disabled | Enable WebGL in browser |

**Reset Camera:**

```typescript
viewer.camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000),
  orientation: {
    heading: 0,
    pitch: -Cesium.Math.PI_OVER_TWO,
    roll: 0,
  },
});
```

---

## Debugging Techniques

### Frontend Debugging

**Console Logging:**

```typescript
// Add to FootprintManager.ts
console.log("[FootprintManager] Loading:", instrument, "bbox:", bbox);
console.log("[FootprintManager] Response:", response.metadata);
```

**React DevTools:**

1. Install React DevTools browser extension
2. Open Components tab
3. Inspect MainPage state
4. Check props passing to children

**Network Tab:**

1. Open DevTools → Network
2. Filter by "api" or "footprints"
3. Check request parameters
4. Verify response data

**Cesium Inspector:**

```typescript
// Enable Cesium inspector
viewer.extend(Cesium.viewerCesiumInspectorMixin);
```

### Backend Debugging

**Add Logging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.get("/api/footprints")
async def get_footprints(...):
    logger.debug(f"Request: instrument={instrument}, bbox={bbox}")
    # ...
    logger.debug(f"Returning {len(features)} features")
```

**Interactive Debugging:**

```bash
# Install debugpy
pip install debugpy

# Run with debugger
python -m debugpy --listen 5678 -m uvicorn app:app --reload
```

Then attach VS Code debugger.

**Test Endpoints Directly:**

```bash
# Use curl with verbose output
curl -v "http://localhost:8000/api/footprints?instrument=CRISM&bbox=-10,-5,10,5"

# Use jq to parse JSON
curl -s ".../api/footprints?..." | jq '.metadata'
```

---

## Log Locations

| Component | Log Location |
|-----------|--------------|
| Backend (dev) | Terminal stdout |
| Backend (prod) | `journalctl -u marslab` |
| Frontend (dev) | Browser console |
| Nginx | `/var/log/nginx/access.log`, `/var/log/nginx/error.log` |

---

## Known Issues

### 1. Memory Usage with Large Overlays

**Issue:** Activating many overlays increases memory usage significantly.

**Workaround:** Limit to ~10 active overlays; deactivate unused ones.

**Future Fix:** Implement overlay virtualization or tile-based rendering.

### 2. SHARAD Antimeridian Rendering

**Issue:** SHARAD tracks crossing the antimeridian may render incorrectly.

**Workaround:** Current code interpolates through antimeridian, but edge cases exist.

**Code Reference:** `frontend/src/utils/FootprintManager.ts:366-424`

### 3. Slow Initial Load with Many Footprints

**Issue:** First footprint load can be slow due to index parsing.

**Workaround:** LRU cache warms after first request.

**Future Fix:** Preload indexes on server startup.

### 4. Overlay Cache Growth

**Issue:** `.overlay_cache/` directory grows unbounded.

**Workaround:** Periodically clear old cache entries:

```bash
# Clear cache entries older than 30 days
find backend/.overlay_cache -name "*.png" -mtime +30 -delete
```

---

## Error Messages Reference

| Error Message | Meaning | Solution |
|---------------|---------|----------|
| `Index not found at...` | GeoJSON index missing | Create or restore index file |
| `Invalid bbox format` | Malformed bbox parameter | Use `minLon,minLat,maxLon,maxLat` |
| `Unknown instrument` | Invalid instrument name | Use `CRISM`, `HIRISE`, or `SHARAD` |
| `Product not found` | Product ID doesn't exist | Verify product ID spelling |
| `Failed to read GeoTIFF` | Corrupted or missing file | Re-download product |
| `ENVI cube not found` | CRISM data files missing | Download observation files |
| `ODE search failed` | External API error | Check internet connectivity |
| `Download task not found` | Invalid task ID | Use valid UUID from POST response |

---

## Getting Help

### Self-Service Resources

1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Review PERF_NOTES.md for performance issues
4. Check browser console and network tab

### Reporting Issues

When reporting issues, include:

1. **Environment:**
   - OS and version
   - Browser and version
   - Python version
   - Node.js version

2. **Steps to reproduce:**
   - Exact actions taken
   - Expected vs actual behavior

3. **Logs:**
   - Browser console output
   - Backend server logs
   - Network request/response

4. **Screenshots:**
   - Error messages
   - UI state

### GitHub Issues

File issues at: https://github.com/anthropics/claude-code/issues

Include the label `marslab` for project-specific issues.
