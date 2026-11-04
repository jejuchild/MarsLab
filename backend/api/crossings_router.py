"""
SHARAD × HiRISE crossings — where a radar ground track passes over an image.

- GET /api/crossings/sharad/{product_id}/hirise
    Trace ranges of a SHARAD hi-res radargram that lie inside HiRISE images.
- GET /api/crossings/hirise/{product_id}/sharad
    Ground-track segments of SHARAD hi-res products that cross a HiRISE image.

Both use the per-trace RDR geometry (planetocentric, like the map) rather
than the coarse index polylines, so the lines land where the radar was.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from .validation import validate_product_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/crossings", tags=["Crossings"])

# Coarse index lines can be off by a few km from the real track.
_INDEX_MARGIN_DEG = 0.2


def _lon180(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def _polygon(feature: dict):
    from shapely.geometry import shape

    geom = feature.get("geometry")
    if not geom:
        return None
    coords = geom.get("coordinates")
    if geom.get("type") == "Polygon" and coords:
        ring = [[float(_lon180(x)), float(y)] for x, y in coords[0]]
        try:
            return shape({"type": "Polygon", "coordinates": [ring]})
        except Exception:
            return None
    return None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] index pairs (inclusive) of consecutive True values."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


_track_cache: dict[str, tuple] = {}
_TRACK_CACHE_MAX = 200


def _track(product_id: str):
    """
    Per-trace (lat, lon, n) without touching the radar echoes: going through
    the radargram loader also maps and NaN-scans the multi-GB power array,
    which made a 50-track lookup take minutes.
    """
    import os
    from .sharad_highres_router import (
        LAT_OFFSET, LON_OFFSET, ROW_BYTES, _cache_path, _get_product_data,
    )

    key = product_id.upper()
    if key in _track_cache:
        return _track_cache[key]

    data = _get_product_data(product_id)
    if data.get("geometry") is not None:
        lat, lon = data["geometry"]["lat"], data["geometry"]["lon"]
    else:
        geom_path = _cache_path(data, "geometry.npz")
        if os.path.exists(geom_path):
            with np.load(geom_path) as g:
                lat, lon = g["lat"], g["lon"]
        else:
            dat = data["info"]["dat_path"]
            if not os.path.exists(dat):
                raise FileNotFoundError(f"SHARAD RDR .dat not found: {dat}")
            mm = np.memmap(dat, dtype=np.uint8, mode="r")
            rows = min(int(data["info"]["total_rows"]), mm.size // ROW_BYTES)
            raw = mm[: rows * ROW_BYTES].reshape(rows, ROW_BYTES)
            lat = np.ascontiguousarray(raw[:, LAT_OFFSET:LAT_OFFSET + 8]).view("<f8").ravel()
            lon = np.ascontiguousarray(raw[:, LON_OFFSET:LON_OFFSET + 8]).view("<f8").ravel()

    out = (np.asarray(lat, dtype=float), _lon180(lon), int(len(lat)))
    if len(_track_cache) >= _TRACK_CACHE_MAX:
        _track_cache.pop(next(iter(_track_cache)))
    _track_cache[key] = out
    return out


@router.get("/sharad/{product_id}/hirise")
def sharad_over_hirise(product_id: str, step: int = Query(5, ge=1, le=100)):
    """HiRISE images crossed by this radargram, as full-resolution trace ranges."""
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID")
    try:
        lats, lons, total = _track(product_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    import shapely
    from .footprints_router import get_or_build_spatial_index

    idx = np.arange(0, total, step)
    pts = shapely.points(lons[idx], lats[idx])
    tree, boxes, features = get_or_build_spatial_index("hirise")
    if tree is None:
        return {"product_id": product_id, "total_traces": total, "crossings": []}

    pairs = tree.query(pts, predicate="intersects")  # (point idx, feature idx)
    crossings = []
    for fi in np.unique(pairs[1]):
        feat = features[int(fi)]
        poly = _polygon(feat)
        if poly is None:
            continue
        cand = pairs[0][pairs[1] == fi]
        inside = np.zeros(len(idx), dtype=bool)
        inside[cand] = shapely.contains_xy(poly, lons[idx][cand], lats[idx][cand])
        for a, b in _runs(inside):
            mid = (a + b) // 2
            crossings.append({
                "product_id": feat["properties"].get("product_id"),
                "start_trace": int(idx[a]),
                "end_trace": int(min(total - 1, idx[b] + step - 1)),
                "lat": round(float(lats[idx][mid]), 5),
                "lon": round(float(lons[idx][mid]), 5),
            })
    crossings.sort(key=lambda c: c["start_trace"])
    return {"product_id": product_id, "total_traces": total, "crossings": crossings}


@router.get("/hirise/{product_id}/sharad")
def hirise_crossed_by_sharad(product_id: str, max_points: int = Query(400, ge=20, le=2000)):
    """SHARAD hi-res ground-track segments inside a HiRISE image's footprint."""
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID")

    from shapely.geometry import LineString, box, shape
    import shapely
    from .footprints_router import get_or_build_spatial_index, load_geojson_index

    hirise = next(
        (f for f in load_geojson_index("hirise").get("features", [])
         if str(f.get("properties", {}).get("product_id", "")).upper() == product_id.upper()),
        None,
    )
    poly = _polygon(hirise) if hirise else None
    if poly is None:
        raise HTTPException(status_code=404, detail=f"No footprint for {product_id}")

    w, s, e, n = poly.bounds
    near = box(w - _INDEX_MARGIN_DEG, s - _INDEX_MARGIN_DEG, e + _INDEX_MARGIN_DEG, n + _INDEX_MARGIN_DEG)
    tree, boxes, features = get_or_build_spatial_index("sharad_highres")
    candidates = []
    if tree is not None:
        for fi in tree.query(near):
            feat = features[int(fi)]
            coords = feat.get("geometry", {}).get("coordinates") or []
            if len(coords) < 2:
                continue
            xy = np.asarray(coords, dtype=float)
            xy[:, 0] = _lon180(xy[:, 0])
            # Split at the antimeridian so the line doesn't span the globe.
            breaks = np.flatnonzero(np.abs(np.diff(xy[:, 0])) > 180) + 1
            if any(LineString(part).intersects(near) for part in np.split(xy, breaks) if len(part) >= 2):
                candidates.append(feat["properties"]["product_id"])

    tracks = []
    for pid in candidates:
        try:
            lats, lons, total = _track(pid)
        except Exception as exc:  # no local radargram data → can't open it anyway
            logger.debug("crossings: skip %s (%s)", pid, exc)
            continue
        inside = shapely.contains_xy(poly, lons, lats)
        for a, b in _runs(inside):
            sel = np.arange(a, b + 1)
            if len(sel) > max_points:
                sel = sel[np.linspace(0, len(sel) - 1, max_points).astype(int)]
            tracks.append({
                "product_id": pid,
                "start_trace": int(a),
                "end_trace": int(b),
                "coords": [[round(float(lons[i]), 6), round(float(lats[i]), 6)] for i in sel],
            })
    return {"product_id": product_id, "bounds": [w, s, e, n], "tracks": tracks}
