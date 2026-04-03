# api/mastcam_spice_router.py
"""
Mastcam-Z SPICE Co-registration API
Serves per-sol RAS textures and lon/lat coordinate arrays from the
PDS XYZ + SPICE pipeline output.
"""

import json
import logging
import struct
from pathlib import Path
from io import BytesIO
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

logger = logging.getLogger("marslab.mastcam_spice")
router = APIRouter(prefix="/api/mastcam-spice", tags=["Mastcam-Z SPICE"])

COREGISTER_OUTPUT = Path("/disk1/cspark/mastcam/coregister_data/output")
PDS_CACHE = Path("/disk1/cspark/mastcam/coregister_data/pds_cache/mastcamz")
LABEL_DIR = Path("/disk1/cspark/mastcam/labels/spice")
LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────

def _parse_pds4_label(label_path: Path) -> dict:
    """Minimal PDS4 XML label parser for image dimensions."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(label_path)
    root = tree.getroot()
    meta = {}
    for el in root.iter():
        local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if local == "lines" and el.text:
            try: meta["lines"] = int(el.text.strip())
            except: pass
        elif local == "samples" and el.text:
            try: meta["samples"] = int(el.text.strip())
            except: pass
        elif local == "bands" and el.text:
            try: meta["bands"] = int(el.text.strip())
            except: pass
        elif local == "start_byte" and el.text:
            try: meta["start_byte"] = int(el.text.strip())
            except: pass
    return meta


def _parse_pds3_img_header(img_path: Path) -> dict:
    """Parse embedded PDS3/ODL header from an IMG file."""
    meta = {}
    with open(img_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                if line == "END":
                    break
                continue
            key, _, val = line.partition("=")
            key = key.strip().upper()
            val = val.strip().strip('"').strip("'")
            if "<" in val:
                val = val[:val.index("<")].strip()
            try:
                if key == "RECORD_BYTES":
                    meta["record_bytes"] = int(val)
                elif key == "^IMAGE" and "HEADER" not in line.upper().split("=")[0]:
                    meta["image_start"] = int(val)
                elif key == "LINES":
                    meta["lines"] = int(val)
                elif key == "LINE_SAMPLES":
                    meta["samples"] = int(val)
                elif key == "BANDS":
                    meta["bands"] = int(val)
                elif key == "SAMPLE_TYPE":
                    meta["sample_type"] = val
                elif key == "SAMPLE_BITS":
                    meta["sample_bits"] = int(val)
            except (ValueError, IndexError):
                pass
            if line == "END":
                break
    return meta


@lru_cache(maxsize=8)
def _load_ras_texture(sol: int) -> np.ndarray | None:
    """Load RAS texture for a sol, returning (H, W, 3) uint8 RGB."""
    sol_dir = PDS_CACHE / f"sol{sol:05d}"
    if not sol_dir.exists():
        return None

    ras_files = sorted(sol_dir.glob("*RAS*.IMG"))
    if not ras_files:
        return None

    ras_path = ras_files[0]

    # Parse embedded PDS3 header
    meta = _parse_pds3_img_header(ras_path)
    lines = meta.get("lines", 0)
    samples = meta.get("samples", 0)
    bands = meta.get("bands", 3)
    if lines == 0 or samples == 0:
        return None

    # Calculate data offset
    record_bytes = meta.get("record_bytes", 0)
    image_start = meta.get("image_start", 1)
    offset = record_bytes * (image_start - 1)

    # Determine dtype from header
    sample_bits = meta.get("sample_bits", 16)
    sample_type = meta.get("sample_type", "MSB_INTEGER")

    if sample_bits == 16:
        if "LSB" in sample_type or "PC" in sample_type:
            dtype = np.dtype("<i2")
        else:
            dtype = np.dtype(">i2")
    else:  # 32-bit float
        if "LSB" in sample_type or "PC" in sample_type:
            dtype = np.dtype("<f4")
        else:
            dtype = np.dtype(">f4")

    raw = np.fromfile(str(ras_path), dtype=dtype, offset=offset)
    expected = bands * lines * samples
    if raw.size < expected:
        return None

    raw = raw[:expected].reshape(bands, lines, samples)
    if bands >= 3:
        rgb = np.moveaxis(raw[:3], 0, -1).astype(np.float64)
    else:
        rgb = np.stack([raw[0]] * 3, axis=-1).astype(np.float64)

    positive = rgb[rgb > 0]
    if positive.size > 0:
        vmin, vmax = np.nanpercentile(positive, [2, 98])
    else:
        vmin, vmax = float(np.nanmin(rgb)), float(np.nanmax(rgb))
    denom = vmax - vmin if vmax != vmin else 1.0
    return np.clip((rgb - vmin) / denom * 255, 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def _load_lonlat(sol: int):
    """Load lon/lat arrays for a sol."""
    npz_path = COREGISTER_OUTPUT / f"sol{sol:05d}_lonlat.npz"
    if not npz_path.exists():
        return None, None
    data = np.load(str(npz_path))
    return data["lon"], data["lat"]


# ── API Endpoints ──────────────────────────────────────────

@router.get("/sols")
def list_sols():
    """List all processed sols with metadata."""
    results_path = COREGISTER_OUTPUT / "batch_results.json"
    if not results_path.exists():
        return []

    with open(results_path) as f:
        data = json.load(f)

    sols = []
    for sol_str, info in sorted(data.items(), key=lambda x: int(x[0])):
        lr = info.get("lon_range", [0, 0])
        if not (77.3 < lr[0] < 77.7):
            continue
        sols.append({
            "sol": int(sol_str),
            "product_id": info["product_id"],
            "valid_points": info["valid_points"],
            "lon_range": info["lon_range"],
            "lat_range": info["lat_range"],
        })
    return sols


@router.get("/texture/{sol}")
def get_texture(sol: int, quality: int = Query(85, ge=10, le=100)):
    """Get RAS texture as JPEG for a given sol."""
    texture = _load_ras_texture(sol)
    if texture is None:
        raise HTTPException(404, f"No texture for sol {sol}")

    img = Image.fromarray(texture)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/jpeg")


@router.get("/coords/{sol}")
def get_coords(sol: int):
    """Get coordinate metadata for a sol (valid mask bounds, not full arrays)."""
    lon, lat = _load_lonlat(sol)
    if lon is None:
        raise HTTPException(404, f"No coordinates for sol {sol}")

    valid = ~np.isnan(lon)
    return {
        "sol": sol,
        "width": int(lon.shape[1]),
        "height": int(lon.shape[0]),
        "valid_count": int(np.sum(valid)),
        "total_pixels": int(lon.size),
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
    }


@router.get("/pixel-coord/{sol}")
def get_pixel_coord(sol: int, x: int = Query(...), y: int = Query(...)):
    """Get lon/lat for a specific pixel."""
    lon, lat = _load_lonlat(sol)
    if lon is None:
        raise HTTPException(404, f"No coordinates for sol {sol}")
    if y < 0 or y >= lon.shape[0] or x < 0 or x >= lon.shape[1]:
        raise HTTPException(400, "Pixel out of bounds")

    plon = float(lon[y, x])
    plat = float(lat[y, x])
    valid = not (np.isnan(plon) or np.isnan(plat))
    return {"x": x, "y": y, "lon": plon if valid else None, "lat": plat if valid else None, "valid": valid}


@router.get("/valid-mask/{sol}")
def get_valid_mask(sol: int):
    """Get valid pixel mask as 1-bit PNG (white=valid, black=invalid)."""
    lon, _ = _load_lonlat(sol)
    if lon is None:
        raise HTTPException(404, f"No coordinates for sol {sol}")

    valid = (~np.isnan(lon)).astype(np.uint8) * 255
    img = Image.fromarray(valid, "L")
    buf = BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


@router.get("/ground-to-pixel/{sol}")
def ground_to_pixel(
    sol: int,
    lon: float = Query(...),
    lat: float = Query(...),
    hirise_cell_m: float = Query(0.25, description="HiRISE cell size in meters"),
):
    """Find ALL Mastcam-Z pixels within a HiRISE cell centered on (lon, lat).

    Returns the bounding box of matched pixels and the nearest pixel.
    """
    lonmap, latmap = _load_lonlat(sol)
    if lonmap is None:
        raise HTTPException(404, f"No coordinates for sol {sol}")

    valid = ~np.isnan(lonmap)
    if not np.any(valid):
        return {"found": False}

    m_per_deg_lat = 59274.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(lat))

    # Find all Mastcam-Z pixels within the HiRISE cell (25cm x 25cm)
    half_cell_deg_lon = (hirise_cell_m / 2) / m_per_deg_lon
    half_cell_deg_lat = (hirise_cell_m / 2) / m_per_deg_lat

    in_cell = (
        valid &
        (lonmap >= lon - half_cell_deg_lon) & (lonmap <= lon + half_cell_deg_lon) &
        (latmap >= lat - half_cell_deg_lat) & (latmap <= lat + half_cell_deg_lat)
    )

    n_in_cell = int(np.sum(in_cell))

    if n_in_cell == 0:
        # Fallback: find nearest pixel
        dlat = latmap - lat
        dlon = lonmap - lon
        dist_m2 = (dlon * m_per_deg_lon)**2 + (dlat * m_per_deg_lat)**2
        dist_m2[~valid] = np.inf
        flat_idx = int(np.argmin(dist_m2))
        py, px = divmod(flat_idx, lonmap.shape[1])
        min_dist_m = float(np.sqrt(dist_m2[py, px]))

        if min_dist_m > 0.5:
            return {
                "found": False,
                "reason": f"Outside Mastcam-Z coverage (nearest: {min_dist_m:.1f}m)",
                "nearest_offset_m": round(min_dist_m, 2),
            }

        return {
            "found": True,
            "pixel_x": int(px),
            "pixel_y": int(py),
            "pixel_lon": float(lonmap[py, px]),
            "pixel_lat": float(latmap[py, px]),
            "offset_m": round(min_dist_m, 2),
            "pixels_in_cell": 1,
            "bbox": {"x0": int(px), "y0": int(py), "x1": int(px), "y1": int(py)},
        }

    # Get bounding box of all matched pixels
    ys, xs = np.where(in_cell)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # Center pixel
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    return {
        "found": True,
        "pixel_x": cx,
        "pixel_y": cy,
        "pixel_lon": float(lonmap[cy, cx]),
        "pixel_lat": float(latmap[cy, cx]),
        "offset_m": 0.0,
        "pixels_in_cell": n_in_cell,
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "bbox_size_px": {"w": x1 - x0 + 1, "h": y1 - y0 + 1},
    }


@router.get("/texture-crop/{sol}")
def get_texture_crop(
    sol: int,
    x: int = Query(...), y: int = Query(...),
    radius: int = Query(64, ge=16, le=256),
    quality: int = Query(90, ge=10, le=100),
    bbox_x0: int = Query(None), bbox_y0: int = Query(None),
    bbox_x1: int = Query(None), bbox_y1: int = Query(None),
):
    """Get a cropped region of the RAS texture centered on (x, y).

    If bbox is provided, draws a green rectangle showing the HiRISE cell boundary.
    """
    texture = _load_ras_texture(sol)
    if texture is None:
        raise HTTPException(404, f"No texture for sol {sol}")

    h, w = texture.shape[:2]
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(w, x + radius)
    y1 = min(h, y + radius)

    crop = texture[y0:y1, x0:x1].copy()

    # Draw HiRISE cell boundary on crop
    if bbox_x0 is not None and bbox_y0 is not None and bbox_x1 is not None and bbox_y1 is not None:
        # Convert bbox to crop-local coordinates
        bx0 = max(0, bbox_x0 - x0)
        by0 = max(0, bbox_y0 - y0)
        bx1 = min(crop.shape[1] - 1, bbox_x1 - x0)
        by1 = min(crop.shape[0] - 1, bbox_y1 - y0)

        # Draw green rectangle (HiRISE cell = these Mastcam-Z pixels)
        green = np.array([0, 255, 0], dtype=np.uint8)
        for t in range(2):  # 2px thick
            if by0 + t < crop.shape[0]:
                crop[by0 + t, bx0:bx1 + 1] = green
            if by1 - t >= 0:
                crop[by1 - t, bx0:bx1 + 1] = green
            if bx0 + t < crop.shape[1]:
                crop[by0:by1 + 1, bx0 + t] = green
            if bx1 - t >= 0:
                crop[by0:by1 + 1, bx1 - t] = green

    img = Image.fromarray(crop)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/jpeg")


@router.get("/crop-scale/{sol}")
def get_crop_scale(
    sol: int,
    x: int = Query(...), y: int = Query(...),
    radius: int = Query(64, ge=16, le=256),
):
    """Get ground scale (meters/pixel) for a Mastcam-Z crop region.

    Uses SPICE lon/lat differences between adjacent pixels to compute
    the actual ground resolution at the crop center.
    """
    lonmap, latmap = _load_lonlat(sol)
    if lonmap is None:
        raise HTTPException(404, f"No coordinates for sol {sol}")

    h, w = lonmap.shape
    if y < 0 or y >= h or x < 0 or x >= w:
        raise HTTPException(400, "Pixel out of bounds")

    # Sample pixel pairs with increasing step sizes to find scale
    m_per_deg = 59274.0  # Mars approx
    scales = []
    for step in [10, 20, 50]:
        for dx, dy in [(step, 0), (0, step), (-step, 0), (0, -step)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                lon1, lat1 = float(lonmap[y, x]), float(latmap[y, x])
                lon2, lat2 = float(lonmap[ny, nx]), float(latmap[ny, nx])
                if not (np.isnan(lon1) or np.isnan(lon2)):
                    dist_m = np.sqrt(
                        ((lon2 - lon1) * m_per_deg * np.cos(np.radians(lat1)))**2 +
                        ((lat2 - lat1) * m_per_deg)**2
                    )
                    if dist_m > 0:
                        scales.append(dist_m / step)  # normalize to per-pixel

    if not scales:
        return {"scale_m_per_px": None, "crop_width_m": None}

    avg_scale = float(np.mean(scales))
    crop_size = radius * 2
    return {
        "scale_m_per_px": round(avg_scale, 4),
        "crop_width_m": round(avg_scale * crop_size, 2),
        "crop_height_m": round(avg_scale * crop_size, 2),
    }


# ── Label CRUD ─────────────────────────────────────────────

class SpiceAnnotation(BaseModel):
    pixel_x: int
    pixel_y: int
    lon: float | None = None
    lat: float | None = None
    category: str
    rock_size_m: float = 0.0
    confidence: float = 0.8
    notes: str = ""


class SpiceLabelPayload(BaseModel):
    sol: int
    product_id: str = ""
    annotations: list[SpiceAnnotation]


@router.post("/labels")
def save_labels(payload: SpiceLabelPayload):
    """Save annotations for a sol."""
    from datetime import datetime
    label_file = LABEL_DIR / f"sol{payload.sol:05d}.json"
    data = {
        "sol": payload.sol,
        "product_id": payload.product_id,
        "annotations": [a.model_dump() for a in payload.annotations],
        "saved_at": datetime.now().isoformat(),
    }
    with open(label_file, "w") as f:
        json.dump(data, f, indent=2)
    return {"status": "ok", "count": len(payload.annotations)}


@router.get("/labels/{sol}")
def load_labels(sol: int):
    """Load annotations for a sol."""
    label_file = LABEL_DIR / f"sol{sol:05d}.json"
    if not label_file.exists():
        return {"sol": sol, "annotations": []}
    with open(label_file) as f:
        return json.load(f)


@router.get("/labels")
def list_all_labels():
    """List all sols with saved labels."""
    result = []
    for f in sorted(LABEL_DIR.glob("sol*.json")):
        with open(f) as fh:
            data = json.load(fh)
            result.append({
                "sol": data.get("sol", 0),
                "count": len(data.get("annotations", [])),
                "saved_at": data.get("saved_at", ""),
            })
    return result
