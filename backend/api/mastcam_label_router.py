# api/mastcam_label_router.py
"""
Mastcam-Z Roughness Labeling API
- Coordinate transforms: Mastcam pixel → ground lat/lon → HiRISE projected coords
- WMS proxy for HiRISE imagery from FU Berlin
- Label CRUD: save/load roughness labels per panorama
"""

import os
import math
import json
import logging
import time
from pathlib import Path
from io import BytesIO

import requests as ext_requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger("marslab.mastcam_label")
router = APIRouter(prefix="/api/mastcam-label", tags=["Mastcam-Z Labeling"])

# ── Constants (Mastcam-Z specs: Bell et al. 2021, Space Science Reviews) ──
MARS_R = 3396190.0  # Mars radius (m) - FU Berlin map projection
# Mastcam-Z camera height: RSM elevation axis at 191.9 cm + boresight 8.0 cm above
# + rover ground clearance ~20 cm on flat terrain → total ~2.12 m (Bell et al. 2021)
ROVER_CAM_HEIGHT = 2.12  # Mastcam-Z boresight height above surface (m)
WMS_BASE = "https://maps.planet.fu-berlin.de/jez-bin/wms"
WMS_SRS = "EPSG:49911"  # Mars sinusoidal
LABEL_DIR = Path(os.environ.get("MASTCAM_LABEL_DIR", "/disk1/cspark/mastcam/labels"))
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Coordinate math ─────────────────────────────────────

def lonlat_to_proj(lon: float, lat: float) -> tuple[float, float]:
    """Convert Mars geographic (lon, lat) to EPSG:49911 projected (x, y)."""
    x = lon / 180.0 * math.pi * MARS_R
    y = lat / 180.0 * math.pi * MARS_R
    return x, y


def proj_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """Convert EPSG:49911 projected (x, y) to Mars geographic (lon, lat)."""
    lon = x / (math.pi * MARS_R) * 180.0
    lat = y / (math.pi * MARS_R) * 180.0
    return lon, lat


def mastcam_pixel_to_ground(
    px: float, py: float,
    img_width: int, img_height: int,
    rover_lon: float, rover_lat: float,
    heading_offset: float = 0.0,
    cam_height: float = ROVER_CAM_HEIGHT,
) -> dict | None:
    """
    Convert Mastcam equirectangular pixel (px, py) to ground coordinates.
    Uses flat-terrain approximation (good for <50m distance).

    Returns dict with ground_lon, ground_lat, distance_m, azimuth_deg, elevation_deg
    or None if pixel looks above horizon.
    """
    # Pixel to spherical coordinates
    azimuth_deg = (px / img_width) * 360.0 + heading_offset
    elevation_deg = 90.0 - (py / img_height) * 180.0

    # Only ground pixels (looking below horizon)
    if elevation_deg >= 0:
        return None  # sky

    # Flat terrain: distance = height / tan(-elevation)
    elev_rad = math.radians(-elevation_deg)
    if elev_rad < 0.01:  # nearly horizontal → very far
        return None  # too far for flat approx

    distance_m = cam_height / math.tan(elev_rad)
    if distance_m > 100:  # limit to nearby
        return None

    # Ground point offset
    az_rad = math.radians(azimuth_deg)
    # Mars: 1 degree lat ≈ 59.27 km, 1 degree lon ≈ 59.27 * cos(lat) km
    m_per_deg_lat = MARS_R * math.pi / 180.0  # ~59,274 m/deg
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(rover_lat))

    dlat = distance_m * math.cos(az_rad) / m_per_deg_lat
    dlon = distance_m * math.sin(az_rad) / m_per_deg_lon

    ground_lon = rover_lon + dlon
    ground_lat = rover_lat + dlat

    return {
        "ground_lon": round(ground_lon, 8),
        "ground_lat": round(ground_lat, 8),
        "distance_m": round(distance_m, 2),
        "azimuth_deg": round(azimuth_deg % 360, 2),
        "elevation_deg": round(elevation_deg, 2),
    }


def ground_to_mastcam_pixel(
    ground_lon: float, ground_lat: float,
    img_width: int, img_height: int,
    rover_lon: float, rover_lat: float,
    heading_offset: float = 0.0,
    cam_height: float = ROVER_CAM_HEIGHT,
) -> dict | None:
    """
    Inverse: convert ground (lat, lon) to Mastcam equirectangular pixel (px, py).
    Returns dict with px, py, azimuth_deg, elevation_deg, distance_m
    or None if point is behind/above horizon.
    """
    m_per_deg_lat = MARS_R * math.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(rover_lat))

    dx_m = (ground_lon - rover_lon) * m_per_deg_lon  # east offset
    dy_m = (ground_lat - rover_lat) * m_per_deg_lat  # north offset
    distance_m = math.sqrt(dx_m**2 + dy_m**2)

    if distance_m < 0.5 or distance_m > 100:
        return None

    azimuth_rad = math.atan2(dx_m, dy_m)  # atan2(east, north) → clockwise from N
    azimuth_deg = math.degrees(azimuth_rad) % 360

    elevation_rad = math.atan2(-cam_height, distance_m)  # negative = looking down
    elevation_deg = math.degrees(elevation_rad)

    # To equirectangular pixel
    px = ((azimuth_deg - heading_offset) % 360) / 360.0 * img_width
    py = (90.0 - elevation_deg) / 180.0 * img_height

    return {
        "px": round(px, 1),
        "py": round(py, 1),
        "azimuth_deg": round(azimuth_deg, 2),
        "elevation_deg": round(elevation_deg, 2),
        "distance_m": round(distance_m, 2),
    }


# ── API: Coordinate transform ──────────────────────────

class PixelToGroundRequest(BaseModel):
    px: float
    py: float
    img_width: int = 4096
    img_height: int = 2048
    rover_lon: float
    rover_lat: float
    heading_offset: float = 0.0


@router.post("/pixel-to-ground")
def pixel_to_ground(req: PixelToGroundRequest):
    """Convert Mastcam equirectangular pixel to ground lat/lon."""
    result = mastcam_pixel_to_ground(
        req.px, req.py, req.img_width, req.img_height,
        req.rover_lon, req.rover_lat, req.heading_offset,
    )
    if result is None:
        return JSONResponse(content={"error": "sky or too far"}, status_code=200)
    # Add projected coords for HiRISE overlay
    gx, gy = lonlat_to_proj(result["ground_lon"], result["ground_lat"])
    result["proj_x"] = round(gx, 2)
    result["proj_y"] = round(gy, 2)
    return JSONResponse(content=result)


class BrushToGroundRequest(BaseModel):
    """Batch: convert multiple pixels to ground coords (for brush strokes)."""
    pixels: list[list[float]]  # [[px, py], ...]
    img_width: int = 4096
    img_height: int = 2048
    rover_lon: float
    rover_lat: float
    heading_offset: float = 0.0


@router.post("/brush-to-ground")
def brush_to_ground(req: BrushToGroundRequest):
    """Batch convert brush stroke pixels to ground lat/lon."""
    results = []
    for px, py in req.pixels:
        r = mastcam_pixel_to_ground(
            px, py, req.img_width, req.img_height,
            req.rover_lon, req.rover_lat, req.heading_offset,
        )
        if r:
            gx, gy = lonlat_to_proj(r["ground_lon"], r["ground_lat"])
            results.append({
                "px": px, "py": py,
                "lon": r["ground_lon"], "lat": r["ground_lat"],
                "proj_x": round(gx, 2), "proj_y": round(gy, 2),
                "distance_m": r["distance_m"],
            })
    return JSONResponse(content={"points": results, "count": len(results)})


class GroundToMastcamRequest(BaseModel):
    ground_lon: float
    ground_lat: float
    img_width: int = 4096
    img_height: int = 2048
    rover_lon: float
    rover_lat: float
    heading_offset: float = 0.0


@router.post("/ground-to-mastcam")
def ground_to_mastcam(req: GroundToMastcamRequest):
    """Convert ground lat/lon to Mastcam equirectangular pixel."""
    result = ground_to_mastcam_pixel(
        req.ground_lon, req.ground_lat,
        req.img_width, req.img_height,
        req.rover_lon, req.rover_lat, req.heading_offset,
    )
    if result is None:
        return JSONResponse(content={"error": "out of range"})
    return JSONResponse(content=result)


class HiriseGridRequest(BaseModel):
    """Generate a grid of HiRISE pixels around the rover and map each to Mastcam coords."""
    rover_lon: float
    rover_lat: float
    heading_offset: float = 0.0
    radius_m: float = 50
    hirise_pixel_m: float = 0.25  # HiRISE resolution: 25cm/px
    img_width: int = 4096
    img_height: int = 2048


@router.post("/hirise-grid")
def hirise_grid(req: HiriseGridRequest):
    """Generate HiRISE pixel grid around rover, with Mastcam pixel mapping for each."""
    m_per_deg_lat = MARS_R * math.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(req.rover_lat))

    step = req.hirise_pixel_m
    n = int(req.radius_m / step)

    cells = []
    for iy in range(-n, n + 1):
        for ix in range(-n, n + 1):
            dx_m = ix * step
            dy_m = iy * step
            dist = math.sqrt(dx_m**2 + dy_m**2)
            if dist > req.radius_m or dist < 0.5:
                continue

            glon = req.rover_lon + dx_m / m_per_deg_lon
            glat = req.rover_lat + dy_m / m_per_deg_lat

            mc = ground_to_mastcam_pixel(
                glon, glat,
                req.img_width, req.img_height,
                req.rover_lon, req.rover_lat,
                req.heading_offset,
            )

            cells.append({
                "ix": ix, "iy": iy,
                "dx_m": round(dx_m, 2), "dy_m": round(dy_m, 2),
                "lon": round(glon, 8), "lat": round(glat, 8),
                "distance_m": round(dist, 2),
                "mc_px": mc["px"] if mc else None,
                "mc_py": mc["py"] if mc else None,
            })

    return JSONResponse(content={
        "cells": cells,
        "count": len(cells),
        "hirise_pixel_m": step,
        "radius_m": req.radius_m,
    })


# ── API: WMS proxy for HiRISE ──────────────────────────

@router.get("/hirise-tile")
def get_hirise_tile(
    lon: float = Query(...),
    lat: float = Query(...),
    radius_m: float = Query(100),
    width: int = Query(800),
    height: int = Query(800),
    layer: str = Query("HiRISE-hsv"),
):
    """Proxy WMS GetMap for HiRISE imagery around a point."""
    cx, cy = lonlat_to_proj(lon, lat)
    bbox = f"{cx-radius_m},{cy-radius_m},{cx+radius_m},{cy+radius_m}"

    url = (
        f"{WMS_BASE}?service=WMS&version=1.1.1&request=GetMap"
        f"&layers={layer}&styles=&bbox={bbox}"
        f"&width={width}&height={height}&srs={WMS_SRS}&format=image/jpeg"
    )

    try:
        r = ext_requests.get(url, timeout=15)
        if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
            if len(r.content) < 2000:
                raise HTTPException(404, "No HiRISE coverage at this location")
            return Response(
                content=r.content,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"},
            )
        raise HTTPException(502, f"WMS error: {r.status_code}")
    except ext_requests.RequestException as e:
        raise HTTPException(502, f"WMS fetch failed: {e}")


# ── API: Label CRUD ─────────────────────────────────────

class LabelData(BaseModel):
    panorama_id: str
    rover_lon: float
    rover_lat: float
    heading_offset: float = 0.0
    labels: list[dict]  # [{class, pixels:[[px,py],...], ground_points:[{lon,lat},...]}]


@router.post("/labels")
def save_labels(data: LabelData):
    """Save roughness labels for a panorama."""
    safe_id = data.panorama_id.replace("/", "_").replace("..", "")
    path = LABEL_DIR / f"{safe_id}.json"
    payload = data.model_dump()
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"[LABEL] Saved {len(data.labels)} label groups for {safe_id}")
    return JSONResponse(content={"status": "ok", "path": str(path)})


@router.get("/labels/{panorama_id}")
def load_labels(panorama_id: str):
    """Load saved labels for a panorama."""
    safe_id = panorama_id.replace("/", "_").replace("..", "")
    path = LABEL_DIR / f"{safe_id}.json"
    if not path.exists():
        return JSONResponse(content={"labels": [], "panorama_id": panorama_id})
    data = json.loads(path.read_text())
    return JSONResponse(content=data)


@router.get("/labels")
def list_labels():
    """List all panoramas with saved labels."""
    results = []
    for f in sorted(LABEL_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            results.append({
                "panorama_id": data.get("panorama_id", f.stem),
                "label_count": len(data.get("labels", [])),
                "saved_at": data.get("saved_at"),
            })
        except Exception:
            pass
    return JSONResponse(content=results)


@router.post("/export")
def export_dataset():
    """Export all labels as ML training dataset."""
    all_labels = []
    for f in sorted(LABEL_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            for label_group in data.get("labels", []):
                for gp in label_group.get("ground_points", []):
                    all_labels.append({
                        "panorama_id": data["panorama_id"],
                        "class": label_group["class"],
                        "lon": gp["lon"],
                        "lat": gp["lat"],
                        "distance_m": gp.get("distance_m"),
                    })
        except Exception:
            pass

    return JSONResponse(content={
        "total_points": len(all_labels),
        "labels": all_labels,
    })
