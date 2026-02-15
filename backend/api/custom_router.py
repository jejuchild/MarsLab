# api/custom_router.py
"""
Custom user data upload API.

Allows users to upload GeoTIFF files with Mars CRS validation,
automatic bounds extraction, and overlay generation for Cesium rendering.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from typing import Optional
import os
import json
import uuid
import shutil
from datetime import datetime, timezone

import numpy as np
import rasterio
from PIL import Image

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_DATA_DIR = os.path.join(BASE_DIR, "custom_data")

# Ensure custom data directory exists
os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)

# Mars ellipsoid semi-major axis range (meters) for CRS validation
MARS_RADIUS_MIN = 3_350_000
MARS_RADIUS_MAX = 3_450_000

# Earth semi-major axis (WGS84) for rejection
EARTH_RADIUS = 6_378_137

MAX_OVERLAY_SIZE = 2048
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def validate_mars_crs(crs, filepath: str) -> dict:
    """
    Validate that a CRS is Mars-compatible.

    Returns dict with:
      - valid: bool
      - crs_string: str representation
      - warning: optional warning message
      - error: optional error message (if rejected)
    """
    if crs is None:
        return {
            "valid": True,
            "crs_string": "none",
            "warning": "No CRS found - assuming raw Mars geographic coordinates (lat/lon degrees).",
        }

    crs_str = str(crs)

    # Check for IAU authority (Mars codes are in 49900 series)
    if crs.to_authority():
        authority, code = crs.to_authority()
        if authority.upper() == "IAU":
            return {"valid": True, "crs_string": f"IAU:{code}"}

    # Check the WKT/proj string for Mars indicators
    wkt = crs.to_wkt() if hasattr(crs, "to_wkt") else ""
    proj4 = crs.to_proj4() if hasattr(crs, "to_proj4") else ""

    # Check for "Mars" in the CRS name/WKT
    if "mars" in wkt.lower() or "mars" in proj4.lower():
        return {"valid": True, "crs_string": crs_str}

    # Check semi-major axis from ellipsoid attribute or WKT SPHEROID
    semi_major = None
    try:
        ellipsoid = getattr(crs, "ellipsoid", None)
        if ellipsoid:
            semi_major = getattr(ellipsoid, "semi_major_metre", None)
    except Exception:
        pass

    # Fallback: parse SPHEROID from WKT (e.g. SPHEROID["name",3396190,...])
    if semi_major is None:
        import re
        m = re.search(r'SPHEROID\s*\[\s*"[^"]*"\s*,\s*([\d.]+)', wkt)
        if m:
            semi_major = float(m.group(1))

    if semi_major is not None:
        if MARS_RADIUS_MIN <= semi_major <= MARS_RADIUS_MAX:
            return {"valid": True, "crs_string": crs_str}
        if abs(semi_major - EARTH_RADIUS) < 1000:
            return {
                "valid": False,
                "crs_string": crs_str,
                "error": f"Earth CRS detected (semi-major axis={semi_major:.0f}m). "
                         "This appears to be an Earth dataset, not Mars.",
            }

    # Check for common Earth EPSG codes
    try:
        epsg = crs.to_epsg()
        if epsg is not None:
            # EPSG codes < 32768 are almost always Earth-based
            return {
                "valid": False,
                "crs_string": f"EPSG:{epsg}",
                "error": f"Earth CRS detected (EPSG:{epsg}). "
                         "This appears to be an Earth dataset, not Mars.",
            }
    except Exception:
        pass

    # Unknown CRS - accept with warning
    return {
        "valid": True,
        "crs_string": crs_str,
        "warning": "CRS could not be identified as Mars or Earth. Proceeding with caution.",
    }


def normalize_lon_180(lon: float) -> float:
    """Normalize longitude to -180 to 180 range."""
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def extract_bounds(ds) -> dict:
    """
    Extract and normalize bounds from a rasterio dataset.
    Handles both -180/180 and 0/360 longitude conventions.
    """
    bounds = ds.bounds  # (left, bottom, right, top)
    west = bounds.left
    south = bounds.bottom
    east = bounds.right
    north = bounds.top

    # Detect 0-360 longitude convention and convert
    if west > 180 or east > 180:
        west = normalize_lon_180(west)
        east = normalize_lon_180(east)

    # Clamp latitude
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))

    return {"west": west, "south": south, "east": east, "north": north}


def generate_overlay_png(tif_path: str, out_path: str, max_size: int = MAX_OVERLAY_SIZE):
    """
    Generate a downsampled RGBA PNG overlay from a GeoTIFF.

    - Single-band: grayscale with transparent nodata
    - 3+ bands: RGB with transparent nodata
    """
    with rasterio.open(tif_path) as ds:
        # Calculate output dimensions
        scale = max(ds.width, ds.height) / max_size
        if scale < 1:
            scale = 1
        out_w = max(1, int(ds.width / scale))
        out_h = max(1, int(ds.height / scale))

        nodata = ds.nodata
        band_count = ds.count

        if band_count >= 3:
            # Multi-band: read first 3 as RGB
            r = ds.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
            g = ds.read(2, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
            b = ds.read(3, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)

            # Normalize each band to 0-255
            rgb = np.stack([r, g, b], axis=-1).astype(np.float64)

            # Build nodata mask (any band has nodata)
            mask = np.ones((out_h, out_w), dtype=bool)
            if nodata is not None:
                for band_data in [r, g, b]:
                    mask &= (band_data != nodata)
            mask &= np.isfinite(r) & np.isfinite(g) & np.isfinite(b)

            # Percentile stretch per band
            for i in range(3):
                valid = rgb[:, :, i][mask]
                if len(valid) > 0:
                    p2 = np.percentile(valid, 2)
                    p98 = np.percentile(valid, 98)
                    if p98 > p2:
                        rgb[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2) * 255, 0, 255)
                    else:
                        rgb[:, :, i] = 128
                else:
                    rgb[:, :, i] = 0

            rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
            rgba[:, :, :3] = rgb.astype(np.uint8)
            rgba[:, :, 3] = np.where(mask, 255, 0)

        else:
            # Single-band: grayscale
            data = ds.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)

            # Build nodata mask
            mask = np.isfinite(data)
            if nodata is not None:
                mask &= (data != nodata)

            # Normalize to 0-255 using percentile stretch
            valid = data[mask]
            if len(valid) > 0:
                p2 = np.percentile(valid, 2)
                p98 = np.percentile(valid, 98)
                if p98 > p2:
                    normalized = np.clip((data - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    normalized = np.full_like(data, 128.0)
            else:
                normalized = np.zeros_like(data)

            gray = normalized.astype(np.uint8)

            rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
            rgba[:, :, 0] = gray
            rgba[:, :, 1] = gray
            rgba[:, :, 2] = gray
            rgba[:, :, 3] = np.where(mask, 255, 0)

    # Save as PNG using Pillow
    img = Image.fromarray(rgba, "RGBA")
    img.save(out_path, "PNG")


@router.post("/api/custom/validate")
async def validate_custom_dataset(
    file: UploadFile = File(...),
):
    """
    Validate a GeoTIFF file without saving it.

    Checks Mars CRS, extracts bounds and metadata, and returns
    validation results so the user can review before confirming upload.
    """
    import tempfile

    filename = file.filename or "upload.tif"
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF files (.tif, .tiff) are accepted.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 500 MB.")

    # Write to a temp file for rasterio to read
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as ds:
            crs_result = validate_mars_crs(ds.crs, tmp_path)
            bounds = extract_bounds(ds)

            result = {
                "valid": crs_result.get("valid", True),
                "filename": filename,
                "filesize": len(content),
                "crs": crs_result.get("crs_string", "unknown"),
                "crs_valid": crs_result.get("valid", True),
                "crs_warning": crs_result.get("warning"),
                "crs_error": crs_result.get("error"),
                "bounds": bounds,
                "width": ds.width,
                "height": ds.height,
                "bands": ds.count,
                "dtype": str(ds.dtypes[0]),
                "nodata": ds.nodata,
            }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read GeoTIFF: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/api/custom/upload")
async def upload_custom_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    """
    Upload a GeoTIFF file for visualization on the Mars map.

    Validates Mars CRS, extracts geographic bounds, and generates
    a downsampled overlay PNG for Cesium rendering.
    """
    # Validate file extension
    filename = file.filename or "upload.tif"
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF files (.tif, .tiff) are accepted.")

    # Generate unique ID
    dataset_id = f"custom_{uuid.uuid4().hex[:8]}"
    dataset_dir = os.path.join(CUSTOM_DATA_DIR, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)

    tif_path = os.path.join(dataset_dir, "data.tif")
    overlay_path = os.path.join(dataset_dir, "overlay.png")
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    try:
        # Save uploaded file
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 500 MB.")

        with open(tif_path, "wb") as f:
            f.write(content)

        # Open with rasterio and validate
        with rasterio.open(tif_path) as ds:
            # Validate CRS
            crs_result = validate_mars_crs(ds.crs, tif_path)
            if not crs_result.get("valid", True) and crs_result.get("error"):
                shutil.rmtree(dataset_dir)
                raise HTTPException(status_code=400, detail=crs_result["error"])

            # Extract bounds
            bounds = extract_bounds(ds)

            # Validate bounds are reasonable for Mars
            if bounds["south"] < -90 or bounds["north"] > 90:
                shutil.rmtree(dataset_dir)
                raise HTTPException(status_code=400, detail="Latitude bounds outside valid range (-90 to 90).")

            # Build metadata
            dataset_name = name or os.path.splitext(filename)[0]
            metadata = {
                "id": dataset_id,
                "name": dataset_name,
                "bounds": bounds,
                "crs": crs_result.get("crs_string", "unknown"),
                "crs_valid": crs_result.get("valid", True),
                "crs_warning": crs_result.get("warning"),
                "width": ds.width,
                "height": ds.height,
                "bands": ds.count,
                "dtype": str(ds.dtypes[0]),
                "nodata": ds.nodata,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "original_filename": filename,
            }

        # Generate overlay PNG
        generate_overlay_png(tif_path, overlay_path)

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return JSONResponse(content=metadata)

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on any error
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        raise HTTPException(status_code=500, detail=f"Failed to process GeoTIFF: {str(e)}")


@router.get("/api/custom/datasets")
async def list_custom_datasets():
    """List all uploaded custom datasets."""
    datasets = []

    if not os.path.exists(CUSTOM_DATA_DIR):
        return JSONResponse(content={"datasets": []})

    for entry in sorted(os.listdir(CUSTOM_DATA_DIR)):
        metadata_path = os.path.join(CUSTOM_DATA_DIR, entry, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                datasets.append(json.load(f))

    return JSONResponse(content={"datasets": datasets})


@router.get("/api/custom/{dataset_id}/overlay.png")
async def get_custom_overlay(dataset_id: str):
    """Serve the pre-rendered overlay PNG for a custom dataset."""
    overlay_path = os.path.join(CUSTOM_DATA_DIR, dataset_id, "overlay.png")

    if not os.path.exists(overlay_path):
        raise HTTPException(status_code=404, detail=f"Overlay not found: {dataset_id}")

    with open(overlay_path, "rb") as f:
        return Response(f.read(), media_type="image/png")


@router.delete("/api/custom/{dataset_id}")
async def delete_custom_dataset(dataset_id: str):
    """Delete an uploaded custom dataset and all its files."""
    dataset_dir = os.path.join(CUSTOM_DATA_DIR, dataset_id)

    if not os.path.exists(dataset_dir):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    shutil.rmtree(dataset_dir)

    return JSONResponse(content={"deleted": dataset_id})
