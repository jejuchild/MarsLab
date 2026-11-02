"""HiRISE DTM elevation query endpoint."""
import json
import os

import numpy as np
import rasterio
import rasterio.windows
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.paths import HIRISE_DTM_DIR
from api.rate_limit import limiter
from api.validation import validate_product_id

router = APIRouter()


@router.get("/hirise_dtm/elevation/{product_id}")
@limiter.limit("20/minute")
def get_hirise_dtm_elevation(
    request: Request,
    product_id: str,
    lat: float,
    lon: float,
    radius: float = 0.01
):
    """
    Get elevation data from HiRISE DTM at a specific location.
    Returns elevation value and statistics within radius.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    from pyproj import CRS, Transformer

    # Find DTM file
    index_path = os.path.join(HIRISE_DTM_DIR, "index.geojson")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="HiRISE DTM index not found")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    dtm_file = None
    for feature in index.get("features", []):
        props = feature.get("properties", {})
        if props.get("product_id") == product_id:
            dtm_file = props.get("dtm_file")
            break

    if not dtm_file:
        raise HTTPException(status_code=404, detail=f"DTM not found for: {product_id}")

    dtm_path = os.path.join(HIRISE_DTM_DIR, dtm_file)
    if not os.path.exists(dtm_path):
        raise HTTPException(status_code=404, detail=f"DTM file not found: {dtm_file}")

    try:
        ds = rasterio.open(dtm_path)

        # Transform lat/lon to DTM CRS
        mars_lonlat = CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")
        transformer = Transformer.from_crs(mars_lonlat, ds.crs.to_wkt(), always_xy=True)

        x, y = transformer.transform(lon, lat)

        # Get pixel coordinates
        row, col = ds.index(x, y)

        # Check bounds
        if row < 0 or row >= ds.height or col < 0 or col >= ds.width:
            ds.close()
            raise HTTPException(status_code=400, detail="Location outside DTM bounds")

        # Read single value
        elevation = ds.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]

        # Read a small patch for stats
        patch_size = 50
        row_start = max(0, row - patch_size // 2)
        col_start = max(0, col - patch_size // 2)
        row_end = min(ds.height, row + patch_size // 2)
        col_end = min(ds.width, col + patch_size // 2)

        patch = ds.read(1, window=rasterio.windows.Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start
        ))

        # Filter out nodata
        nodata = ds.nodata
        if nodata is not None:
            valid = patch[patch != nodata]
        else:
            valid = patch[np.isfinite(patch)]

        ds.close()

        return JSONResponse(content={
            "product_id": product_id,
            "lat": lat,
            "lon": lon,
            "elevation_m": float(elevation) if np.isfinite(elevation) else None,
            "patch_stats": {
                "min_m": float(np.min(valid)) if len(valid) > 0 else None,
                "max_m": float(np.max(valid)) if len(valid) > 0 else None,
                "mean_m": float(np.mean(valid)) if len(valid) > 0 else None,
                "std_m": float(np.std(valid)) if len(valid) > 0 else None,
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
