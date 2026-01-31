# api/hirise_pixel.py
from fastapi import APIRouter, HTTPException
import os
from functools import lru_cache

import rasterio
from rasterio.transform import rowcol
from rasterio.errors import RasterioIOError
import numpy as np

router = APIRouter()

# ======================================================
# Config
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRISE_DATA_DIR = os.path.join(BASE_DIR, "hirise_data")

# ======================================================
# Utils
# ======================================================
def hirise_tif_path(product_id: str) -> str:
    """
    ESP_XXXXX_YYYY → hirise_data/ESP_XXXXX_YYYY_RED.tif
    """
    return os.path.join(HIRISE_DATA_DIR, f"{product_id}_RED.tif")


@lru_cache(maxsize=16)
def open_dataset(path: str) -> rasterio.DatasetReader:
    return rasterio.open(path)


def xy_to_pixel(
    ds: rasterio.DatasetReader,
    x: float,
    y: float,
) -> tuple[int, int]:
    """
    Projected (x, y) [meters] → (row, col)
    """
    try:
        row, col = rowcol(ds.transform, x, y)
        return int(row), int(col)
    except Exception as e:
        raise ValueError(f"xy to pixel failed: {e}")


# ======================================================
# API
# ======================================================
@router.get("/pixel_xy")
def inspect_pixel_xy(
    productId: str,
    x: float,
    y: float,
):
    """
    HiRISE pixel inspection (projected-native)
    """

    tif_path = hirise_tif_path(productId)

    if not os.path.exists(tif_path):
        raise HTTPException(
            status_code=404,
            detail=f"HiRISE TIF not found: {productId}_RED.tif",
        )

    try:
        ds = open_dataset(tif_path)
    except RasterioIOError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        row, col = xy_to_pixel(ds, x, y)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not (0 <= row < ds.height and 0 <= col < ds.width):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Requested x/y is outside image bounds",
                "projected_xy": {"x": x, "y": y},
                "pixel": {"row": row, "col": col},
                "image": {
                    "width": ds.width,
                    "height": ds.height,
                    "bounds": ds.bounds,
                    "crs": str(ds.crs),
                },
            },
        )

    try:
        dn = ds.read(
            1,
            window=((row, row + 1), (col, col + 1)),
        )[0, 0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pixel read failed: {e}")

    return {
        "productId": productId,
        "projected_xy": {"x": x, "y": y},
        "pixel": {
            "row": row,
            "col": col,
            "dn": int(dn),
        },
        "image": {
            "width": ds.width,
            "height": ds.height,
            "crs": str(ds.crs),
        },
    }


@router.get("/window_xy")
def inspect_window_xy(
    productId: str,
    x: float,
    y: float,
    halfSize: int = 32,
):
    """
    HiRISE DN window inspection
    """

    tif_path = hirise_tif_path(productId)

    if not os.path.exists(tif_path):
        raise HTTPException(
            status_code=404,
            detail=f"HiRISE TIF not found: {productId}_RED.tif",
        )

    ds = open_dataset(tif_path)

    try:
        row, col = xy_to_pixel(ds, x, y)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    r0 = max(0, row - halfSize)
    r1 = min(ds.height, row + halfSize)
    c0 = max(0, col - halfSize)
    c1 = min(ds.width, col + halfSize)

    if r0 >= r1 or c0 >= c1:
        raise HTTPException(
            status_code=400,
            detail="Invalid window (outside image)",
        )

    try:
        window = ds.read(
            1,
            window=((r0, r1), (c0, c1)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Window read failed: {e}")

    # ===== 중심 픽셀 DN (🔥 추가) =====
    try:
        center_dn = ds.read(
            1,
            window=((row, row + 1), (col, col + 1)),
        )[0, 0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Center pixel read failed: {e}",
        )

    # nodata / invalid 제거
    values = window.astype(np.float32).flatten()
    nodata = ds.nodata

    if nodata is not None:
        values = values[values != nodata]

    values = values[np.isfinite(values)]

    if values.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Window contains no valid pixels",
        )

    hist, _ = np.histogram(values, bins=256)

    mean = float(values.mean())
    std = float(values.std())

    # ===== z-score (🔥 추가) =====
    zscore = (
        float((center_dn - mean) / std)
        if std > 0
        else None
    )

    return {
        "productId": productId,
        "projected_xy": {"x": x, "y": y},

        # 🔥 중심 픽셀 정보
        "center": {
            "row": row,
            "col": col,
            "dn": int(center_dn),
            "zscore": zscore,
        },

        "window": {
            "row_range": [r0, r1],
            "col_range": [c0, c1],
            "size": [r1 - r0, c1 - c0],
        },
        "stats": {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": mean,
            "median": float(np.median(values)),
            "std": std,
            "count": int(values.size),
        },
        "histogram": {
            "bins": 256,
            "counts": hist.tolist(),
        },
    }
