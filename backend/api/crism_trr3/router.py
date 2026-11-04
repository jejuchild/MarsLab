"""
CRISM TRR3 — Spectrum + RGB endpoints.

Serves spectrum data and RGB band composites for TRR3 observations
stored in backend/mineral_cnn_data/.
"""

import re
import io

import numpy as np
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from .loader import (
    get_cube_memmap, get_shape, get_wavelengths, get_pixel_spectrum,
    UnsupportedCubeError, require_supported_bands,
)
from api.crism.rgb import nearest_band
from api.mineral_cnn.constants import FILL_VALUE
from api.validation import validate_product_id

router = APIRouter(prefix="/api/crism-trr3", tags=["CRISM TRR3"])

_OBS_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


def _validate_obs_id(obs_id: str):
    if not _OBS_ID_PATTERN.match(obs_id):
        raise HTTPException(status_code=400, detail="Invalid observation ID")
    try:
        validate_product_id(obs_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid observation ID")


class SpectrumRequest(BaseModel):
    line: int
    sample: int


class RGBRequest(BaseModel):
    r_um: float = 2.53
    g_um: float = 1.51
    b_um: float = 1.08
    vmin: float = 0.02
    vmax: float = 0.25


@router.post("/{obs_id}/spectrum")
def spectrum(obs_id: str, req: SpectrumRequest):
    """Get spectrum at a pixel location."""
    _validate_obs_id(obs_id)
    try:
        wavelengths, values = get_pixel_spectrum(obs_id, req.line, req.sample)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except UnsupportedCubeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build valid mask (non-fill, finite)
    valid = np.isfinite(values) & (values < FILL_VALUE - 1) & (values >= 0)

    return {
        "wavelengths": wavelengths[valid].tolist(),
        "reflectance": values[valid].tolist(),
        "valid_bands": int(valid.sum()),
        "total_bands": len(values),
        "line": req.line,
        "sample": req.sample,
    }


@router.post("/{obs_id}/rgb")
def rgb(obs_id: str, req: RGBRequest):
    """Generate RGB composite PNG from three wavelengths."""
    _validate_obs_id(obs_id)
    try:
        mm, rows, cols, bands = get_cube_memmap(obs_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        require_supported_bands(bands)
    except UnsupportedCubeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if req.vmax <= req.vmin:
        raise HTTPException(status_code=400, detail="vmax must be greater than vmin")

    wavelengths = get_wavelengths()
    if len(wavelengths) != bands:
        from api.mineral_cnn.constants import BANDS as IR_WL, IR_START
        wavelengths = IR_WL

    # Only extract the 3 needed bands (not full cube — saves ~1 GB RAM)
    r_idx = nearest_band(wavelengths, req.r_um)
    g_idx = nearest_band(wavelengths, req.g_um)
    b_idx = nearest_band(wavelengths, req.b_um)

    # Offset for IR-only case
    offset = IR_START if len(wavelengths) != bands else 0

    # memmap shape: (rows, bands, cols) — extract 3 bands as (rows, cols) each
    def extract_band(band_idx):
        band = np.array(mm[:, offset + band_idx, :], dtype=np.float32)
        band[band >= FILL_VALUE - 1] = np.nan
        band[~np.isfinite(band)] = np.nan
        band[band < 0] = np.nan
        return band

    R = extract_band(r_idx)
    G = extract_band(g_idx)
    B = extract_band(b_idx)

    # Create mask from G band
    mask = np.isfinite(G)

    def stretch(img):
        out = img.copy()
        out[~mask] = 0
        out = np.clip(out, req.vmin, req.vmax)
        return (out - req.vmin) / (req.vmax - req.vmin)

    rgb_img = np.dstack([stretch(R), stretch(G), stretch(B)])

    # Convert to RGBA PNG with transparent background
    from PIL import Image
    rgb_uint8 = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb_uint8, alpha])

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")


@router.get("/{obs_id}/wavelengths")
def wavelengths(obs_id: str):
    """Return full wavelength list for this observation."""
    _validate_obs_id(obs_id)
    try:
        _, _, _, bands = get_cube_memmap(obs_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        require_supported_bands(bands)
    except UnsupportedCubeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    wl = get_wavelengths()
    if len(wl) != bands:
        from api.mineral_cnn.constants import BANDS as IR_WL
        wl = IR_WL

    return {"wavelengths": wl.tolist(), "count": len(wl)}


@lru_cache(maxsize=8)
def _pixel_locator(obs_id: str):
    from api.mineral_cnn.data_loader import resolve_trr_files
    from api.mineral_cnn.georef import PixelLocator, load_ddr_latlon

    files = resolve_trr_files(obs_id)
    if not files.get("ddr_img") or not files.get("ddr_lbl"):
        raise FileNotFoundError(f"DDR not found for {obs_id}")
    return PixelLocator(load_ddr_latlon(files["ddr_img"], files["ddr_lbl"]))


@router.get("/{obs_id}/pixel")
def pixel_at(obs_id: str, lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-360, le=360)):
    """
    Sensor (line, sample) of the pixel nearest to a lat/lon, from the DDR.

    TRR3 cubes are in sensor geometry (rows/samples are not north-up), so a
    linear lat/lon→pixel mapping over the bounding box picks the wrong pixel.
    `inside` is false when the point is outside the observation's footprint.
    """
    _validate_obs_id(obs_id)
    try:
        return _pixel_locator(obs_id).locate(lat, lon)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/{obs_id}/shape")
def shape(obs_id: str):
    """Return cube dimensions {rows, cols, bands}."""
    _validate_obs_id(obs_id)
    try:
        return get_shape(obs_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
