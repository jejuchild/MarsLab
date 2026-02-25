# api/crism/router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response, JSONResponse
import numpy as np
import io
from PIL import Image

from .resolver import resolve_crism_paths
from .loader import load_cube, load_wavelength
from .rgb import make_rgb, IGNORE_VALUE
from api.validation import validate_product_id
print("🔥 CRISM ROUTER LOADED 🔥")

router = APIRouter(tags=["CRISM"])

class RGBRequest(BaseModel):
    r_um: float
    g_um: float
    b_um: float
    vmin: float = 0.02
    vmax: float = 0.25

@router.post("/{product_id}/rgb")
def crism_rgb(product_id: str, req: RGBRequest):
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    try:
        print("[CRISM RGB] product_id =", product_id)
        hdr, img, wv = resolve_crism_paths(product_id)
        print("[CRISM RGB] hdr =", hdr)
        print("[CRISM RGB] img =", img)
        print("[CRISM RGB] wv  =", wv)
        cube = load_cube(hdr)
        wavelength = load_wavelength(wv)

        rgb = make_rgb(
            cube,
            wavelength,
            req.r_um,
            req.g_um,
            req.b_um,
            req.vmin,
            req.vmax,
        )

        # Convert to uint8
        img8 = (rgb * 255).astype("uint8")

        # Create alpha channel - black pixels become transparent
        # Areas where all RGB channels are 0 should be transparent
        alpha = np.where((img8[:,:,0] == 0) & (img8[:,:,1] == 0) & (img8[:,:,2] == 0), 0, 255).astype("uint8")

        # Stack RGB with alpha to create RGBA
        rgba = np.dstack([img8, alpha])

        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.fromarray(rgba, mode='RGBA').save(buf, format="PNG")
        buf.seek(0)

        return Response(buf.getvalue(), media_type="image/png")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


class SpectrumRequest(BaseModel):
    line: int
    sample: int


@router.post("/{product_id}/spectrum")
def crism_spectrum(product_id: str, req: SpectrumRequest):
    """
    Get the spectrum (reflectance vs wavelength) for a single pixel.
    Returns wavelengths (in micrometers) and reflectance values.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    try:
        print(f"[CRISM Spectrum] product_id={product_id}, line={req.line}, sample={req.sample}")
        hdr, img, wv = resolve_crism_paths(product_id)

        cube = load_cube(hdr)  # (bands, lines, samples)
        wavelengths = load_wavelength(wv)  # (bands,) in micrometers

        n_bands, n_lines, n_samples = cube.shape
        print(f"[CRISM Spectrum] Cube shape: {cube.shape}")

        # Validate coordinates
        if req.line < 0 or req.line >= n_lines:
            raise HTTPException(status_code=400, detail=f"Line {req.line} out of range [0, {n_lines})")
        if req.sample < 0 or req.sample >= n_samples:
            raise HTTPException(status_code=400, detail=f"Sample {req.sample} out of range [0, {n_samples})")

        # Extract spectrum for this pixel
        spectrum = cube[:, req.line, req.sample].copy()

        # Clean up ignore values
        spectrum[spectrum >= IGNORE_VALUE - 1] = np.nan
        spectrum[spectrum < 0] = np.nan

        # Count valid values
        valid_count = np.isfinite(spectrum).sum()
        print(f"[CRISM Spectrum] Valid bands: {valid_count}/{len(spectrum)}")

        # Convert to lists for JSON
        return JSONResponse({
            "product_id": product_id,
            "line": req.line,
            "sample": req.sample,
            "wavelengths": wavelengths.tolist(),
            "reflectance": [float(v) if np.isfinite(v) else None for v in spectrum],
            "n_bands": int(n_bands),
            "valid_bands": int(valid_count),
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
