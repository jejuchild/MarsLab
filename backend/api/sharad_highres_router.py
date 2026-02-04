"""
SHARAD High-Res RDR parser and API router.

Parses PDS3 binary RDR (.dat + .lbl) from the Italian (ASI/Sapienza) pipeline.
Extracts radargram (power image), geometry, and provides surface picking.

Supports multiple products — each endpoint accepts a `product_id` parameter.

Format reference: RDR.FMT from PDS Geosciences Node
  - 5822 bytes/row, 102 columns
  - ECHO_SAMPLES_REAL:      offset 194, 667 × float32
  - ECHO_SAMPLES_IMAGINARY: offset 2862, 667 × float32
  - SUB_SC_EAST_LONGITUDE:  offset 5637, float64
  - SUB_SC_PLANETOCENTRIC_LATITUDE: offset 5645, float64
  - SPACECRAFT_ALTITUDE:    offset 5629, float64
  - RECEIVE_WINDOW_OPENING_TIME: offset 186, float32
"""

import os
import re
import struct
import io
import math
import glob as globmod

import numpy as np
from PIL import Image
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response, JSONResponse

router = APIRouter(prefix="/api/sharad_highres", tags=["SHARAD High-Res"])

# ── Paths ──────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARAD_HR_DIR = os.path.join(_BACKEND_DIR, "sharad_highres")

# ── RDR binary layout constants ────────────────────────
ROW_BYTES = 5822
N_RANGE_BINS = 667

# Byte offsets (0-indexed) and sizes for fields we need
ECHO_REAL_OFFSET = 194      # start_byte 195
ECHO_IMAG_OFFSET = 2862     # start_byte 2863
ECHO_BYTES = N_RANGE_BINS * 4  # 2668

RWOT_OFFSET = 186            # RECEIVE_WINDOW_OPENING_TIME, float32
LON_OFFSET = 5637            # SUB_SC_EAST_LONGITUDE, float64
LAT_OFFSET = 5645            # SUB_SC_PLANETOCENTRIC_LATITUDE, float64
ALT_OFFSET = 5629            # SPACECRAFT_ALTITUDE, float64

# SHARAD timing: 1/20 MHz sample interval → 0.0375 µs per range bin
SHARAD_SAMPLE_INTERVAL_US = 1.0 / 20.0 * 0.75  # ~0.0375 µs
SPEED_OF_LIGHT = 299792458.0  # m/s


# ── Per-product cache management ──────────────────────
# Caches keyed by product_id (uppercase)
_product_caches: dict[str, dict] = {}


def _resolve_product(product_id: str) -> dict:
    """
    Resolve a product_id to file paths and metadata.
    Searches SHARAD_HR_DIR for matching .dat and .lbl files.
    Returns dict with 'dat_path', 'lbl_path', 'total_rows', 'cache_dir'.
    """
    pid_lower = product_id.lower()

    # Try direct filename match
    dat_path = os.path.join(SHARAD_HR_DIR, f"{pid_lower}.dat")
    lbl_path = os.path.join(SHARAD_HR_DIR, f"{pid_lower}.lbl")

    if not os.path.exists(dat_path):
        # Search for files matching the product_id pattern
        pattern = os.path.join(SHARAD_HR_DIR, f"*{pid_lower}*")
        dat_matches = [f for f in globmod.glob(pattern) if f.lower().endswith(".dat")]
        if dat_matches:
            dat_path = dat_matches[0]
            lbl_path = dat_path.rsplit(".", 1)[0] + ".lbl"
        else:
            raise FileNotFoundError(f"No .dat file found for product_id: {product_id}")

    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"SHARAD RDR .dat not found: {dat_path}")

    # Parse total rows from .lbl or compute from file size
    total_rows = _parse_total_rows(lbl_path, dat_path)

    # Per-product cache directory
    cache_dir = os.path.join(SHARAD_HR_DIR, ".cache", product_id.upper())
    os.makedirs(cache_dir, exist_ok=True)

    return {
        "dat_path": dat_path,
        "lbl_path": lbl_path,
        "total_rows": total_rows,
        "cache_dir": cache_dir,
    }


def _parse_total_rows(lbl_path: str, dat_path: str) -> int:
    """Parse FILE_RECORDS from .lbl, or compute from file size."""
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                m = re.match(r"\s*FILE_RECORDS\s*=\s*(\d+)", line)
                if m:
                    return int(m.group(1))

    # Fallback: compute from file size
    file_size = os.path.getsize(dat_path)
    return file_size // ROW_BYTES


def _get_product_data(product_id: str) -> dict:
    """
    Get or create cached data for a product.
    Returns dict with 'power', 'geometry', 'surface', 'total_rows'.
    """
    pid = product_id.upper()
    if pid in _product_caches:
        return _product_caches[pid]

    info = _resolve_product(product_id)
    cache = {
        "info": info,
        "power": None,
        "geometry": None,
        "surface": None,
        "total_rows": info["total_rows"],
    }
    _product_caches[pid] = cache
    return cache


def _cache_path(cache: dict, name: str) -> str:
    return os.path.join(cache["info"]["cache_dir"], name)


def _parse_and_cache(cache: dict):
    """Parse the full .dat file and cache power + geometry arrays."""
    info = cache["info"]
    total_rows = info["total_rows"]
    dat_path = info["dat_path"]

    power_path = _cache_path(cache, "power.npy")
    geom_path = _cache_path(cache, "geometry.npz")

    # Return from disk cache if available
    if os.path.exists(power_path) and os.path.exists(geom_path):
        cache["power"] = np.load(power_path, mmap_mode="r")
        g = np.load(geom_path)
        cache["geometry"] = {"lat": g["lat"], "lon": g["lon"], "alt": g["alt"]}
        return

    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"SHARAD RDR .dat not found: {dat_path}")

    # Allocate arrays
    power = np.empty((total_rows, N_RANGE_BINS), dtype=np.float32)
    lats = np.empty(total_rows, dtype=np.float64)
    lons = np.empty(total_rows, dtype=np.float64)
    alts = np.empty(total_rows, dtype=np.float64)

    with open(dat_path, "rb") as f:
        for i in range(total_rows):
            row = f.read(ROW_BYTES)
            if len(row) < ROW_BYTES:
                power[i:] = 0
                break

            real = np.frombuffer(row, dtype="<f4", offset=ECHO_REAL_OFFSET, count=N_RANGE_BINS)
            imag = np.frombuffer(row, dtype="<f4", offset=ECHO_IMAG_OFFSET, count=N_RANGE_BINS)
            power[i] = real ** 2 + imag ** 2

            lons[i] = struct.unpack_from("<d", row, LON_OFFSET)[0]
            lats[i] = struct.unpack_from("<d", row, LAT_OFFSET)[0]
            alts[i] = struct.unpack_from("<d", row, ALT_OFFSET)[0]

    np.save(power_path, power)
    np.savez_compressed(geom_path, lat=lats, lon=lons, alt=alts)

    cache["power"] = np.load(power_path, mmap_mode="r")
    cache["geometry"] = {"lat": lats, "lon": lons, "alt": alts}


def _get_power(product_id: str) -> tuple[np.ndarray, int]:
    """Return (power array, total_rows) for given product."""
    cache = _get_product_data(product_id)
    if cache["power"] is None:
        _parse_and_cache(cache)
    return cache["power"], cache["total_rows"]


def _get_geometry(product_id: str) -> tuple[dict, int]:
    """Return (geometry dict, total_rows) for given product."""
    cache = _get_product_data(product_id)
    if cache["geometry"] is None:
        _parse_and_cache(cache)
    return cache["geometry"], cache["total_rows"]


def _lon_to_180(lon: np.ndarray) -> np.ndarray:
    """Convert 0..360 East-positive longitude to -180..180."""
    return np.where(lon > 180, lon - 360, lon)


# ── Surface picking ───────────────────────────────────
def _pick_surface(
    product_id: str,
    power: np.ndarray,
    min_offset: int = 15,
    max_offset: int = 250,
    snr_threshold: float = 3.0,
    smooth_kernel: int = 31,
) -> np.ndarray:
    """
    Pick the surface return per trace.

    The surface is the strongest peak in a search band below a coarse
    upper-window argmax.  Uses SNR gating and outlier rejection for
    lateral continuity while allowing gaps where no clear surface exists.

    Returns: 1-D int32 array [n_traces] with surface delay-bin index
             (-1 where no valid surface detected).
    """
    cache = _get_product_data(product_id)
    if cache["surface"] is not None:
        return cache["surface"]

    cp = _cache_path(cache, "surface_v3.npy")
    if os.path.exists(cp):
        cache["surface"] = np.load(cp)
        return cache["surface"]

    n_traces, n_bins = power.shape

    # Step 1 — coarse anchor: argmax in upper 120 bins
    coarse = np.argmax(power[:, :120], axis=1).astype(np.int32)

    # Step 2 — refine: strongest peak in band [coarse+min_offset, coarse+max_offset]
    surface = np.full(n_traces, -1, dtype=np.int32)
    surface_snr = np.zeros(n_traces, dtype=np.float32)

    for i in range(n_traces):
        c = int(coarse[i])
        lo = min(c + min_offset, n_bins - 1)
        hi = min(c + max_offset, n_bins)
        if hi <= lo:
            continue

        band = power[i, lo:hi].copy()
        noise = np.median(band) + 1e-12
        peak_idx = int(np.argmax(band))
        peak_val = band[peak_idx]
        snr = peak_val / noise

        if snr >= snr_threshold:
            surface[i] = lo + peak_idx
            surface_snr[i] = snr

    # Step 3 — outlier rejection (no smoothing of valid picks)
    valid_mask = surface >= 0
    if valid_mask.sum() >= 10:
        from scipy.ndimage import median_filter as mf

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > smooth_kernel:
            y_valid = surface[valid_mask].astype(np.float64)
            y_filtered = mf(y_valid, size=smooth_kernel)
            diff = np.abs(y_valid - y_filtered)
            outlier = diff > 20
            surface[valid_indices[outlier]] = -1
            surface_snr[valid_indices[outlier]] = 0

    np.save(cp, surface)
    cache["surface"] = surface
    return surface


# ── Radargram rendering ───────────────────────────────
def _render_radargram_png(
    power: np.ndarray,
    downsample: int = 1,
    use_log: bool = True,
    pmin: float = 1.0,
    pmax: float = 99.0,
) -> bytes:
    """Render the radargram as a grayscale PNG."""
    if downsample > 1:
        n = power.shape[0] // downsample * downsample
        p = power[:n].reshape(-1, downsample, power.shape[1]).mean(axis=1)
    else:
        p = power.copy() if not np.isfortran(power) else np.ascontiguousarray(power)

    if use_log:
        p = np.log10(np.maximum(p, 1e-12))

    vmin = np.percentile(p, pmin)
    vmax = np.percentile(p, pmax)
    if vmax <= vmin:
        vmax = vmin + 1

    img = ((p - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
    img = img.T

    pil_img = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# ── List available products ───────────────────────────
@router.get("/products")
async def list_products():
    """List available SHARAD High-Res RDR products."""
    products = []
    for lbl_file in sorted(globmod.glob(os.path.join(SHARAD_HR_DIR, "*.lbl"))):
        pid = None
        orbit = None
        with open(lbl_file, "r") as f:
            for line in f:
                m = re.match(r'\s*PRODUCT_ID\s*=\s*"?([^"\s]+)"?', line)
                if m:
                    pid = m.group(1)
                m2 = re.match(r"\s*ORBIT_NUMBER\s*=\s*(\d+)", line)
                if m2:
                    orbit = int(m2.group(1))
        if pid:
            dat_path = lbl_file.rsplit(".", 1)[0] + ".dat"
            products.append({
                "product_id": pid,
                "orbit_number": orbit,
                "has_dat": os.path.exists(dat_path),
            })
    return JSONResponse(content={"products": products, "count": len(products)})


# ── API Endpoints ──────────────────────────────────────

@router.get("/metadata")
async def get_metadata(
    product_id: str = Query(..., description="Product ID (e.g. R_5663601_001_SS19_700_A)"),
):
    """Return dataset metadata."""
    try:
        geom, total_rows = _get_geometry(product_id)
        lons180 = _lon_to_180(geom["lon"])
        return JSONResponse(content={
            "product_id": product_id.upper(),
            "instrument": "SHARAD_HIGHRES",
            "rows": total_rows,
            "range_bins": N_RANGE_BINS,
            "row_bytes": ROW_BYTES,
            "sample_interval_us": SHARAD_SAMPLE_INTERVAL_US,
            "lat_range": [float(geom["lat"].min()), float(geom["lat"].max())],
            "lon_range": [float(lons180.min()), float(lons180.max())],
            "alt_range_km": [float(geom["alt"].min()), float(geom["alt"].max())],
            "display": {
                "recommended_log": True,
                "recommended_downsample": 50,
                "x_axis": "along-track index",
                "y_axis": "range bin (delay)",
            },
        })
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/radargram")
async def get_radargram(
    product_id: str = Query(..., description="Product ID"),
    downsample: int = Query(50, ge=1, le=500, description="Along-track downsample factor"),
    log: int = Query(1, ge=0, le=1, description="Log10 scale (1=yes, 0=no)"),
    pmin: float = Query(1.0, ge=0, le=50, description="Lower percentile clip"),
    pmax: float = Query(99.0, ge=50, le=100, description="Upper percentile clip"),
):
    """Return radargram as PNG image."""
    try:
        power, _ = _get_power(product_id)
        png = _render_radargram_png(power, downsample=downsample, use_log=bool(log),
                                    pmin=pmin, pmax=pmax)
        n_traces_out = power.shape[0] // downsample
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "X-Traces": str(n_traces_out),
                "X-Range-Bins": str(N_RANGE_BINS),
                "X-Downsample": str(downsample),
            },
        )
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/radargram_meta")
async def get_radargram_meta(
    product_id: str = Query(..., description="Product ID"),
    downsample: int = Query(50, ge=1, le=500),
):
    """Return radargram axis info (for overlaying coordinates)."""
    try:
        geom, total_rows = _get_geometry(product_id)
        n_traces_out = total_rows // downsample

        indices = np.arange(0, total_rows, downsample)[:n_traces_out]
        lats = geom["lat"][indices]
        lons = _lon_to_180(geom["lon"][indices])

        return JSONResponse(content={
            "n_traces": int(n_traces_out),
            "n_bins": N_RANGE_BINS,
            "downsample": downsample,
            "sample_interval_us": SHARAD_SAMPLE_INTERVAL_US,
            "lats": [round(float(v), 4) for v in lats],
            "lons": [round(float(v), 4) for v in lons],
        })
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/surface")
async def get_surface(
    product_id: str = Query(..., description="Product ID"),
    downsample: int = Query(50, ge=1, le=500),
):
    """Return auto-picked surface line as polyline in radargram coordinates."""
    try:
        power, total_rows = _get_power(product_id)
        surface = _pick_surface(product_id, power)

        n = total_rows // downsample
        indices = np.arange(0, total_rows, downsample)[:n]
        surface_ds = surface[indices]

        # Only include traces where surface was detected
        points = []
        for i in range(n):
            if surface_ds[i] >= 0:
                points.append({"x": int(i), "y": int(surface_ds[i])})

        return JSONResponse(content={
            "surface": points,
            "downsample": downsample,
            "total_traces": n,
            "detected_traces": len(points),
            "method": "peak detection + outlier rejection",
        })
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/depth_conversion")
async def depth_conversion(
    product_id: str = Query(..., description="Product ID"),
    trace_idx: int = Query(..., ge=0, description="Trace index (downsampled)"),
    cursor_bin: int = Query(..., ge=0, description="Cursor range-bin position"),
    downsample: int = Query(50, ge=1, le=500),
    epsilon_r1: float = Query(3.1, gt=0, le=20, description="Layer 1 dielectric constant"),
    epsilon_r2: float = Query(3.1, gt=0, le=20, description="Layer 2 dielectric constant"),
    boundary_m: float = Query(0.0, ge=0, le=5000, description="Boundary depth between layers (m)"),
    surface_bin_override: int = Query(None, description="Manual surface bin (overrides auto-pick)"),
):
    """
    Convert delay between surface and a cursor position to depth.

    The surface line is used as the reference baseline.
    Δt is computed from surface bin to cursor_bin.
    If surface_bin_override is provided, it replaces the auto-picked surface
    (used when the user has manually adjusted the surface line).

    Piecewise 2-layer dielectric model:
      Layer 1: surface → boundary_m  with εr₁
      Layer 2: boundary_m → below    with εr₂

    If boundary_m == 0 or epsilon_r1 == epsilon_r2, uses uniform εr₁.
    """
    try:
        power, total_rows = _get_power(product_id)

        full_idx = trace_idx * downsample
        if full_idx >= total_rows:
            return JSONResponse(content={"error": "Trace index out of range"}, status_code=400)

        if surface_bin_override is not None:
            s_bin = surface_bin_override
        else:
            surface = _pick_surface(product_id, power)
            s_bin = int(surface[full_idx])

        if s_bin < 0:
            return JSONResponse(content={
                "trace_idx": trace_idx,
                "surface_bin": None,
                "depth_m": None,
                "message": "No surface detected at this trace.",
            })

        if cursor_bin <= s_bin:
            return JSONResponse(content={
                "trace_idx": trace_idx,
                "surface_bin": s_bin,
                "cursor_bin": cursor_bin,
                "depth_m": 0.0,
                "message": "Cursor is at or above the surface.",
            })

        delta_bins = cursor_bin - s_bin
        total_dt_s = delta_bins * SHARAD_SAMPLE_INTERVAL_US * 1e-6  # two-way travel time

        # Piecewise depth computation
        if boundary_m <= 0 or epsilon_r1 == epsilon_r2:
            v = SPEED_OF_LIGHT / np.sqrt(epsilon_r1)
            depth_m = v * total_dt_s / 2.0
        else:
            v1 = SPEED_OF_LIGHT / np.sqrt(epsilon_r1)
            v2 = SPEED_OF_LIGHT / np.sqrt(epsilon_r2)
            twt_layer1 = 2.0 * boundary_m / v1

            if total_dt_s <= twt_layer1:
                depth_m = v1 * total_dt_s / 2.0
            else:
                remaining_dt = total_dt_s - twt_layer1
                depth_in_layer2 = v2 * remaining_dt / 2.0
                depth_m = boundary_m + depth_in_layer2

        geom, _ = _get_geometry(product_id)
        lons180 = _lon_to_180(geom["lon"])
        return JSONResponse(content={
            "trace_idx": trace_idx,
            "full_trace_idx": int(full_idx),
            "surface_bin": s_bin,
            "cursor_bin": cursor_bin,
            "delta_bins": int(delta_bins),
            "delta_t_us": round(delta_bins * SHARAD_SAMPLE_INTERVAL_US, 4),
            "epsilon_r1": epsilon_r1,
            "epsilon_r2": epsilon_r2,
            "boundary_m": boundary_m,
            "depth_m": round(float(depth_m), 1),
            "lat": round(float(geom["lat"][full_idx]), 4),
            "lon": round(float(lons180[full_idx]), 4),
        })
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── Ground track coordinates ───────────────────────────
@router.get("/track")
async def get_track(
    product_id: str = Query(..., description="Product ID"),
    downsample: int = Query(50, ge=1, le=500),
):
    """Return ground-track lat/lon arrays (for Cesium polyline)."""
    try:
        geom, total_rows = _get_geometry(product_id)
        n = total_rows // downsample
        indices = np.arange(0, total_rows, downsample)[:n]
        lats = geom["lat"][indices]
        lons = _lon_to_180(geom["lon"][indices])

        return JSONResponse(content={
            "lats": [round(float(v), 4) for v in lats],
            "lons": [round(float(v), 4) for v in lons],
            "n_traces": n,
            "downsample": downsample,
        })
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── MOLA elevation profile along SHARAD track ─────────
MARS_MEAN_RADIUS = (2 * 3_396_190.0 + 3_376_200.0) / 3  # metres

_DEM_PATH = os.path.join(
    os.path.dirname(_BACKEND_DIR),  # project root
    "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
)
_dem_ds = None


def _get_dem():
    """Open (and cache) the MOLA DEM dataset."""
    global _dem_ds
    if _dem_ds is None:
        import rasterio
        if not os.path.exists(_DEM_PATH):
            raise FileNotFoundError(f"DEM file not found: {_DEM_PATH}")
        _dem_ds = rasterio.open(_DEM_PATH)
    return _dem_ds


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on Mars (metres)."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return MARS_MEAN_RADIUS * 2 * math.asin(math.sqrt(min(a, 1.0)))


@router.get("/mola_profile")
async def get_mola_profile(
    product_id: str = Query(..., description="Product ID"),
    downsample: int = Query(50, ge=1, le=500),
):
    """
    Return MOLA elevation profile sampled at each downsampled SHARAD trace.

    Returns arrays aligned with the radargram X-axis:
      - distance_km: cumulative along-track distance
      - elevation_m: MOLA elevation at each trace
      - lats / lons: coordinates of each sample (in -180..180 lon)
    """
    try:
        from rasterio.windows import Window

        geom, total_rows = _get_geometry(product_id)
        ds = _get_dem()

        n_traces = total_rows // downsample
        indices = np.arange(0, total_rows, downsample)[:n_traces]
        lats = geom["lat"][indices]
        # Convert 0-360 → -180..180 to match DEM coordinate system
        lons = _lon_to_180(geom["lon"][indices])

        # Convert lat/lon to pixel coordinates
        inv_transform = ~ds.transform
        cols, rows = inv_transform * (lons, lats)
        cols = np.round(cols).astype(int)
        rows = np.round(rows).astype(int)
        cols = np.clip(cols, 0, ds.width - 1)
        rows = np.clip(rows, 0, ds.height - 1)

        # Read a minimal bounding window
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        window = Window(col_min, row_min, col_max - col_min + 1, row_max - row_min + 1)
        elev_block = ds.read(1, window=window).astype(np.float64)

        if ds.nodata is not None:
            elev_block[elev_block == ds.nodata] = np.nan

        local_rows = rows - row_min
        local_cols = cols - col_min
        elevations = elev_block[local_rows, local_cols]

        # Cumulative along-track distance
        distances_m = np.zeros(n_traces)
        for i in range(1, n_traces):
            distances_m[i] = distances_m[i - 1] + _haversine(
                float(lats[i - 1]), float(lons[i - 1]),
                float(lats[i]), float(lons[i]),
            )

        return JSONResponse(content={
            "distance_km": [round(d / 1000.0, 3) for d in distances_m],
            "elevation_m": [round(float(e), 1) if not np.isnan(e) else None for e in elevations],
            "lats": [round(float(v), 4) for v in lats],
            "lons": [round(float(v), 4) for v in lons],
            "n_traces": n_traces,
            "downsample": downsample,
            "total_distance_km": round(float(distances_m[-1] / 1000.0), 3) if n_traces > 0 else 0,
        })
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
