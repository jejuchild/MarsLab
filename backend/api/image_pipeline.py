"""Pure image-processing helpers for overlay / quickview / tile generation.

No FastAPI imports here: functions take arrays / paths / datasets and return
arrays or PNG bytes. HTTP concerns (status codes, caching headers, disk cache
lookups) live in ``api.overlay_router``.

Encoding helpers return ``None`` when OpenCV fails to encode so callers can
raise the exact HTTP error they always raised at that point.
"""
import os
import time
from functools import lru_cache

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

TILE_SIZE = 256
MAX_ZOOM = 8

MARS_A = 3396190.0  # equatorial radius (m)
MARS_B = 3376200.0  # polar radius (m)


class ImagePipelineError(Exception):
    """Raised for expected processing failures; message is the HTTP detail."""


# ======================================================
# Generic helpers
# ======================================================
def blank_png() -> bytes:
    blank = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    ok, png = cv2.imencode(".png", blank)
    return png.tobytes() if ok else b""


@lru_cache(maxsize=32)
def open_ds(path: str):
    return rasterio.open(path)


def ds_world_bounds(ds):
    b = ds.bounds
    return (b.left, b.bottom, b.right, b.top)


def gray_to_rgba(gray: np.ndarray, alpha_threshold: int) -> np.ndarray:
    """Grey → RGBA; pixels <= ``alpha_threshold`` become fully transparent."""
    h, w = gray.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = np.where(gray > alpha_threshold, 255, 0)
    return rgba


def encode_rgba_png(rgba: np.ndarray, params=None) -> bytes | None:
    """Encode an RGBA array as PNG (converted to OpenCV's BGRA order)."""
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    if params is None:
        ok, png = cv2.imencode(".png", bgra)
    else:
        ok, png = cv2.imencode(".png", bgra, params)
    return png.tobytes() if ok else None


def encode_png(img: np.ndarray) -> bytes | None:
    """Encode an array as PNG exactly as given (no channel reordering)."""
    ok, png = cv2.imencode(".png", img)
    return png.tobytes() if ok else None


# ======================================================
# Overlay disk cache (TTL + LRU eviction)
# ======================================================
def cleanup_overlay_cache(cache_dir: str, max_mb: float, ttl_days: float, now: float | None = None):
    """Evict oldest files when cache exceeds size limit or TTL.

    Files older than ``ttl_days`` (by mtime) are removed; if the remaining
    total exceeds ``max_mb``, oldest files are removed until the total is
    at or below 80% of the limit. Subdirectories are ignored.
    """
    if now is None:
        now = time.time()
    ttl_seconds = ttl_days * 86400
    files = []
    total_size = 0
    for fname in os.listdir(cache_dir):
        fpath = os.path.join(cache_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stat = os.stat(fpath)
        # Remove files older than TTL
        if now - stat.st_mtime > ttl_seconds:
            os.remove(fpath)
            continue
        files.append((fpath, stat.st_mtime, stat.st_size))
        total_size += stat.st_size

    # LRU eviction if over size limit
    max_bytes = max_mb * 1024 * 1024
    if total_size > max_bytes:
        files.sort(key=lambda x: x[1])  # oldest first
        while total_size > max_bytes * 0.8 and files:  # evict to 80%
            fpath, _, fsize = files.pop(0)
            os.remove(fpath)
            total_size -= fsize


# ======================================================
# World tiles
# ======================================================
def render_world_tile(ds, extent, z: int, x: int, y: int) -> bytes:
    """Render one 256px world tile from ``ds`` given the union ``extent``."""
    left, bottom, right, top = extent

    scale = 2 ** (MAX_ZOOM - z)
    tile_px = TILE_SIZE * scale

    x0 = left + x * tile_px
    x1 = x0 + tile_px
    y1 = top - y * tile_px
    y0 = y1 - tile_px

    dl, db, dr, dt = ds_world_bounds(ds)
    if x1 <= dl or x0 >= dr or y1 <= db or y0 >= dt:
        return blank_png()

    try:
        win = from_bounds(x0, y0, x1, y1, transform=ds.transform)
        data = ds.read(
            1,
            window=win,
            out_shape=(TILE_SIZE, TILE_SIZE),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
    except Exception:
        return blank_png()

    tile = np.clip(data / 4, 0, 255).astype(np.uint8)
    ok, png = cv2.imencode(".png", tile)
    return png.tobytes() if ok else blank_png()


# ======================================================
# HiRISE overlay / quickview
# ======================================================
def hirise_gray_from_jpg(path: str, max_size: int) -> np.ndarray | None:
    """Read a quickview JPG as grey, upscaling to ``max_size`` if smaller.

    Returns None if the file cannot be decoded.
    """
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    h, w = gray.shape
    scale = max_size / max(w, h)
    if scale > 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return gray


def hirise_gray_from_raster(ds, max_size: int) -> np.ndarray:
    """Downsample band 1 of ``ds`` to ``max_size`` and 1–99% stretch non-zero DNs."""
    scale = max(ds.width, ds.height) / max_size
    if scale < 1:
        scale = 1
    out_w, out_h = int(ds.width / scale), int(ds.height / scale)
    data = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
    nonzero = data[data > 0]
    if nonzero.size > 0:
        vmin, vmax = float(np.percentile(nonzero, 1)), float(np.percentile(nonzero, 99))
        gray = np.clip((data.astype(np.float32) - vmin) / max(vmax - vmin, 1) * 255, 0, 255).astype(np.uint8)
    else:
        gray = np.zeros_like(data, dtype=np.uint8)
    return gray


def read_jp2_browse(jp2_path: str) -> np.ndarray:
    """Read a heavily downsampled (~800px wide) 8-bit browse from a HiRISE JP2."""
    with rasterio.open(jp2_path) as ds:
        full_w, full_h = ds.width, ds.height
        # Read a heavily downsampled version (target ~800px wide)
        scale = max(1, full_w // 800)
        out_w, out_h = full_w // scale, full_h // scale
        arr = ds.read(1, out_shape=(out_h, out_w))
    # Normalize to 0-255
    nonzero = arr[arr > 0]
    if nonzero.size > 0:
        vmin, vmax = float(np.percentile(nonzero, 1)), float(np.percentile(nonzero, 99))
        if vmax > vmin:
            arr = np.clip((arr.astype(float) - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def crop_to_aspect(gray: np.ndarray, rdr_lines: int, rdr_samples: int) -> np.ndarray:
    """Center-crop ``gray`` to the RDR product's width/height ratio (>1% off).

    Browse images (abrowse.jpg) are non-uniformly downscaled from the full
    RDR product and may include annotation padding, causing ~8-9% aspect
    ratio mismatch that leads to misaligned quickview overlays.
    """
    h, w = gray.shape
    target_ratio = rdr_samples / rdr_lines  # width/height
    current_ratio = w / h
    if abs(current_ratio - target_ratio) / target_ratio > 0.01:
        if current_ratio < target_ratio:
            # Image is too tall — crop top/bottom
            new_h = int(w / target_ratio)
            crop = (h - new_h) // 2
            gray = gray[crop:crop + new_h, :]
        else:
            # Image is too wide — crop left/right
            new_w = int(h * target_ratio)
            crop = (w - new_w) // 2
            gray = gray[:, crop:crop + new_w]
    return gray


def polar_stereo_crs(proj_center_lat: float) -> str:
    """PROJ string of the Mars polar stereographic CRS for the given hemisphere."""
    sign = 1 if proj_center_lat > 0 else -1
    return (
        f"+proj=stere +lat_0={sign * 90} +lon_0=0 "
        f"+k=1 +x_0=0 +y_0=0 +a={MARS_A} +b={MARS_B} +units=m +no_defs"
    )


MARS_GEO_CRS = f"+proj=longlat +a={MARS_A} +b={MARS_B} +no_defs"


def reproject_polar_browse(gray: np.ndarray, feature: dict) -> np.ndarray:
    """Reproject a polar stereographic browse image to equirectangular (lat/lon).
    Uses pyproj for coordinate transforms and cv2.remap for fast warping."""
    from pyproj import Transformer

    coords = feature["geometry"]["coordinates"][0]
    proj_center_lat = feature["properties"].get("proj_center_lat", 90.0)

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    stereo_crs = polar_stereo_crs(proj_center_lat)
    geo_crs = MARS_GEO_CRS

    # Source image extent in polar stereographic coordinates
    geo_to_stereo = Transformer.from_crs(geo_crs, stereo_crs, always_xy=True)
    xs, ys = [], []
    for lon, lat in coords:
        x, y = geo_to_stereo.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    src_left, src_right = min(xs), max(xs)
    src_bottom, src_top = min(ys), max(ys)

    h_src, w_src = gray.shape

    # Destination grid in geographic coordinates
    dst_h = max(h_src, 512)
    dst_w = max(w_src, 512)

    # For each destination pixel, compute its (lon, lat) → (x_stereo, y_stereo) → source pixel
    lon_arr = np.linspace(west, east, dst_w, dtype=np.float64)
    lat_arr = np.linspace(north, south, dst_h, dtype=np.float64)  # top=north → bottom=south
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    # Transform geographic → polar stereographic
    x_grid, y_grid = geo_to_stereo.transform(lon_grid.ravel(), lat_grid.ravel())
    x_grid = np.array(x_grid).reshape(dst_h, dst_w)
    y_grid = np.array(y_grid).reshape(dst_h, dst_w)

    # Map stereo coords to source pixel coords
    map_x = ((x_grid - src_left) / (src_right - src_left) * (w_src - 1)).astype(np.float32)
    map_y = ((src_top - y_grid) / (src_top - src_bottom) * (h_src - 1)).astype(np.float32)

    dst = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    return dst


def downscale_to_width(gray: np.ndarray, max_width: int = 1024) -> np.ndarray:
    """Downscale (INTER_AREA) so width <= ``max_width``, keeping aspect."""
    h, w = gray.shape
    if w > max_width:
        scale = max_width / w
        new_w, new_h = max_width, int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray


# ======================================================
# CRISM quickview
# ======================================================
def trr3_quickview_png(trr_img_path: str, meta: dict) -> bytes:
    """Render a VNIR quickview PNG (transparent invalid pixels) from a TRR3 cube.

    ``meta`` is the parsed PDS3 label (needs LINES, LINE_SAMPLES, BANDS).
    Raises ImagePipelineError if the band has no valid data or encoding fails.
    """
    rows = int(meta["LINES"])
    cols = int(meta["LINE_SAMPLES"])
    bands = int(meta["BANDS"])

    # Use memmap to avoid loading full cube into RAM
    mm = np.memmap(trr_img_path, dtype=np.float32, mode="r",
                   shape=(rows, bands, cols))

    # Pick a VNIR band (~60% through bands for good contrast)
    band_idx = int(bands * 0.6)
    single = np.array(mm[:, band_idx, :], dtype=np.float64)

    valid = (single > 0) & (single < 1.5) & np.isfinite(single)
    if valid.sum() < 10:
        raise ImagePipelineError("No valid data in selected band")

    p2 = float(np.percentile(single[valid], 2))
    p98 = float(np.percentile(single[valid], 98))
    if p98 <= p2:
        p98 = p2 + 0.01

    stretched = np.clip((single - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    stretched[~valid] = 0

    # RGBA with transparency for invalid pixels
    h, w = stretched.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = stretched
    rgba[:, :, 1] = stretched
    rgba[:, :, 2] = stretched
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)

    png_bytes = encode_rgba_png(rgba)
    if png_bytes is None:
        raise ImagePipelineError("Failed to encode PNG")
    return png_bytes


def crism_image_to_rgba(img: np.ndarray) -> np.ndarray:
    """Add transparency (grey <= 5 → alpha 0) to a CRISM browse image.

    Input/output are in OpenCV channel order; 4-channel input is returned as-is.
    """
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            # Already RGBA
            rgba = img
        else:
            # RGB - convert and add alpha
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img
            rgba[:, :, 3] = np.where(gray > 5, 255, 0)
    else:
        h, w = img.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = img
        rgba[:, :, 1] = img
        rgba[:, :, 2] = img
        rgba[:, :, 3] = np.where(img > 5, 255, 0)
    return rgba


# ======================================================
# HiRISE DTM / GeoTIFF → PNG
# ======================================================
def read_downsampled(source_path: str, max_size: int):
    """Read band 1 bilinearly downsampled so the long side is <= ``max_size``.

    Returns ``(data, nodata)``.
    """
    ds = rasterio.open(source_path)

    # Calculate downsampling factor
    scale = max(ds.width, ds.height) / max_size
    if scale < 1:
        scale = 1

    out_width = int(ds.width / scale)
    out_height = int(ds.height / scale)

    # Capture nodata before closing
    nodata_val = ds.nodata

    # Read with resampling
    data = ds.read(
        1,
        out_shape=(out_height, out_width),
        resampling=Resampling.bilinear
    )
    ds.close()
    return data, nodata_val


def raster_to_rgba(data: np.ndarray, nodata_val, use_hillshade: bool) -> np.ndarray:
    """Render a single-band raster as an RGBA overlay.

    ``use_hillshade``: treat ``data`` as elevation and shade it (az 315°, alt 45°);
    0 / NaN / nodata (or -32768 if nodata is unset) become transparent.
    Otherwise: 2–98% stretch of positive values; DN <= 0 become transparent.
    """
    out_height, out_width = data.shape
    if use_hillshade:
        # Generate hillshade from elevation data
        nodata_mask = (data == 0) | np.isnan(data.astype(float))
        if nodata_val is not None:
            nodata_mask = nodata_mask | (data == nodata_val)
        else:
            nodata_mask = nodata_mask | (data == -32768)
        # Compute gradient for hillshade
        dy, dx = np.gradient(data.astype(float))
        azimuth_rad = np.radians(315)
        altitude_rad = np.radians(45)
        slope = np.sqrt(dx**2 + dy**2)
        aspect = np.arctan2(-dy, dx)
        shaded = (np.sin(altitude_rad) * np.cos(np.arctan(slope)) +
                  np.cos(altitude_rad) * np.sin(np.arctan(slope)) *
                  np.cos(azimuth_rad - aspect))
        shaded = np.clip(shaded * 255, 0, 255).astype(np.uint8)
        # Create RGBA with transparency for nodata
        rgba = np.zeros((out_height, out_width, 4), dtype=np.uint8)
        rgba[:, :, 0] = shaded
        rgba[:, :, 1] = shaded
        rgba[:, :, 2] = shaded
        rgba[:, :, 3] = np.where(nodata_mask, 0, 255)
    else:
        # Normalize orthoimage to 0-255
        p2, p98 = np.percentile(data[data > 0], [2, 98]) if np.any(data > 0) else (0, 255)
        if p98 > p2:
            data_norm = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        else:
            data_norm = np.clip(data, 0, 255).astype(np.uint8)

        # Create RGBA image with transparency only for nodata (original==0)
        rgba = np.zeros((out_height, out_width, 4), dtype=np.uint8)
        rgba[:, :, 0] = data_norm  # R
        rgba[:, :, 1] = data_norm  # G
        rgba[:, :, 2] = data_norm  # B
        rgba[:, :, 3] = np.where(data > 0, 255, 0)
    return rgba


def geotiff_to_png(source_path: str, max_size: int, use_hillshade: bool = False) -> bytes | None:
    """GeoTIFF/JP2/IMG raster → transparent overlay PNG bytes (None if encoding fails)."""
    data, nodata_val = read_downsampled(source_path, max_size)
    rgba = raster_to_rgba(data, nodata_val, use_hillshade)
    return encode_rgba_png(rgba)
