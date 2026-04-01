"""Texture draping: project Mastcam-Z imagery onto HiRISE DTM grid.

Takes Mastcam-Z lon/lat per pixel and resamples the texture onto the
HiRISE DTM grid, producing a GeoTIFF with the Mastcam-Z colors draped
on the DTM footprint.
"""

from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from PIL import Image

from .config import OUTPUT_DIR

# Mars 2000 geographic CRS in WKT (avoids PROJ4 DB version issues)
MARS_CRS_WKT = (
    'GEOGCRS["Mars 2000",'
    'DATUM["D_Mars_2000",'
    'ELLIPSOID["Mars_2000_IAU_IAG",3396190,169.89444722361179]],'
    'PRIMEM["Reference_Meridian",0],'
    'CS[ellipsoidal,2],'
    'AXIS["Latitude",north,ORDER[1]],'
    'AXIS["Longitude",east,ORDER[2]],'
    'UNIT["degree",0.0174532925199433]]'
)


def _read_pds_img_texture(image_path: Path) -> np.ndarray:
    """Read a PDS3/PDS4 IMG file as RGB array using label metadata."""
    from .mastcam_xyz import parse_pds4_label

    label_path = image_path.with_suffix(".xml")
    if not label_path.exists():
        label_path = image_path.with_suffix(".lbl")

    meta = {}
    if label_path.exists():
        meta = parse_pds4_label(label_path)

    lines = meta.get("lines", 0)
    samples = meta.get("samples", 0)
    bands = meta.get("bands", 3)
    start_byte = meta.get("start_byte", 0)

    if lines == 0 or samples == 0:
        raise ValueError(f"Cannot parse dimensions for {image_path.name}")

    raw = np.fromfile(str(image_path), dtype=">f4", offset=start_byte)
    expected = bands * lines * samples
    if raw.size < expected:
        # Try 2-byte int
        raw = np.fromfile(str(image_path), dtype=">i2", offset=start_byte)
    if raw.size < expected:
        raise ValueError(f"RAS file too small: need {expected}, got {raw.size}")

    raw = raw[:expected].reshape(bands, lines, samples)

    if bands >= 3:
        rgb = np.moveaxis(raw[:3], 0, -1).astype(np.float64)
    else:
        rgb = np.stack([raw[0]] * 3, axis=-1).astype(np.float64)

    # Normalize to uint8
    positive = rgb[rgb > 0]
    if positive.size > 0:
        vmin, vmax = np.nanpercentile(positive, [2, 98])
    else:
        vmin, vmax = np.nanmin(rgb), np.nanmax(rgb)
    denom = vmax - vmin
    if denom == 0:
        denom = 1.0
    return np.clip((rgb - vmin) / denom * 255, 0, 255).astype(np.uint8)


def load_mastcamz_texture(image_path: Path) -> np.ndarray:
    """Load a Mastcam-Z RAS (or calibrated) image as RGB array.

    Supports TIFF, JPEG, PNG, and PDS IMG formats.
    Returns (H, W, 3) uint8 array.
    """
    suffix = image_path.suffix.lower()

    if suffix in (".tif", ".tiff") and HAS_RASTERIO:
        with rasterio.open(image_path) as ds:
            bands = ds.read()
            if bands.shape[0] >= 3:
                rgb = np.moveaxis(bands[:3], 0, -1)
            else:
                rgb = np.stack([bands[0]] * 3, axis=-1)

            if rgb.dtype != np.uint8:
                positive = rgb[rgb > 0]
                if positive.size > 0:
                    vmin, vmax = np.nanpercentile(positive, [2, 98])
                else:
                    vmin, vmax = np.nanmin(rgb), np.nanmax(rgb)
                denom = vmax - vmin
                if denom == 0:
                    denom = 1.0
                rgb = np.clip((rgb - vmin) / denom * 255, 0, 255).astype(np.uint8)
            return rgb

    if suffix == ".img":
        return _read_pds_img_texture(image_path)

    # Fall back to PIL for JPEG/PNG/etc
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def drape_on_dtm(
    lon: np.ndarray,
    lat: np.ndarray,
    texture: np.ndarray,
    dtm,
    output_path: Path = None,
    resolution_m: float = None,
) -> Path:
    """Drape Mastcam-Z texture onto HiRISE DTM grid.

    Args:
        lon: (H, W) longitude array from Mastcam-Z XYZ processing
        lat: (H, W) latitude array from Mastcam-Z XYZ processing
        texture: (H, W, 3) RGB texture to drape (same dimensions as lon/lat)
        dtm: HiRISEDTM instance
        output_path: where to save the GeoTIFF
        resolution_m: output resolution in meters (default: DTM resolution)

    Returns:
        Path to the output GeoTIFF
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for GeoTIFF output. pip install rasterio")

    if output_path is None:
        output_path = OUTPUT_DIR / "mastcamz_on_hirise.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get valid Mastcam-Z coverage
    valid = ~np.isnan(lon) & ~np.isnan(lat)
    if not np.any(valid):
        raise ValueError("No valid Mastcam-Z lon/lat points to drape")

    valid_lon = lon[valid]
    valid_lat = lat[valid]
    valid_rgb = texture[valid]  # (N, 3)

    lon_min, lon_max = valid_lon.min(), valid_lon.max()
    lat_min, lat_max = valid_lat.min(), valid_lat.max()

    print(f"  Mastcam-Z coverage: lon=[{lon_min:.6f}, {lon_max:.6f}], "
          f"lat=[{lat_min:.6f}, {lat_max:.6f}]")
    print(f"  Valid pixels: {valid_rgb.shape[0]}")

    # Determine output grid
    if resolution_m is None:
        resolution_m = dtm.resolution_m

    # Convert resolution from meters to degrees (approximate for Mars)
    mars_r_m = 3389500.0
    res_deg = np.degrees(resolution_m / mars_r_m)

    # Add a small buffer around the coverage
    buf = res_deg * 5
    grid_lon_min = lon_min - buf
    grid_lon_max = lon_max + buf
    grid_lat_min = lat_min - buf
    grid_lat_max = lat_max + buf

    # Grid dimensions
    grid_w = max(1, int(np.ceil((grid_lon_max - grid_lon_min) / res_deg)))
    grid_h = max(1, int(np.ceil((grid_lat_max - grid_lat_min) / res_deg)))

    # Clamp to reasonable size
    max_dim = 20000
    if grid_w > max_dim or grid_h > max_dim:
        scale = max_dim / max(grid_w, grid_h)
        grid_w = int(grid_w * scale)
        grid_h = int(grid_h * scale)
        res_deg = (grid_lon_max - grid_lon_min) / grid_w
        print(f"  Clamped output to {grid_w}x{grid_h}")

    print(f"  Output grid: {grid_w}x{grid_h} ({resolution_m:.2f} m/px)")

    # Create output arrays: RGB + alpha (for transparency where no data)
    out_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    out_alpha = np.zeros((grid_h, grid_w), dtype=np.uint8)
    out_count = np.zeros((grid_h, grid_w), dtype=np.int32)
    out_accum = np.zeros((grid_h, grid_w, 3), dtype=np.float64)

    # Map Mastcam-Z pixels to output grid
    col_idx = ((valid_lon - grid_lon_min) / res_deg).astype(np.int32)
    row_idx = ((grid_lat_max - valid_lat) / res_deg).astype(np.int32)  # lat flipped

    # Clamp to grid bounds
    col_idx = np.clip(col_idx, 0, grid_w - 1)
    row_idx = np.clip(row_idx, 0, grid_h - 1)

    # Accumulate (average where multiple pixels map to same cell)
    np.add.at(out_accum, (row_idx, col_idx), valid_rgb.astype(np.float64))
    np.add.at(out_count, (row_idx, col_idx), 1)

    # Average
    has_data = out_count > 0
    for c in range(3):
        out_rgb[:, :, c][has_data] = (out_accum[:, :, c][has_data] / out_count[has_data]).astype(np.uint8)
    out_alpha[has_data] = 255

    filled_pct = np.sum(has_data) / (grid_w * grid_h) * 100
    print(f"  Grid fill: {filled_pct:.1f}%")

    # Get DTM elevation for the grid
    grid_lons = np.linspace(grid_lon_min, grid_lon_max, grid_w)
    grid_lats = np.linspace(grid_lat_max, grid_lat_min, grid_h)  # top to bottom
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)

    dtm_rows, dtm_cols = dtm.lonlat_to_pixel(
        grid_lon_2d.ravel(), grid_lat_2d.ravel()
    )
    elevation = dtm.get_elevation(dtm_rows, dtm_cols).reshape(grid_h, grid_w)

    # Write GeoTIFF with 4 bands: R, G, B, elevation
    transform = from_bounds(
        grid_lon_min, grid_lat_min, grid_lon_max, grid_lat_max,
        grid_w, grid_h,
    )

    # Mars CRS - use a simple geographic CRS on Mars (IAU 2015 Mars sphere)
    # EPSG-like code for Mars: IAU:49900 or we define a custom WKT
    mars_crs = CRS.from_wkt(MARS_CRS_WKT)

    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=grid_h,
        width=grid_w,
        count=5,  # R, G, B, Alpha, Elevation
        dtype="float32",
        crs=mars_crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(out_rgb[:, :, 0].astype(np.float32), 1)
        dst.write(out_rgb[:, :, 1].astype(np.float32), 2)
        dst.write(out_rgb[:, :, 2].astype(np.float32), 3)
        dst.write(out_alpha.astype(np.float32), 4)
        dst.write(elevation.astype(np.float32), 5)

        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        dst.set_band_description(4, "Alpha")
        dst.set_band_description(5, "Elevation_m")

    print(f"  Saved: {output_path}")
    print(f"  Bands: RGB + Alpha + Elevation")

    # Also save a quick-look PNG (RGB only)
    png_path = output_path.with_suffix(".png")
    rgba = np.dstack([out_rgb, out_alpha])
    Image.fromarray(rgba, "RGBA").save(str(png_path))
    print(f"  Quick-look: {png_path}")

    return output_path


def drape_simple_overlay(
    lon: np.ndarray,
    lat: np.ndarray,
    texture: np.ndarray,
    output_path: Path = None,
) -> Path:
    """Create a simple georeferenced overlay without DTM (fallback).

    Useful when DTM is not available but you still want a georeferenced
    Mastcam-Z image.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required")

    if output_path is None:
        output_path = OUTPUT_DIR / "mastcamz_overlay.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid = ~np.isnan(lon) & ~np.isnan(lat)
    if not np.any(valid):
        raise ValueError("No valid points")

    valid_lon = lon[valid]
    valid_lat = lat[valid]

    lon_min, lon_max = valid_lon.min(), valid_lon.max()
    lat_min, lat_max = valid_lat.min(), valid_lat.max()

    # Guard against zero range
    mars_r_m = 3389500.0
    res_deg = np.degrees(1.0 / mars_r_m)
    if lon_max - lon_min < res_deg:
        lon_min -= res_deg * 50
        lon_max += res_deg * 50
    if lat_max - lat_min < res_deg:
        lat_min -= res_deg * 50
        lat_max += res_deg * 50

    grid_w = max(1, min(10000, int((lon_max - lon_min) / res_deg)))
    grid_h = max(1, min(10000, int((lat_max - lat_min) / res_deg)))

    res_deg_x = (lon_max - lon_min) / grid_w
    res_deg_y = (lat_max - lat_min) / grid_h

    out_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    out_count = np.zeros((grid_h, grid_w), dtype=np.int32)
    out_accum = np.zeros((grid_h, grid_w, 3), dtype=np.float64)

    valid_rgb = texture[valid]
    col_idx = np.clip(((valid_lon - lon_min) / res_deg_x).astype(np.int32), 0, grid_w - 1)
    row_idx = np.clip(((lat_max - valid_lat) / res_deg_y).astype(np.int32), 0, grid_h - 1)

    np.add.at(out_accum, (row_idx, col_idx), valid_rgb.astype(np.float64))
    np.add.at(out_count, (row_idx, col_idx), 1)

    has_data = out_count > 0
    for c in range(3):
        out_rgb[:, :, c][has_data] = (out_accum[:, :, c][has_data] / out_count[has_data]).astype(np.uint8)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, grid_w, grid_h)
    mars_crs = CRS.from_wkt(MARS_CRS_WKT)

    with rasterio.open(
        str(output_path), "w", driver="GTiff",
        height=grid_h, width=grid_w, count=3,
        dtype="uint8", crs=mars_crs, transform=transform, compress="deflate",
    ) as dst:
        for i in range(3):
            dst.write(out_rgb[:, :, i], i + 1)

    print(f"  Simple overlay saved: {output_path}")
    return output_path
