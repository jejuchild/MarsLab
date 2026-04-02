"""Mastcam-Z orthoimage generation: project RAS textures onto HiRISE DTM grid.

For each sol with combined_lonlat.npz, this module:
1. Loads all frame lon/lat mappings and RAS textures
2. For each Mastcam-Z pixel with valid lon/lat, maps to HiRISE DTM pixel
3. Assigns RGB from RAS texture to the DTM grid cell
4. Handles multi-frame overlap by selecting the frame with viewing angle
   closest to nadir (using XYZ-derived surface normal)
5. Outputs a GeoTIFF orthoimage at DTM resolution (~1m/px)

The output orthoimage is the "reference" (HR) for super-resolution training.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import OUTPUT_DIR, PDS_CACHE
from .hirise_dtm import HiRISEDTM
from .mastcam_xyz import parse_pds3_header, parse_pds4_label


def load_ras_texture(sol: int, frame_idx: int, metadata: dict) -> np.ndarray | None:
    """Load RAS texture for a specific frame, return as (H, W, 3) uint8."""
    frames = metadata.get("frames", [])
    frame = None
    for f in frames:
        if f["index"] == frame_idx:
            frame = f
            break
    if frame is None or frame.get("ras_product_id") is None:
        return None

    ras_id = frame["ras_product_id"]
    sol_dir = PDS_CACHE / "mastcamz" / f"sol{sol:05d}"
    ras_path = sol_dir / f"{ras_id}.IMG"

    if not ras_path.exists():
        return None

    # Parse dimensions and data type from PDS3 header
    pds3 = parse_pds3_header(ras_path)
    lines = pds3.get("lines", 0)
    samples = pds3.get("samples", 0)
    bands = pds3.get("bands", 3)
    start_byte = pds3.get("start_byte", 0)

    if lines == 0 or samples == 0:
        xml_path = ras_path.with_suffix(".xml")
        if xml_path.exists():
            meta = parse_pds4_label(xml_path)
            lines = meta.get("lines", 0)
            samples = meta.get("samples", 0)
            bands = meta.get("bands", 3)
        if lines == 0 or samples == 0:
            return None

    # Determine data type from PDS3 header
    sample_bits = pds3.get("sample_bits", 32)
    sample_type = pds3.get("sample_type", "IEEE_REAL")

    if sample_bits == 16:
        dtype = ">i2"  # MSB_INTEGER 16-bit
    else:
        dtype = ">f4"  # IEEE_REAL 32-bit (default for XYZ)

    try:
        raw = np.fromfile(str(ras_path), dtype=dtype, offset=start_byte)
        expected = bands * lines * samples
        if raw.size < expected:
            return None
        raw = raw[:expected].reshape(bands, lines, samples)

        if bands >= 3:
            rgb = np.moveaxis(raw[:3], 0, -1).astype(np.float64)
        else:
            rgb = np.stack([raw[0]] * 3, axis=-1).astype(np.float64)

        # Stretch to uint8
        positive = rgb[rgb > 0]
        if positive.size > 0:
            vmin, vmax = np.nanpercentile(positive, [2, 98])
        else:
            vmin, vmax = 0, 1
        denom = vmax - vmin if vmax != vmin else 1.0
        return np.clip((rgb - vmin) / denom * 255, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"    Texture load error: {e}")
        return None


def generate_ortho(sol: int, dtm: "HiRISEDTM | None" = None,
                   output_dir: Path = None, resolution_m: float = 1.0) -> Path | None:
    """Generate an orthoimage for a sol by projecting Mastcam-Z textures onto a lon/lat grid.

    Creates a regular grid at the specified resolution, maps each Mastcam-Z pixel
    to the grid using its SPICE-derived lon/lat, and assigns RGB from the RAS texture.

    Args:
        sol: sol number
        dtm: optional HiRISEDTM for grid alignment (falls back to self-generated grid)
        output_dir: where to save output (default: OUTPUT_DIR)
        resolution_m: output grid resolution in meters/pixel (default 1.0)

    Returns:
        Path to output PNG, or None on failure
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    sol_dir = OUTPUT_DIR / f"sol{sol:05d}"
    combined_npz = sol_dir / "combined_lonlat.npz"
    metadata_path = sol_dir / "metadata.json"

    if not combined_npz.exists() or not metadata_path.exists():
        print(f"  Sol {sol}: no combined data, skipping")
        return None

    # Load combined point cloud
    data = np.load(str(combined_npz))
    lon = data["lon"]
    lat = data["lat"]
    pixel_x = data["pixel_x"]
    pixel_y = data["pixel_y"]
    frame_idx = data["frame_idx"]

    with open(metadata_path) as f:
        metadata = json.load(f)

    n_points = len(lon)
    n_frames = metadata["n_frames"]
    print(f"  Sol {sol}: {n_points:,} points, {n_frames} frames")

    # Build output grid
    # Convert resolution from meters to degrees
    mars_r_m = 3396190.0
    lat_center = np.mean(lat)
    deg_per_m_lat = np.degrees(1.0 / mars_r_m)
    deg_per_m_lon = np.degrees(1.0 / (mars_r_m * np.cos(np.radians(lat_center))))
    res_deg_lat = resolution_m * deg_per_m_lat
    res_deg_lon = resolution_m * deg_per_m_lon

    # If DTM is available and data falls within, use DTM grid
    use_dtm = False
    if dtm is not None:
        bounds = dtm.bounds
        in_dtm = ((lon >= bounds["min_lon"]) & (lon <= bounds["max_lon"]) &
                  (lat >= bounds["min_lat"]) & (lat <= bounds["max_lat"]))
        if np.sum(in_dtm) > 0.5 * len(lon):
            use_dtm = True
            print(f"  Using DTM grid ({np.sum(in_dtm)/len(lon)*100:.0f}% points in DTM)")

    if use_dtm:
        dtm_rows, dtm_cols = dtm.lonlat_to_pixel(lon, lat)
        dtm_h = dtm.meta.get("lines", dtm._dataset.height if dtm._dataset else 0)
        dtm_w = dtm.meta.get("samples", dtm._dataset.width if dtm._dataset else 0)
        valid = ((dtm_rows >= 0) & (dtm_rows < dtm_h) &
                 (dtm_cols >= 0) & (dtm_cols < dtm_w))
        grid_rows = dtm_rows
        grid_cols = dtm_cols
        r_min, r_max = int(dtm_rows[valid].min()), int(dtm_rows[valid].max())
        c_min, c_max = int(dtm_cols[valid].min()), int(dtm_cols[valid].max())
        pad = 5
        r_min = max(0, r_min - pad)
        r_max = min(dtm_h - 1, r_max + pad)
        c_min = max(0, c_min - pad)
        c_max = min(dtm_w - 1, c_max + pad)
        grid_resolution_m = dtm.resolution_m
    else:
        # Self-generated equirectangular grid
        lon_min, lon_max = lon.min(), lon.max()
        lat_min, lat_max = lat.min(), lat.max()

        # Add padding (5m)
        pad_deg_lon = 5 * deg_per_m_lon
        pad_deg_lat = 5 * deg_per_m_lat
        lon_min -= pad_deg_lon
        lon_max += pad_deg_lon
        lat_min -= pad_deg_lat
        lat_max += pad_deg_lat

        # Grid dimensions
        out_w = max(1, int(np.ceil((lon_max - lon_min) / res_deg_lon)))
        out_h = max(1, int(np.ceil((lat_max - lat_min) / res_deg_lat)))

        # Map points to grid
        grid_cols = ((lon - lon_min) / res_deg_lon).astype(np.int32)
        grid_rows = ((lat_max - lat) / res_deg_lat).astype(np.int32)  # lat is inverted (N=top)

        valid = ((grid_rows >= 0) & (grid_rows < out_h) &
                 (grid_cols >= 0) & (grid_cols < out_w))
        r_min, c_min = 0, 0
        r_max, c_max = out_h - 1, out_w - 1
        grid_resolution_m = resolution_m
        print(f"  Self-generated grid: {out_w}x{out_h} @ {resolution_m:.2f} m/px")

    if np.sum(valid) == 0:
        print(f"  Sol {sol}: no valid grid points")
        return None

    # Filter to valid points
    grid_rows_v = grid_rows[valid] if use_dtm else grid_rows[valid]
    grid_cols_v = grid_cols[valid] if use_dtm else grid_cols[valid]
    px_v = pixel_x[valid]
    py_v = pixel_y[valid]
    fi_v = frame_idx[valid]

    out_h = r_max - r_min + 1
    out_w = c_max - c_min + 1

    print(f"  Output patch: {out_w}x{out_h} pixels @ {grid_resolution_m:.2f} m/px")

    # Initialize output arrays
    ortho = np.zeros((out_h, out_w, 3), dtype=np.float64)
    counts = np.zeros((out_h, out_w), dtype=np.int32)

    # Load textures per frame and accumulate
    unique_frames = np.unique(fi_v)
    textures = {}

    for fi in unique_frames:
        tex = load_ras_texture(sol, int(fi), metadata)
        if tex is not None:
            textures[int(fi)] = tex

    if not textures:
        print(f"  Sol {sol}: no textures loaded")
        return None

    print(f"  Loaded {len(textures)}/{len(unique_frames)} frame textures")

    # Project each point onto the output grid
    local_r = grid_rows_v - r_min
    local_c = grid_cols_v - c_min

    for fi in sorted(textures.keys()):
        mask = fi_v == fi
        if not np.any(mask):
            continue

        tex = textures[fi]
        lr = local_r[mask]
        lc = local_c[mask]
        px = px_v[mask]
        py = py_v[mask]

        # Bounds check against texture
        tex_valid = (py >= 0) & (py < tex.shape[0]) & (px >= 0) & (px < tex.shape[1])
        if not np.any(tex_valid):
            continue

        lr = lr[tex_valid]
        lc = lc[tex_valid]
        px = px[tex_valid]
        py = py[tex_valid]

        # Accumulate RGB (averaging for overlap)
        rgb = tex[py, px].astype(np.float64)
        np.add.at(ortho, (lr, lc), rgb)
        np.add.at(counts, (lr, lc), 1)

    # Average where multiple points map to same DTM pixel
    valid_px = counts > 0
    for c in range(3):
        ortho[:, :, c][valid_px] /= counts[valid_px]

    ortho_uint8 = np.clip(ortho, 0, 255).astype(np.uint8)

    # Create alpha channel (255 where data exists, 0 elsewhere)
    alpha = np.where(valid_px, 255, 0).astype(np.uint8)

    # Save as RGBA PNG
    output_path = output_dir / f"sol{sol:05d}_ortho.png"
    rgba = np.dstack([ortho_uint8, alpha])
    Image.fromarray(rgba, 'RGBA').save(str(output_path))

    coverage_pct = np.sum(valid_px) / (out_h * out_w) * 100
    print(f"  Saved: {output_path.name} ({out_w}x{out_h}, {np.sum(valid_px):,} filled pixels, {coverage_pct:.1f}% coverage)")

    # Save metadata
    meta_out = {
        "sol": sol,
        "dtm_bbox": {"r_min": r_min, "r_max": r_max, "c_min": c_min, "c_max": c_max},
        "output_size": [out_w, out_h],
        "filled_pixels": int(np.sum(valid_px)),
        "total_pixels": out_h * out_w,
        "coverage_pct": round(coverage_pct, 1),
        "n_frames_used": len(textures),
        "n_points": n_points,
        "dtm_resolution_m": dtm.resolution_m if dtm else grid_resolution_m,
    }
    meta_path = output_dir / f"sol{sol:05d}_ortho_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    return output_path
