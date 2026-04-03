"""Project Mastcam-Z textures directly onto HiRISE pixel grid.

Instead of creating a separate ortho grid, this module maps each Mastcam-Z
point directly to HiRISE pixel coordinates using the same map projection.
The result is perfectly aligned with HiRISE by construction.

For x4 SR: each HiRISE pixel (0.25m) is subdivided into 4x4 sub-pixels (0.0625m).
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy

from .config import OUTPUT_DIR, PDS_CACHE
from .ortho import load_ras_texture


def lonlat_to_hirise_subpixel(lon, lat, hir_ds, scale=4):
    """Convert lon/lat to HiRISE sub-pixel coordinates.

    Args:
        lon, lat: arrays in degrees (areocentric, east-positive)
        hir_ds: open rasterio dataset of HiRISE JP2
        scale: sub-pixel scale (4 = x4 SR, each HiRISE px → 4x4 sub-px)

    Returns:
        sub_row, sub_col: sub-pixel coordinates (int)
    """
    # HiRISE uses equirectangular projection with standard_parallel=0, central_meridian=180
    # and latitude_of_origin=15 (from the CRS)
    # The projection equations:
    #   x = R * cos(lat_ref) * (lon - lon_0) * pi/180
    #   y = R * lat * pi/180
    # where R is from the CRS SPHEROID, lat_ref from CRS, lon_0=180
    R = 3394839.8133163  # from HiRISE CRS SPHEROID
    lat_ref = 15.0       # from latitude_of_origin in CRS

    proj_x = R * np.cos(np.radians(lat_ref)) * (lon - 180.0) * np.pi / 180.0
    proj_y = R * lat * np.pi / 180.0

    # Convert to HiRISE pixel coords (float for sub-pixel precision)
    # rasterio rowcol returns integer, we need float
    # transform: | res_x, 0, origin_x |
    #            | 0, -res_y, origin_y |
    t = hir_ds.transform
    col_f = (proj_x - t.c) / t.a  # (x - origin_x) / res_x
    row_f = (proj_y - t.f) / t.e  # (y - origin_y) / res_y (res_y is negative)

    # Scale to sub-pixel grid
    sub_col = (col_f * scale).astype(np.int32)
    sub_row = (row_f * scale).astype(np.int32)

    return sub_row, sub_col


def generate_ortho_on_hirise(sol: int, hir_ds, scale: int = 4,
                              output_dir: Path = None) -> Path | None:
    """Generate Mastcam-Z ortho directly on HiRISE pixel grid.

    Args:
        sol: sol number
        hir_ds: open rasterio dataset of HiRISE JP2
        scale: sub-pixel scale factor (4 = x4 SR)
        output_dir: output directory

    Returns:
        Path to output files, or None
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    sol_dir = OUTPUT_DIR / f"sol{sol:05d}"
    combined_npz = sol_dir / "combined_lonlat.npz"
    metadata_path = sol_dir / "metadata.json"

    if not combined_npz.exists() or not metadata_path.exists():
        return None

    data = np.load(str(combined_npz))
    lon = data["lon"]
    lat = data["lat"]
    pixel_x = data["pixel_x"]
    pixel_y = data["pixel_y"]
    frame_idx = data["frame_idx"]

    with open(metadata_path) as f:
        metadata = json.load(f)

    n_points = len(lon)
    print(f"  Sol {sol}: {n_points:,} points, {metadata['n_frames']} frames")

    # Map to HiRISE sub-pixel grid
    sub_row, sub_col = lonlat_to_hirise_subpixel(lon, lat, hir_ds, scale)

    # HiRISE bounds in sub-pixels
    hir_h = hir_ds.height * scale
    hir_w = hir_ds.width * scale

    # Filter valid
    valid = ((sub_row >= 0) & (sub_row < hir_h) &
             (sub_col >= 0) & (sub_col < hir_w))

    if np.sum(valid) == 0:
        print(f"  Sol {sol}: no points in HiRISE bounds")
        return None

    sub_row_v = sub_row[valid]
    sub_col_v = sub_col[valid]
    px_v = pixel_x[valid]
    py_v = pixel_y[valid]
    fi_v = frame_idx[valid]

    # Bounding box in sub-pixel space
    r_min, r_max = int(sub_row_v.min()), int(sub_row_v.max())
    c_min, c_max = int(sub_col_v.min()), int(sub_col_v.max())

    # Add padding
    pad = scale * 2
    r_min = max(0, r_min - pad)
    r_max = min(hir_h - 1, r_max + pad)
    c_min = max(0, c_min - pad)
    c_max = min(hir_w - 1, c_max + pad)

    out_h = r_max - r_min + 1
    out_w = c_max - c_min + 1

    print(f"  Sub-pixel patch: {out_w}x{out_h} (HiRISE {out_w//scale}x{out_h//scale} px)")

    # Build HR image
    hr = np.zeros((out_h, out_w, 3), dtype=np.float64)
    counts = np.zeros((out_h, out_w), dtype=np.int32)

    # Load textures
    unique_frames = np.unique(fi_v)
    textures = {}
    for fi in unique_frames:
        tex = load_ras_texture(sol, int(fi), metadata)
        if tex is not None:
            textures[int(fi)] = tex

    if not textures:
        print(f"  Sol {sol}: no textures")
        return None

    print(f"  Loaded {len(textures)}/{len(unique_frames)} textures")

    local_r = sub_row_v - r_min
    local_c = sub_col_v - c_min

    for fi in sorted(textures.keys()):
        mask = fi_v == fi
        if not np.any(mask):
            continue

        tex = textures[fi]
        lr = local_r[mask]
        lc = local_c[mask]
        px = px_v[mask]
        py = py_v[mask]

        tex_valid = (py >= 0) & (py < tex.shape[0]) & (px >= 0) & (px < tex.shape[1])
        if not np.any(tex_valid):
            continue

        rgb = tex[py[tex_valid], px[tex_valid]].astype(np.float64)
        np.add.at(hr, (lr[tex_valid], lc[tex_valid]), rgb)
        np.add.at(counts, (lr[tex_valid], lc[tex_valid]), 1)

    filled = counts > 0
    for c in range(3):
        hr[:, :, c][filled] /= counts[filled]

    hr_uint8 = np.clip(hr, 0, 255).astype(np.uint8)
    alpha = np.where(filled, 255, 0).astype(np.uint8)

    # Also extract corresponding LR (HiRISE) patch
    # sub-pixel bbox → HiRISE pixel bbox
    hir_r_min = r_min // scale
    hir_c_min = c_min // scale
    hir_r_max = (r_max + 1) // scale
    hir_c_max = (c_max + 1) // scale
    lr_h = hir_r_max - hir_r_min
    lr_w = hir_c_max - hir_c_min

    window = rasterio.windows.Window(hir_c_min, hir_r_min, lr_w, lr_h)
    lr_data = hir_ds.read(1, window=window).astype(np.float64)

    # Normalize LR
    valid_lr = lr_data[lr_data > 0]
    if len(valid_lr) > 0:
        vmin, vmax = np.percentile(valid_lr, [2, 98])
        lr_uint8 = np.clip((lr_data - vmin) / max(vmax - vmin, 1) * 255, 0, 255).astype(np.uint8)
    else:
        lr_uint8 = np.zeros_like(lr_data, dtype=np.uint8)

    # Align HR to exact LR*scale size
    aligned_h = lr_h * scale
    aligned_w = lr_w * scale
    hr_aligned = np.zeros((aligned_h, aligned_w, 3), dtype=np.uint8)
    alpha_aligned = np.zeros((aligned_h, aligned_w), dtype=np.uint8)

    # Offset within the aligned grid
    off_r = r_min - hir_r_min * scale
    off_c = c_min - hir_c_min * scale
    copy_h = min(out_h, aligned_h - off_r)
    copy_w = min(out_w, aligned_w - off_c)

    if off_r >= 0 and off_c >= 0 and copy_h > 0 and copy_w > 0:
        hr_aligned[off_r:off_r+copy_h, off_c:off_c+copy_w] = hr_uint8[:copy_h, :copy_w]
        alpha_aligned[off_r:off_r+copy_h, off_c:off_c+copy_w] = alpha[:copy_h, :copy_w]

    # Save
    from PIL import Image

    hr_rgba = np.dstack([hr_aligned, alpha_aligned])
    hr_path = output_dir / f"sol{sol:05d}_hr_aligned.png"
    Image.fromarray(hr_rgba, 'RGBA').save(str(hr_path))

    lr_path = output_dir / f"sol{sol:05d}_lr_aligned.png"
    Image.fromarray(lr_uint8).save(str(lr_path))

    filled_count = int(np.sum(alpha_aligned > 0))
    total = aligned_h * aligned_w
    coverage = filled_count / total * 100

    print(f"  HR: {aligned_w}x{aligned_h} ({filled_count:,} filled, {coverage:.1f}%)")
    print(f"  LR: {lr_w}x{lr_h}")
    print(f"  Saved: {hr_path.name}, {lr_path.name}")

    meta_out = {
        "sol": sol,
        "scale": scale,
        "hir_bbox": {"r_min": hir_r_min, "c_min": hir_c_min,
                     "r_max": hir_r_max, "c_max": hir_c_max},
        "lr_size": [lr_w, lr_h],
        "hr_size": [aligned_w, aligned_h],
        "filled_pixels": filled_count,
        "coverage_pct": round(coverage, 1),
    }
    with open(output_dir / f"sol{sol:05d}_aligned_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    return hr_path
