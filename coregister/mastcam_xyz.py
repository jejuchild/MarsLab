"""Mastcam-Z XYZ product parsing and coordinate transformation.

Reads PDS4 XYZ products (3D point clouds from stereo reconstruction),
transforms coordinates from rover frame to Mars body-fixed (IAU_MARS),
and converts to areocentric lon/lat/alt.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import spiceypy as spice


def parse_pds4_label(label_path: Path) -> dict:
    """Parse a PDS4 XML label to extract image and geometry metadata.

    Returns dict with keys: shape, bands, data_type, start_byte, item_bytes,
    axes_names, time_utc, instrument_id, etc.
    """
    tree = ET.parse(label_path)
    root = tree.getroot()

    def find(root, tag_name):
        """Find element by local tag name, namespace-agnostic.

        Uses {*}tag wildcard syntax (Python 3.8+) first, then falls back
        to iterating all elements. For context-sensitive tags, pass a
        parent/child path like 'Array_2D_Image/axes'.
        """
        # Try {*}tag wildcard (matches any namespace)
        el = root.find(f".//{{{root.tag.split('}')[0].strip('{')}}}{tag_name}"
                       if "}" in root.tag else f".//{tag_name}")
        if el is not None:
            return el

        # Wildcard namespace search
        try:
            el = root.find(f".//{{*}}{tag_name}")
            if el is not None:
                return el
        except SyntaxError:
            pass

        # Fallback: iterate and match local name
        for el in root.iter():
            local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if local == tag_name:
                return el
        return None

    def find_text(root, paths, default=""):
        el = find(root, paths)
        return el.text.strip() if el is not None and el.text else default

    meta = {}

    # Image dimensions
    meta["lines"] = int(find_text(root, "lines", "0"))
    meta["samples"] = int(find_text(root, "samples", "0"))
    meta["bands"] = int(find_text(root, "bands", "3"))  # XYZ = 3 bands

    # Data format
    meta["data_type"] = find_text(root, "data_type", "IEEE754MSBSingle")
    meta["start_byte"] = int(find_text(root, "header_bytes", "0"))

    # Observation time
    meta["start_time"] = find_text(root, "start_date_time", "")
    meta["stop_time"] = find_text(root, "stop_date_time", "")

    # Missing constant (invalid XYZ points)
    mc_text = find_text(root, "missing_constant", "")
    meta["missing_constant"] = float(mc_text) if mc_text else None

    # Instrument
    meta["instrument_id"] = find_text(root, "instrument_id", "MASTCAM_Z")

    return meta


def parse_pds3_header(data_path: Path) -> dict:
    """Parse PDS3 embedded header from an IMG file for data offset and dimensions."""
    meta = {}
    with open(data_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                if line == "END":
                    break
                continue
            key, _, val = line.partition("=")
            key = key.strip().upper()
            val = val.strip().strip('"')
            if "<" in val:
                val = val[:val.index("<")].strip()
            try:
                if key == "RECORD_BYTES":
                    meta["record_bytes"] = int(val)
                elif key == "^IMAGE" and "HEADER" not in line.upper().split("=")[0]:
                    meta["image_start"] = int(val)
                elif key == "LINES":
                    meta["lines"] = int(val)
                elif key == "LINE_SAMPLES":
                    meta["samples"] = int(val)
                elif key == "BANDS":
                    meta["bands"] = int(val)
            except ValueError:
                pass
            if line == "END":
                break
    # Compute data offset
    if "record_bytes" in meta and "image_start" in meta:
        meta["start_byte"] = meta["record_bytes"] * (meta["image_start"] - 1)
    return meta


def read_xyz_product(data_path: Path, label_path: Path = None) -> tuple[np.ndarray, dict]:
    """Read a Mastcam-Z XYZ product.

    XYZ products contain 3 bands (X, Y, Z) in rover frame coordinates.

    Returns:
        xyz: ndarray of shape (H, W, 3) with X, Y, Z values in meters (rover frame)
        meta: dict with label metadata
    """
    # Parse label
    if label_path is None:
        label_path = data_path.with_suffix(".xml")
        if not label_path.exists():
            label_path = data_path.with_suffix(".lbl")

    meta = {}
    if label_path.exists():
        meta = parse_pds4_label(label_path)

    # Always parse PDS3 embedded header for reliable data offset
    pds3_meta = parse_pds3_header(data_path)

    # Use PDS3 offset (reliable) over XML header_bytes (often 0)
    if pds3_meta.get("start_byte", 0) > 0:
        meta["start_byte"] = pds3_meta["start_byte"]

    # Fill in dimensions from PDS3 if missing from XML
    for key in ("lines", "samples", "bands"):
        if meta.get(key, 0) == 0 and key in pds3_meta:
            meta[key] = pds3_meta[key]

    # Read binary data
    # PDS4 XYZ products are typically 3-band IEEE 754 single precision float, BSQ
    suffix = data_path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        # GeoTIFF format
        import rasterio
        with rasterio.open(data_path) as ds:
            bands = ds.read()  # shape: (bands, H, W)
            xyz = np.moveaxis(bands, 0, -1)  # -> (H, W, bands)
            if not meta:
                meta["lines"] = ds.height
                meta["samples"] = ds.width
                meta["bands"] = ds.count
    else:
        # Raw IMG/VIC format
        lines = meta.get("lines", 0)
        samples = meta.get("samples", 0)
        bands = meta.get("bands", 3)
        start_byte = meta.get("start_byte", 0)

        if lines == 0 or samples == 0:
            raise ValueError(
                f"Cannot determine image dimensions from label for {data_path.name}. "
                "Provide a valid PDS4 XML or PDS3 LBL label."
            )

        raw = np.fromfile(str(data_path), dtype=">f4", offset=start_byte)
        raw = raw.astype(np.float64)  # float32 → float64 BEFORE coordinate transform

        # BSQ layout: band, line, sample
        expected = bands * lines * samples
        if raw.size >= expected:
            raw = raw[:expected]
            xyz = raw.reshape(bands, lines, samples)
            xyz = np.moveaxis(xyz, 0, -1)  # -> (H, W, 3)
        else:
            raise ValueError(
                f"XYZ file too small: expected {expected} floats, got {raw.size}"
            )

    # Mark invalid points
    missing = meta.get("missing_constant")
    if missing is not None:
        invalid = np.any(np.abs(xyz - missing) < 1e-10, axis=-1)
        xyz[invalid] = np.nan

    # Also mark (0,0,0) as invalid
    zero_mask = np.all(xyz == 0, axis=-1)
    xyz[zero_mask] = np.nan

    return xyz, meta


def parse_site_to_rover_transform(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ROVER_COORDINATE_SYSTEM from PDS3 embedded header.

    Returns:
        offset: (3,) rover position in site frame [meters]
        quat: (4,) rotation quaternion (s, v1, v2, v3) site→rover
    """
    offset = np.zeros(3)
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    # Read entire header as one string, then parse
    header_text = ""
    with open(data_path, "r", errors="replace") as f:
        for line in f:
            header_text += line
            if line.strip() == "END":
                break

    import re

    # Find ROVER_COORDINATE_SYSTEM group
    m = re.search(
        r"GROUP\s*=\s*ROVER_COORDINATE_SYSTEM(.*?)END_GROUP",
        header_text, re.DOTALL
    )
    if not m:
        return offset, quat

    block = m.group(1)

    # Parse ORIGIN_OFFSET_VECTOR
    m_off = re.search(r"ORIGIN_OFFSET_VECTOR\s*=\s*\(([^)]+)\)", block)
    if m_off:
        parts = [float(x.strip()) for x in m_off.group(1).split(",")]
        offset = np.array(parts[:3])

    # Parse ORIGIN_ROTATION_QUATERNION (may span lines)
    m_q = re.search(r"ORIGIN_ROTATION_QUATERNION\s*=\s*\(([^)]+)\)", block)
    if m_q:
        raw = m_q.group(1).replace("\n", "").replace("\r", "")
        parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
        quat = np.array(parts[:4])

    return offset, quat


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (s, v1, v2, v3) to 3x3 rotation matrix."""
    s, v1, v2, v3 = q
    return np.array([
        [1 - 2*(v2**2 + v3**2), 2*(v1*v2 - s*v3),     2*(v1*v3 + s*v2)],
        [2*(v1*v2 + s*v3),     1 - 2*(v1**2 + v3**2), 2*(v2*v3 - s*v1)],
        [2*(v1*v3 - s*v2),     2*(v2*v3 + s*v1),     1 - 2*(v1**2 + v2**2)],
    ])


def site_to_rover_frame(
    xyz_site: np.ndarray,
    rover_offset: np.ndarray,
    rover_quat: np.ndarray,
) -> np.ndarray:
    """Transform XYZ from SITE_FRAME to ROVER_NAV_FRAME.

    Args:
        xyz_site: (H, W, 3) in site frame [meters]
        rover_offset: (3,) rover position in site frame
        rover_quat: (4,) quaternion (s, v1, v2, v3)

    Returns:
        xyz_rover: (H, W, 3) in rover frame [meters]
    """
    h, w, _ = xyz_site.shape
    valid = ~np.isnan(xyz_site[:, :, 0])
    xyz_rover = np.full_like(xyz_site, np.nan)

    # site → rover: subtract offset, then rotate
    # R_site_to_rover = quat_to_rotation_matrix(quat)
    # xyz_rover = R @ (xyz_site - offset)
    rot = quat_to_rotation_matrix(rover_quat)

    pts = xyz_site[valid] - rover_offset  # translate
    rotated = (rot @ pts.T).T  # rotate
    xyz_rover[valid] = rotated

    return xyz_rover


def site_to_mars_bodyfixed(
    xyz_site: np.ndarray,
    utc_time: str,
    rover_offset: np.ndarray,
    rover_quat: np.ndarray,
) -> np.ndarray:
    """Transform XYZ from SITE_FRAME to Mars body-fixed (IAU_MARS).

    Chain: SITE_FRAME → ROVER_NAV_FRAME → IAU_MARS

    Args:
        xyz_site: (H, W, 3) array, XYZ in site frame [meters]
        utc_time: UTC time string
        rover_offset: (3,) rover position in site frame
        rover_quat: (4,) site-to-rover quaternion

    Returns:
        xyz_mars: (H, W, 3) array, XYZ in IAU_MARS frame [km]
    """
    # Step 1: SITE → ROVER
    xyz_rover = site_to_rover_frame(xyz_site, rover_offset, rover_quat)

    # Step 2: ROVER → IAU_MARS (via SPICE)
    et = spice.utc2et(utc_time)
    rot = spice.pxform("M2020_ROVER", "IAU_MARS", et)
    rover_pos_km, _ = spice.spkpos("M2020", et, "IAU_MARS", "NONE", "MARS")

    h, w, _ = xyz_rover.shape
    valid = ~np.isnan(xyz_rover[:, :, 0])
    xyz_mars = np.full_like(xyz_rover, np.nan)

    valid_pts = xyz_rover[valid] / 1000.0  # m → km
    rotated = (rot @ valid_pts.T).T
    translated = rotated + rover_pos_km

    xyz_mars[valid] = translated
    return xyz_mars


def bodyfixed_to_lonlat(xyz_mars: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Mars body-fixed XYZ [km] to areocentric lon/lat/alt.

    Returns:
        lon: longitude in degrees (0-360 east)
        lat: latitude in degrees (-90 to 90)
        alt: altitude above reference ellipsoid in km
    """
    h, w, _ = xyz_mars.shape
    valid = ~np.isnan(xyz_mars[:, :, 0])

    lon = np.full((h, w), np.nan)
    lat = np.full((h, w), np.nan)
    alt = np.full((h, w), np.nan)

    valid_pts = xyz_mars[valid]  # (N, 3)
    if valid_pts.size == 0:
        return lon, lat, alt

    # Use SPICE recgeo for geodetic coordinates
    # Mars radii from PCK kernel
    radii = spice.bodvrd("MARS", "RADII", 3)[1]
    re = radii[0]  # equatorial
    rp = radii[2]  # polar
    f = (re - rp) / re  # flattening

    i, j = np.where(valid)
    for idx in range(valid_pts.shape[0]):
        pt = valid_pts[idx]
        lo, la, al = spice.recgeo(pt, re, f)
        lon_val = np.degrees(lo)
        if lon_val < 0:
            lon_val += 360.0
        lon[i[idx], j[idx]] = lon_val
        lat[i[idx], j[idx]] = np.degrees(la)
        alt[i[idx], j[idx]] = al

    return lon, lat, alt


def bodyfixed_to_lonlat_vectorized(xyz_mars: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized version of bodyfixed_to_lonlat (much faster for large arrays).

    Computes areocentric (spherical) lon/lat rather than geodetic,
    which is sufficient for matching to HiRISE DTM grids.
    """
    h, w, _ = xyz_mars.shape
    valid = ~np.isnan(xyz_mars[:, :, 0])

    lon = np.full((h, w), np.nan)
    lat = np.full((h, w), np.nan)
    alt = np.full((h, w), np.nan)

    pts = xyz_mars[valid]  # (N, 3)
    if pts.size == 0:
        return lon, lat, alt

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Areocentric (spherical) coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    lat_rad = np.arcsin(z / r)
    lon_rad = np.arctan2(y, x)

    lon_deg = np.degrees(lon_rad)
    lon_deg[lon_deg < 0] += 360.0
    lat_deg = np.degrees(lat_rad)

    # Altitude above sphere (approximate)
    mars_r = 3389.5  # mean radius km
    alt_km = r - mars_r

    lon[valid] = lon_deg
    lat[valid] = lat_deg
    alt[valid] = alt_km

    return lon, lat, alt


def process_mastcamz_xyz(
    data_path: Path,
    label_path: Path = None,
    utc_time: str = None,
) -> dict:
    """Full pipeline: read XYZ product -> transform to Mars lon/lat.

    Returns dict with:
        xyz_rover: (H,W,3) rover frame
        xyz_mars: (H,W,3) IAU_MARS body-fixed
        lon, lat, alt: (H,W) arrays
        meta: label metadata
    """
    # Read XYZ product (in SITE_FRAME)
    xyz_site, meta = read_xyz_product(data_path, label_path)
    print(f"  XYZ loaded: {xyz_site.shape[1]}x{xyz_site.shape[0]}, "
          f"{np.sum(~np.isnan(xyz_site[:,:,0]))} valid points")

    # Get observation time from label if not provided
    if utc_time is None:
        utc_time = meta.get("start_time", "")
        if not utc_time:
            raise ValueError("No UTC time in label and none provided")

    # Parse site→rover transform from PDS3 header
    rover_offset, rover_quat = parse_site_to_rover_transform(data_path)
    print(f"  Rover in site frame: offset={rover_offset}, quat={rover_quat}")

    # Transform: SITE_FRAME → ROVER_NAV_FRAME → IAU_MARS
    print(f"  Transforming SITE→ROVER→IAU_MARS (t={utc_time})...")
    xyz_mars = site_to_mars_bodyfixed(xyz_site, utc_time, rover_offset, rover_quat)

    # Convert to lon/lat
    print(f"  Computing lon/lat...")
    lon, lat, alt = bodyfixed_to_lonlat_vectorized(xyz_mars)

    valid_count = np.sum(~np.isnan(lon))
    if valid_count > 0:
        print(f"  Coverage: lon=[{np.nanmin(lon):.4f}, {np.nanmax(lon):.4f}], "
              f"lat=[{np.nanmin(lat):.4f}, {np.nanmax(lat):.4f}]")

    return {
        "xyz_site": xyz_site,
        "xyz_mars": xyz_mars,
        "lon": lon,
        "lat": lat,
        "alt": alt,
        "meta": meta,
        "utc_time": utc_time,
    }
