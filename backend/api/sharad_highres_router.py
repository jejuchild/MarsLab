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
from cachetools import LRUCache

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

# SHARAD timing: ADC samples at 26.67 MHz (80/3 MHz) via bandpass undersampling.
# Sample interval = 1 / 26.67e6 = 3/80 µs = 0.0375 µs two-way travel time per range bin.
# Ref: Seu et al. 2007; PDS SHARAD Radargram V1.0.
SHARAD_SAMPLE_INTERVAL_US = 3.0 / 80.0  # 0.0375 µs (= 1/26.67 MHz)
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

    # Vectorized parsing: read entire file as raw bytes, then extract fields
    raw = np.fromfile(dat_path, dtype=np.uint8)
    actual_rows = len(raw) // ROW_BYTES
    if actual_rows < total_rows:
        total_rows = actual_rows
    raw = raw[:total_rows * ROW_BYTES].reshape(total_rows, ROW_BYTES)

    # Extract real/imag echo columns via view (zero-copy slice)
    real = np.ndarray((total_rows, N_RANGE_BINS), dtype="<f4",
                      buffer=raw, offset=0,
                      strides=(ROW_BYTES, 4))
    # offset within each row for real part
    real = np.frombuffer(raw[:, ECHO_REAL_OFFSET:ECHO_REAL_OFFSET + ECHO_BYTES].tobytes(),
                         dtype="<f4").reshape(total_rows, N_RANGE_BINS)
    imag = np.frombuffer(raw[:, ECHO_IMAG_OFFSET:ECHO_IMAG_OFFSET + ECHO_BYTES].tobytes(),
                         dtype="<f4").reshape(total_rows, N_RANGE_BINS)
    power = (real ** 2 + imag ** 2).astype(np.float32)

    # Extract geometry columns
    lons = np.frombuffer(raw[:, LON_OFFSET:LON_OFFSET + 8].tobytes(),
                         dtype="<f8").copy()
    lats = np.frombuffer(raw[:, LAT_OFFSET:LAT_OFFSET + 8].tobytes(),
                         dtype="<f8").copy()
    alts = np.frombuffer(raw[:, ALT_OFFSET:ALT_OFFSET + 8].tobytes(),
                         dtype="<f8").copy()

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


# Mars IAU 2000 ellipsoid radii (meters)
_MARS_A = 3396190.0   # equatorial radius
_MARS_B = 3376200.0   # polar radius
_MARS_AB2 = (_MARS_A / _MARS_B) ** 2   # ≈ 1.01184


def _centric_to_graphic(lat_centric: np.ndarray) -> np.ndarray:
    """Convert planetocentric latitude (from SHARAD RDR) to planetographic
    (geodetic) latitude expected by Cesium's oblate Mars ellipsoid.

    tan(lat_graphic) = (a/b)^2 * tan(lat_centric)
    """
    rad = np.radians(lat_centric)
    return np.degrees(np.arctan(_MARS_AB2 * np.tan(rad)))


def _centric_to_graphic_scalar(lat_centric: float) -> float:
    """Scalar version for single latitude values."""
    rad = math.radians(lat_centric)
    return math.degrees(math.atan(_MARS_AB2 * math.tan(rad)))


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


# ── Global percentile cache (for consistent tile brightness) ──
_percentile_cache: dict[str, tuple[float, float]] = {}


def _get_global_percentiles(
    product_id: str,
    use_log: bool = True,
    amplitude: bool = False,
    pmin: float = 1.0,
    pmax: float = 99.0,
) -> tuple[float, float]:
    """Compute and cache (vmin, vmax) from the full power array.

    These percentiles are used for tile rendering so that all tiles
    have identical contrast regardless of their local data range.
    """
    key = f"{product_id.upper()}_{use_log}_{amplitude}_{pmin}_{pmax}"
    if key in _percentile_cache:
        return _percentile_cache[key]

    power, _ = _get_power(product_id)
    # Subsample for speed: take every 10th trace (still representative)
    step = max(1, power.shape[0] // 10000)
    p = np.array(power[::step], dtype=np.float32)

    if amplitude:
        np.maximum(p, 0, out=p)
        np.sqrt(p, out=p)
    if use_log:
        p = np.log10(np.maximum(p, 1e-12))

    vmin = float(np.percentile(p, pmin))
    vmax = float(np.percentile(p, pmax))
    if vmax <= vmin:
        vmax = vmin + 1

    _percentile_cache[key] = (vmin, vmax)
    return (vmin, vmax)


# ── Radargram rendering ───────────────────────────────
_png_cache: LRUCache = LRUCache(maxsize=20)  # key -> PNG bytes (bounded LRU)


def _render_radargram_png(
    power: np.ndarray,
    downsample: int = 1,
    use_log: bool = True,
    pmin: float = 1.0,
    pmax: float = 99.0,
    amplitude: bool = False,
    per_trace: bool = False,
    cache_key: str = "",
    global_vmin: float | None = None,
    global_vmax: float | None = None,
) -> bytes:
    """Render the radargram as a grayscale PNG (cached by params).

    amplitude: if True, use sqrt(power) instead of raw power.
    per_trace: if True, normalize each trace independently (like PDS browse images).
    global_vmin/global_vmax: if provided, use these for normalization instead of
      computing percentiles from the tile data (for consistent tile brightness).
    """
    if cache_key and cache_key in _png_cache:
        return _png_cache[cache_key]

    if downsample > 1:
        n = power.shape[0] // downsample * downsample
        p = power[:n].reshape(-1, downsample, power.shape[1]).mean(axis=1)
    else:
        p = np.array(power, dtype=np.float32)

    if amplitude:
        np.maximum(p, 0, out=p)
        np.sqrt(p, out=p)

    if use_log:
        p = np.log10(np.maximum(p, 1e-12))

    if per_trace:
        # Per-trace normalization (vectorized) — matches PDS browse image style
        vmins = np.percentile(p, pmin, axis=1, keepdims=True)
        vmaxs = np.percentile(p, pmax, axis=1, keepdims=True)
        vmaxs = np.where(vmaxs <= vmins, vmins + 1, vmaxs)
        img = ((p - vmins) / (vmaxs - vmins) * 255).clip(0, 255).astype(np.uint8)
    else:
        if global_vmin is not None and global_vmax is not None:
            vmin, vmax = global_vmin, global_vmax
        else:
            vmin = np.percentile(p, pmin)
            vmax = np.percentile(p, pmax)
        if vmax <= vmin:
            vmax = vmin + 1
        img = ((p - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)

    img = img.T

    pil_img = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=False)
    png_bytes = buf.getvalue()

    if cache_key:
        _png_cache[cache_key] = png_bytes

    return png_bytes


# ── Cluttergram (surface clutter simulation) ─────────
_clutter_caches: dict[str, dict] = {}


def _obs_id_from_product(product_id: str) -> str:
    """Extract observation ID from RDR product ID. R_0277201_... -> 0277201"""
    parts = product_id.upper().split("_")
    return parts[1] if len(parts) >= 2 else ""


def _resolve_cluttergram(product_id: str) -> dict | None:
    """
    Find cluttergram sim file for a given RDR product.
    Returns dict with 'sim_path', 'xml_path', 'lines', 'samples',
    'combined_offset', and start/stop coordinates from the XML geometry.
    """
    obs_id = _obs_id_from_product(product_id)
    if not obs_id:
        return None

    sim_id = obs_id.zfill(8)
    sim_path = os.path.join(SHARAD_HR_DIR, f"s_{sim_id}_sim.img")
    xml_path = os.path.join(SHARAD_HR_DIR, f"s_{sim_id}_sim.xml")

    if not os.path.exists(sim_path):
        return None

    # Parse XML label for dimensions, offset, and geometry
    lines = 0
    samples = 0
    combined_offset = -1
    start_lat = start_lon = stop_lat = stop_lon = None

    if os.path.exists(xml_path):
        try:
            import xml.etree.ElementTree as ET
            with open(xml_path, "r") as f:
                root = ET.fromstring(f.read())

            ns = "{http://pds.nasa.gov/pds4/pds/v1}"
            for arr in root.iter(f"{ns}Array_2D_Image"):
                lid = arr.find(f"{ns}local_identifier")
                if lid is not None and "Combined" in lid.text:
                    offset_el = arr.find(f"{ns}offset")
                    if offset_el is not None:
                        combined_offset = int(offset_el.text)
                    for ax in arr.findall(f"{ns}Axis_Array"):
                        ax_name = ax.find(f"{ns}axis_name")
                        ax_elems = ax.find(f"{ns}elements")
                        if ax_name is not None and ax_elems is not None:
                            if ax_name.text == "Line":
                                lines = int(ax_elems.text)
                            elif ax_name.text == "Sample":
                                samples = int(ax_elems.text)
                    break

            # Parse start/stop coordinates from geometry section
            geom_ns = "{http://pds.nasa.gov/pds4/geom/v1}"
            for sg in root.iter(f"{geom_ns}Surface_Geometry_Start_Stop"):
                try:
                    start_lat = float(sg.find(f"{geom_ns}start_subspacecraft_latitude").text)
                    stop_lat = float(sg.find(f"{geom_ns}stop_subspacecraft_latitude").text)
                    start_lon = float(sg.find(f"{geom_ns}start_subspacecraft_longitude").text)
                    stop_lon = float(sg.find(f"{geom_ns}stop_subspacecraft_longitude").text)
                except (AttributeError, ValueError):
                    pass
                break
        except Exception:
            pass

    # Fallback: infer from file size (3 equal sections of float32)
    if lines == 0 or samples == 0:
        file_size = os.path.getsize(sim_path)
        n_floats_per_section = file_size // (3 * 4)
        # Try common sample counts
        for s in [3600, 1433, 2048, 1024]:
            if n_floats_per_section % s == 0:
                samples = s
                lines = n_floats_per_section // s
                break
        else:
            # Last resort: assume square-ish
            samples = int(n_floats_per_section ** 0.5)
            lines = n_floats_per_section // samples

    if combined_offset < 0:
        combined_offset = 2 * lines * samples * 4

    return {
        "sim_path": sim_path,
        "xml_path": xml_path,
        "lines": lines,
        "samples": samples,
        "combined_offset": combined_offset,
        "start_lat": start_lat,
        "stop_lat": stop_lat,
        "start_lon": start_lon,
        "stop_lon": stop_lon,
    }


def _get_cluttergram(product_id: str) -> np.ndarray | None:
    """Load and cache the combined cluttergram power array."""
    pid = product_id.upper()
    if pid in _clutter_caches and _clutter_caches[pid] is not None:
        return _clutter_caches[pid]

    info = _resolve_cluttergram(pid)
    if info is None:
        return None

    cache_dir = os.path.join(SHARAD_HR_DIR, ".cache", pid)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "cluttergram.npy")

    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        _clutter_caches[pid] = arr
        return arr

    # Read combined section from sim.img
    sim_path = info["sim_path"]
    lines = info["lines"]
    samples = info["samples"]
    offset = info["combined_offset"]
    n_floats = lines * samples

    try:
        with open(sim_path, "rb") as f:
            f.seek(offset)
            raw = f.read(n_floats * 4)
        arr = np.frombuffer(raw, dtype="<f4").reshape(lines, samples)
        np.save(cache_path, arr)
        arr = np.load(cache_path, mmap_mode="r")
        _clutter_caches[pid] = arr
        return arr
    except Exception as e:
        print(f"Error loading cluttergram: {e}")
        return None


def _compute_trace_mapping(
    clutter_info: dict,
    rdr_geometry: dict,
    rdr_total_traces: int,
) -> np.ndarray:
    """Map each cluttergram trace to the nearest RDR trace by lat/lon proximity.

    Uses linearly interpolated cluttergram coordinates from XML start/stop
    and KD-tree nearest-neighbor lookup against RDR per-trace coordinates.

    Returns int array of shape (n_clutter_traces,) with RDR trace indices.
    Falls back to simple linear mapping if cluttergram coordinates unavailable.
    """
    n_clutter_traces = clutter_info["samples"]

    start_lat = clutter_info.get("start_lat")
    start_lon = clutter_info.get("start_lon")
    stop_lat = clutter_info.get("stop_lat")
    stop_lon = clutter_info.get("stop_lon")

    if start_lat is None or start_lon is None:
        # Fallback: simple linear trace ratio
        trace_ratio = rdr_total_traces / n_clutter_traces
        return np.clip(
            (np.arange(n_clutter_traces) * trace_ratio).astype(int),
            0, rdr_total_traces - 1,
        )

    # Interpolate cluttergram trace coordinates
    clutter_lats = np.linspace(start_lat, stop_lat, n_clutter_traces)
    # Handle potential anti-meridian crossing by taking the shortest path.
    dlon = stop_lon - start_lon
    if dlon > 180:
        stop_lon -= 360
    elif dlon < -180:
        stop_lon += 360
    clutter_lons = np.linspace(start_lon, stop_lon, n_clutter_traces)

    # KD-tree nearest-neighbor against RDR per-trace coordinates
    from scipy.spatial import cKDTree
    rdr_lons = np.asarray(rdr_geometry["lon"], dtype=np.float64)
    # Unwrap longitudes so spatial distance behaves continuously near 0/360 seam.
    rdr_lons = np.degrees(np.unwrap(np.radians(rdr_lons)))
    clutter_lons = np.degrees(np.unwrap(np.radians(clutter_lons)))
    rdr_coords = np.column_stack([rdr_geometry["lat"], rdr_lons])
    tree = cKDTree(rdr_coords)

    clutter_coords = np.column_stack([clutter_lats, clutter_lons])
    _, indices = tree.query(clutter_coords)

    return indices.astype(np.int32)


def _project_clutter_to_rdr_axis(
    clutter_aligned: np.ndarray,
    trace_map: np.ndarray,
    rdr_total_traces: int,
) -> np.ndarray:
    """Project clutter traces to the full RDR trace axis using mapping-aware interpolation."""
    if clutter_aligned.shape[1] == rdr_total_traces:
        return clutter_aligned.astype(np.float32, copy=False)

    tm = np.clip(np.asarray(trace_map, dtype=np.int32), 0, rdr_total_traces - 1)
    order = np.argsort(tm, kind="mergesort")
    tm_sorted = tm[order]
    vals_sorted = clutter_aligned[:, order].astype(np.float32, copy=False)

    uniq_tm, starts, counts = np.unique(
        tm_sorted, return_index=True, return_counts=True
    )
    if uniq_tm.size == 0:
        return np.zeros((clutter_aligned.shape[0], rdr_total_traces), dtype=np.float32)

    # Aggregate duplicate mappings before interpolation.
    vals_collapsed = np.empty((vals_sorted.shape[0], uniq_tm.size), dtype=np.float32)
    for j, (s_idx, cnt) in enumerate(zip(starts, counts)):
        if cnt == 1:
            vals_collapsed[:, j] = vals_sorted[:, s_idx]
        else:
            vals_collapsed[:, j] = vals_sorted[:, s_idx : s_idx + cnt].mean(axis=1)

    if uniq_tm.size == 1:
        return np.repeat(vals_collapsed[:, :1], rdr_total_traces, axis=1)

    x = np.arange(rdr_total_traces, dtype=np.float32)
    xp = uniq_tm.astype(np.float32)
    out = np.empty((vals_collapsed.shape[0], rdr_total_traces), dtype=np.float32)
    for r in range(vals_collapsed.shape[0]):
        out[r] = np.interp(
            x,
            xp,
            vals_collapsed[r],
            left=float(vals_collapsed[r, 0]),
            right=float(vals_collapsed[r, -1]),
        ).astype(np.float32)
    return out


def _get_aligned_cluttergram(product_id: str) -> np.ndarray | None:
    """Load cluttergram aligned to RDR bin space using coordinate-based trace
    mapping and robust median-anchored vertical alignment.

    Horizontal: cluttergram traces are mapped to RDR traces by interpolating
    the XML start/stop coordinates and KD-tree nearest-neighbor lookup.

    Vertical: smoothed cluttergram surface peaks (argmax + median filter) are
    shifted to match the RDR's median surface bin position — avoids unreliable
    per-trace RDR surface detection.

    Result shape: (667, rdr_total_traces), cached as cluttergram_aligned_v3.npy.
    """
    pid = product_id.upper()

    info = _resolve_product(product_id)
    cache_dir = info["cache_dir"]
    aligned_path = os.path.join(cache_dir, "cluttergram_aligned_v3.npy")

    if os.path.exists(aligned_path):
        return np.load(aligned_path, mmap_mode="r")

    # Load raw cluttergram
    clutter_info = _resolve_cluttergram(product_id)
    if clutter_info is None:
        return None

    clutter = _get_cluttergram(product_id)
    if clutter is None:
        return None

    clutter_bins, n_clutter_traces = clutter.shape

    # --- Horizontal mapping: coordinate-based trace mapping ---
    geom, rdr_total_traces = _get_geometry(product_id)
    trace_map = _compute_trace_mapping(clutter_info, geom, rdr_total_traces)

    # --- Vertical alignment: use surface-anchored extraction when clutter has deep range ---
    if clutter_bins > N_RANGE_BINS:
        clutter_peaks = np.argmax(clutter, axis=0)
        from scipy.ndimage import median_filter

        kernel = min(
            31,
            n_clutter_traces if n_clutter_traces % 2 == 1 else n_clutter_traces - 1,
        )
        kernel = max(kernel, 1)
        smooth_peaks = median_filter(clutter_peaks, size=kernel).astype(np.int32)

        power, _ = _get_power(product_id)
        mapped_rdr_surface = np.argmax(power[trace_map, :200], axis=1)
        rdr_surface_median = int(np.median(mapped_rdr_surface))
        if rdr_surface_median < 0:
            rdr_surface_median = int(N_RANGE_BINS * 0.22)

        per_trace_offset = smooth_peaks - rdr_surface_median
        median_offset = int(np.median(per_trace_offset))
        median_offset = max(0, min(median_offset, clutter_bins - N_RANGE_BINS))
        sane = (per_trace_offset >= 0) & (per_trace_offset + N_RANGE_BINS <= clutter_bins)
        if not sane.all():
            per_trace_offset = np.where(sane, per_trace_offset, median_offset)

        row_grid = np.arange(N_RANGE_BINS)[:, None] + per_trace_offset[None, :]
        valid = (row_grid >= 0) & (row_grid < clutter_bins)
        row_clipped = np.clip(row_grid, 0, clutter_bins - 1)
        col_grid = np.arange(n_clutter_traces)[None, :]
        aligned_sparse = np.where(
            valid, clutter[row_clipped, col_grid], 0.0
        ).astype(np.float32)

        peak_med = int(np.median(smooth_peaks))
        offset_min = int(per_trace_offset.min())
        offset_max = int(per_trace_offset.max())
    else:
        aligned_sparse = np.ascontiguousarray(clutter, dtype=np.float32)
        peak_med = int(np.median(np.argmax(clutter, axis=0)))
        offset_min = 0
        offset_max = 0
        rdr_surface_median = int(np.median(np.argmax(aligned_sparse, axis=0)))

    # Project to full RDR along-track axis so clutter and radargram share the same x-axis.
    aligned = _project_clutter_to_rdr_axis(aligned_sparse, trace_map, rdr_total_traces)

    np.save(aligned_path, aligned)
    aligned = np.load(aligned_path, mmap_mode="r")

    print(
        f"[SHARAD] Coord-aligned cluttergram for {pid}: "
        f"smooth_peak median={peak_med}, "
        f"rdr_surface median={rdr_surface_median}, "
        f"offset range=[{offset_min}, {offset_max}], "
        f"{clutter_bins}->{N_RANGE_BINS} bins, "
        f"trace mapping: {n_clutter_traces} clutter -> {rdr_total_traces} RDR"
    )

    return aligned


def _render_cluttergram_png(
    clutter: np.ndarray,
    downsample: int = 1,
    pmin: float = 1.0,
    pmax: float = 99.0,
    global_vmin: float | None = None,
    global_vmax: float | None = None,
) -> bytes:
    """Render cluttergram as a grayscale PNG (log-scaled).

    Clutter simulation values are very small floats (~1e-13).
    Uses adaptive floor based on actual data range.
    global_vmin/global_vmax: if provided, use for normalization (for consistent tiles).
    """
    clutter = np.asarray(clutter, dtype=np.float32)
    clutter = np.nan_to_num(clutter, nan=0.0, posinf=0.0, neginf=0.0)

    # Cluttergram layout: (range_bins, traces) — downsample along axis 1 (traces)
    if downsample > 1:
        n = clutter.shape[1] // downsample * downsample
        if n == 0:
            # Degenerate slice; keep a single empty-like column to avoid render failure.
            p = np.zeros((clutter.shape[0], 1), dtype=np.float32)
        else:
            p = clutter[:, :n].reshape(clutter.shape[0], -1, downsample).mean(axis=2)
    else:
        p = np.ascontiguousarray(clutter).astype(np.float32)

    # Adaptive floor: use the smallest positive value as reference
    pos_vals = p[p > 0]
    if pos_vals.size > 0:
        floor = float(pos_vals.min()) * 0.1
    else:
        floor = 1e-30
    p = np.log10(np.maximum(p, floor))

    finite_vals = p[np.isfinite(p)]
    if finite_vals.size == 0:
        finite_vals = np.array([0.0], dtype=np.float32)

    if global_vmin is not None and global_vmax is not None:
        vmin, vmax = float(global_vmin), float(global_vmax)
    else:
        vmin = float(np.percentile(finite_vals, pmin))
        vmax = float(np.percentile(finite_vals, pmax))
    if not np.isfinite(vmin):
        vmin = float(finite_vals.min())
    if not np.isfinite(vmax):
        vmax = float(finite_vals.max())
    if vmax <= vmin:
        vmax = vmin + 1

    gray = ((p - vmin) / (vmax - vmin) * 180).clip(0, 180).astype(np.uint8)

    pil_img = Image.fromarray(gray, mode="L")
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
        lats_graphic = _centric_to_graphic(geom["lat"])
        return JSONResponse(content={
            "product_id": product_id.upper(),
            "instrument": "SHARAD_HIGHRES",
            "rows": total_rows,
            "range_bins": N_RANGE_BINS,
            "row_bytes": ROW_BYTES,
            "sample_interval_us": SHARAD_SAMPLE_INTERVAL_US,
            "lat_range": [float(lats_graphic.min()), float(lats_graphic.max())],
            "lon_range": [float(lons180.min()), float(lons180.max())],
            "start_lat": float(lats_graphic[0]),
            "start_lon": float(lons180[0]),
            "stop_lat": float(lats_graphic[-1]),
            "stop_lon": float(lons180[-1]),
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
    downsample: int = Query(1, ge=1, le=500, description="Along-track downsample factor"),
    log: int = Query(1, ge=0, le=1, description="Log10 scale (1=yes, 0=no)"),
    pmin: float = Query(1.0, ge=0, le=50, description="Lower percentile clip"),
    pmax: float = Query(99.0, ge=50, le=100, description="Upper percentile clip"),
    amplitude: int = Query(1, ge=0, le=1, description="Amplitude mode (1=sqrt(power), 0=power)"),
    per_trace: int = Query(1, ge=0, le=1, description="Per-trace normalization (1=yes, 0=global)"),
    start_trace: int = Query(0, ge=0, description="First trace index (full-res, for tiled rendering)"),
    end_trace: int = Query(-1, description="Last trace index exclusive (-1 = all)"),
):
    """Return radargram as PNG image. Supports tiled rendering via start_trace/end_trace."""
    try:
        power, total_rows = _get_power(product_id)

        # Resolve trace range
        st = max(0, min(start_trace, total_rows))
        et = total_rows if end_trace < 0 else max(st, min(end_trace, total_rows))
        is_tile = (st > 0 or et < total_rows)

        tile_power = power[st:et] if is_tile else power

        # For tiles with global normalization, pre-compute percentiles from full array
        gvmin, gvmax = (None, None)
        if is_tile and not bool(per_trace):
            gvmin, gvmax = _get_global_percentiles(
                product_id, use_log=bool(log), amplitude=bool(amplitude),
                pmin=pmin, pmax=pmax,
            )

        ckey = f"{product_id}_{downsample}_{log}_{pmin}_{pmax}_{amplitude}_{per_trace}_{st}_{et}"
        png = _render_radargram_png(
            tile_power, downsample=downsample, use_log=bool(log),
            pmin=pmin, pmax=pmax,
            amplitude=bool(amplitude), per_trace=bool(per_trace),
            cache_key=ckey,
            global_vmin=gvmin, global_vmax=gvmax,
        )
        n_traces_out = tile_power.shape[0] // max(downsample, 1)
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "X-Traces": str(n_traces_out),
                "X-Range-Bins": str(N_RANGE_BINS),
                "X-Downsample": str(downsample),
                "X-Start-Trace": str(st),
                "X-End-Trace": str(et),
                "X-Total-Traces": str(total_rows),
                "Cache-Control": "public, max-age=3600",
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
        lats = _centric_to_graphic(geom["lat"][indices])
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
            "lat": round(_centric_to_graphic_scalar(float(geom["lat"][full_idx])), 4),
            "lon": round(float(lons180[full_idx]), 4),
        })
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── Cluttergram endpoints ──────────────────────────────

@router.get("/cluttergram")
async def get_cluttergram(
    product_id: str = Query(..., description="Product ID (e.g. R_0277201_001_SS19_700_A)"),
    downsample: int = Query(50, ge=1, le=500, description="Along-track downsample factor"),
    pmin: float = Query(1.0, ge=0, le=50, description="Lower percentile clip"),
    pmax: float = Query(99.0, ge=50, le=100, description="Upper percentile clip"),
    start_trace: int = Query(0, ge=0, description="First trace index (in RDR space, for tiled rendering)"),
    end_trace: int = Query(-1, description="Last trace index exclusive (-1 = all)"),
):
    """Return surface clutter simulation aligned to the RDR trace/range axes as PNG."""
    try:
        # Use aligned cluttergram (667 bins, surface-matched to RDR)
        clutter = _get_aligned_cluttergram(product_id)
        if clutter is None:
            return JSONResponse(
                content={"error": f"No cluttergram available for {product_id}"},
                status_code=404,
            )

        # Slice to requested trace range (in RDR trace space)
        total_clutter_traces = clutter.shape[1]
        st = max(0, min(start_trace, total_clutter_traces))
        et = total_clutter_traces if end_trace < 0 else max(st, min(end_trace, total_clutter_traces))
        if st > 0 or et < total_clutter_traces:
            clutter = clutter[:, st:et]

        png = _render_cluttergram_png(clutter, downsample=downsample, pmin=pmin, pmax=pmax)
        n_traces_out = clutter.shape[1] // max(downsample, 1)
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "X-Traces": str(n_traces_out),
                "X-Range-Bins": str(clutter.shape[0]),
                "X-Downsample": str(downsample),
                "X-Start-Trace": str(st),
                "X-End-Trace": str(et),
                "X-Total-Traces": str(total_clutter_traces),
                "X-Source": "aligned_clutter_simulation",
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/cluttergram_meta")
async def get_cluttergram_meta(
    product_id: str = Query(..., description="Product ID"),
):
    """Return cluttergram metadata (availability, dimensions)."""
    info = _resolve_cluttergram(product_id)
    if info is None:
        return JSONResponse(content={"available": False})

    aligned = _get_aligned_cluttergram(product_id)
    raw = _get_cluttergram(product_id)
    return JSONResponse(content={
        "available": True,
        "raw_range_bins": info["lines"],
        "raw_traces": info["samples"],
        "range_bins": aligned.shape[0] if aligned is not None else info["lines"],
        "total_traces": aligned.shape[1] if aligned is not None else info["samples"],
        "aligned": aligned is not None and aligned.shape[0] == N_RANGE_BINS,
    })


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
        lats = _centric_to_graphic(geom["lat"][indices])
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


def _haversine_vec(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance on Mars (metres), vectorized with numpy."""
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2) ** 2
    seg_dist = MARS_MEAN_RADIUS * 2 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
    return np.concatenate([[0.0], np.cumsum(seg_dist)])


_mola_cache: LRUCache = LRUCache(maxsize=30)  # key: f"{product_id}_{downsample}" -> response dict


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
    cache_key = f"{product_id.upper()}_{downsample}"
    if cache_key in _mola_cache:
        return JSONResponse(content=_mola_cache[cache_key])

    try:
        from rasterio.windows import Window

        geom, total_rows = _get_geometry(product_id)
        ds = _get_dem()

        n_traces = total_rows // downsample
        indices = np.arange(0, total_rows, downsample)[:n_traces]
        lats_centric = geom["lat"][indices]
        # Convert 0-360 → -180..180 to match DEM coordinate system
        lons = _lon_to_180(geom["lon"][indices])

        # Convert lat/lon to pixel coordinates (DEM is planetocentric)
        inv_transform = ~ds.transform
        cols, rows = inv_transform * (lons, lats_centric)
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

        # Cumulative along-track distance (vectorized, uses planetocentric for accuracy)
        distances_m = _haversine_vec(lats_centric, lons)

        # Convert to planetographic for frontend/Cesium display
        lats_graphic = _centric_to_graphic(lats_centric)

        result = {
            "distance_km": [round(d / 1000.0, 3) for d in distances_m],
            "elevation_m": [round(float(e), 1) if not np.isnan(e) else None for e in elevations],
            "lats": [round(float(v), 4) for v in lats_graphic],
            "lons": [round(float(v), 4) for v in lons],
            "n_traces": n_traces,
            "downsample": downsample,
            "total_distance_km": round(float(distances_m[-1] / 1000.0), 3) if n_traces > 0 else 0,
        }

        # Cache result (bounded by LRUCache maxsize=30)
        _mola_cache[cache_key] = result

        return JSONResponse(content=result)
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
