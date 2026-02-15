import os
import re
import io
import numpy as np
from PIL import Image

# ── RDR binary layout constants ────────────────────────
ROW_BYTES = 5822
N_RANGE_BINS = 667

# Byte offsets (0-indexed) and sizes for fields we need
ECHO_REAL_OFFSET = 194      # start_byte 195
ECHO_IMAG_OFFSET = 2862     # start_byte 2863
ECHO_BYTES = N_RANGE_BINS * 4  # 2668

LON_OFFSET = 5637            # SUB_SC_EAST_LONGITUDE, float64
LAT_OFFSET = 5645            # SUB_SC_PLANETOCENTRIC_LATITUDE, float64
ALT_OFFSET = 5629            # SPACECRAFT_ALTITUDE, float64

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

def parse_sharad_data(dat_path: str, lbl_path: str):
    """
    Parses SHARAD .dat and .lbl files to extract power and geometry data.
    Returns (power_array, geometry_dict).
    """
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"SHARAD RDR .dat not found: {dat_path}")

    total_rows = _parse_total_rows(lbl_path, dat_path)

    # Vectorized parsing: read entire file as raw bytes, then extract fields
    raw = np.fromfile(dat_path, dtype=np.uint8)
    actual_rows = len(raw) // ROW_BYTES
    if actual_rows < total_rows:
        total_rows = actual_rows
    raw = raw[:total_rows * ROW_BYTES].reshape(total_rows, ROW_BYTES)

    # Extract real/imag echo columns
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

    geometry = {"lat": lats, "lon": lons, "alt": alts}

    return power, geometry

def render_radargram_png(
    power: np.ndarray,
    downsample: int = 1,
    use_log: bool = True,
    pmin: float = 1.0,
    pmax: float = 99.0,
    amplitude: bool = False,
    per_trace: bool = False,
) -> bytes:
    """Render the radargram as a grayscale PNG."""
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
        # Per-trace normalization
        vmins = np.percentile(p, pmin, axis=1, keepdims=True)
        vmaxs = np.percentile(p, pmax, axis=1, keepdims=True)
        vmaxs = np.where(vmaxs <= vmins, vmins + 1, vmaxs)
        img = ((p - vmins) / (vmaxs - vmins) * 255).clip(0, 255).astype(np.uint8)
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
    return buf.getvalue()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View SHARAD radargram from .dat and .lbl files.")
    parser.add_argument("dat_file", help="Path to the SHARAD .dat file.")
    parser.add_argument("lbl_file", help="Path to the SHARAD .lbl file.")
    parser.add_argument("--output", "-o", default="radargram.png",
                        help="Output PNG file path (default: radargram.png).")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Along-track downsample factor (default: 1).")
    parser.add_argument("--no-log", action="store_true",
                        help="Do not apply log10 scaling to power data.")
    parser.add_argument("--pmin", type=float, default=1.0,
                        help="Lower percentile clip (default: 1.0).")
    parser.add_argument("--pmax", type=float, default=99.0,
                        help="Upper percentile clip (default: 99.0).")
    parser.add_argument("--amplitude", action="store_true",
                        help="Use amplitude (sqrt(power)) instead of raw power.")
    parser.add_argument("--per-trace", action="store_true",
                        help="Apply normalization per trace instead of globally.")

    args = parser.parse_args()

    try:
        print(f"Parsing SHARAD data from {args.dat_file} and {args.lbl_file}...")
        power_data, geometry_data = parse_sharad_data(args.dat_file, args.lbl_file)
        print(f"Found {power_data.shape[0]} traces and {power_data.shape[1]} range bins.")

        print(f"Rendering radargram to {args.output}...")
        png_bytes = render_radargram_png(
            power_data,
            downsample=args.downsample,
            use_log=not args.no_log,
            pmin=args.pmin,
            pmax=args.pmax,
            amplitude=args.amplitude,
            per_trace=args.per_trace,
        )

        with open(args.output, "wb") as f:
            f.write(png_bytes)
        print(f"Radargram saved to {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
