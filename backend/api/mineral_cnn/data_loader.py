"""
CNN Mineral Classification — Data Loading

PDS3 binary data loaders for TRR3 cubes, DDR incidence angle, and VS ADR
atmospheric reference files. Adapted from inference_pipeline.py.
"""

import os
import math
import glob
from datetime import datetime

import numpy as np

from .constants import (
    IR_COUNT, FILL_VALUE, TRR_DATA_DIR,
    ADR_IMG_PATH, JCAT_AUTOMATION_DIR,
)

# ============================================================
# ADR directory (for time-based ADR selection)
# ============================================================
_ADR_DIR = os.path.dirname(ADR_IMG_PATH)


# ============================================================
# PDS3 Label Parser
# ============================================================
def parse_pds3_label(path: str) -> dict:
    """Parse PDS3 label file into key-value dict."""
    meta = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("/*") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta


# ============================================================
# TRR3 Cube Loading
# ============================================================
def load_trr_cube(trr_img: str, trr_lbl: str) -> tuple:
    """
    Load TRR3 image cube from PDS3 binary format.
    Returns (cube[rows, cols, bands], rows, cols).
    """
    meta = parse_pds3_label(trr_lbl)
    rows = int(meta["LINES"])
    cols = int(meta["LINE_SAMPLES"])
    bands = int(meta["BANDS"])

    raw = np.fromfile(trr_img, dtype=np.float32)
    expected = rows * cols * bands

    if raw.size < expected:
        available_rows = raw.size // (cols * bands)
        rows = available_rows
        expected = rows * cols * bands

    raw = raw[:expected]
    cube_lbc = raw.reshape(rows, bands, cols)
    cube_rcb = np.transpose(cube_lbc, (0, 2, 1))

    return cube_rcb, rows, cols


# ============================================================
# DDR Incidence Angle Loading
# ============================================================
def load_ddr_ina(ddr_img: str, ddr_lbl: str) -> float:
    """Load incidence angle from DDR, returns radians."""
    meta = parse_pds3_label(ddr_lbl)
    rows = int(meta["LINES"])
    cols = int(meta["LINE_SAMPLES"])
    bands = int(meta["BANDS"])

    raw = np.fromfile(ddr_img, dtype=np.float32)
    raw = raw[:bands * rows * cols]
    cube = raw.reshape(bands, rows, cols)

    # Band 0 is INA (incidence angle)
    ina_deg = float(cube[0, rows // 2, cols // 2])

    if ina_deg >= 65534 or not np.isfinite(ina_deg):
        center = cube[0, rows // 2 - 5:rows // 2 + 5, cols // 2 - 5:cols // 2 + 5]
        valid = center[(center < 65534) & np.isfinite(center)]
        ina_deg = float(np.median(valid)) if len(valid) > 0 else 30.0

    return math.radians(ina_deg)


# ============================================================
# VS ADR Loading
# ============================================================
def load_vs_adr(adr_img: str, ir_count: int) -> tuple:
    """Load VS ADR for atmospheric correction. Returns (vstrans, vsart)."""
    adr_bands = 438
    raw = np.fromfile(adr_img, dtype=np.float32)
    total_records = raw.size // 640

    if total_records == adr_bands + 1:
        lines = 1
    elif total_records == adr_bands * 3 + 1:
        lines = 3
    else:
        raise ValueError(f"Unexpected ADR layout: {total_records} records")

    raw = raw[:adr_bands * lines * 640]
    cube = raw.reshape(lines, adr_bands, 640)

    mid = 640 // 2
    vstrans = cube[0, :ir_count, mid]
    vsart = cube[1, :ir_count, mid] if lines > 1 else np.zeros(ir_count)

    return vstrans.astype(np.float32), vsart.astype(np.float32)


# ============================================================
# TRR3/DDR File Resolution
# ============================================================
def resolve_trr_files(obs_id: str) -> dict:
    """
    Resolve TRR3 + DDR file paths for a CRISM observation.

    Expected layout:
      mineral_cnn_data/{obs_id}_07/  (or {obs_id}_01/, {obs_id}_03/, etc.)
        {obs_id}_??_if???l_trr3.img  +  .lbl
        {obs_id}_??_de???l_ddr1.img  +  .lbl

    Returns dict with keys: trr_img, trr_lbl, ddr_img, ddr_lbl
    """
    obs_dir = os.path.join(TRR_DATA_DIR, f"{obs_id}_07")
    if not os.path.isdir(obs_dir):
        # Try any suffix: {obs_id}_01, _03, _05, etc.
        candidates = glob.glob(os.path.join(TRR_DATA_DIR, f"{obs_id}_*"))
        dirs = [c for c in candidates if os.path.isdir(c)]
        if dirs:
            obs_dir = dirs[0]  # Use first match
        else:
            # Also try without suffix
            obs_dir = os.path.join(TRR_DATA_DIR, obs_id)
            if not os.path.isdir(obs_dir):
                raise FileNotFoundError(
                    f"TRR data directory not found. Expected: "
                    f"{TRR_DATA_DIR}/{obs_id}_*/"
                )

    # Case-insensitive file matching (Linux is case-sensitive but PDS filenames vary)
    all_files = os.listdir(obs_dir)
    trr_imgs = [os.path.join(obs_dir, f) for f in all_files if f.upper().endswith("_TRR3.IMG")]
    trr_lbls = [os.path.join(obs_dir, f) for f in all_files if f.upper().endswith("_TRR3.LBL")]
    ddr_imgs = [os.path.join(obs_dir, f) for f in all_files if f.upper().endswith("_DDR1.IMG")]
    ddr_lbls = [os.path.join(obs_dir, f) for f in all_files if f.upper().endswith("_DDR1.LBL")]

    if not trr_imgs or not trr_lbls:
        raise FileNotFoundError(f"TRR3 files (.img + .lbl) not found in {obs_dir}")
    if not ddr_imgs or not ddr_lbls:
        raise FileNotFoundError(f"DDR files (.img + .lbl) not found in {obs_dir}")

    # Prefer L-sensor I/F data over browse (BRFALL) — L-sensor has 438+ bands for CNN
    def _prefer_if_l(paths: list[str]) -> str:
        for p in paths:
            upper = os.path.basename(p).upper()
            if "_IF" in upper and "L_" in upper:
                return p
        # Fallback: any I/F file, then first file
        for p in paths:
            if "_IF" in os.path.basename(p).upper():
                return p
        return paths[0]

    return {
        "trr_img": _prefer_if_l(trr_imgs),
        "trr_lbl": _prefer_if_l(trr_lbls),
        "ddr_img": ddr_imgs[0],
        "ddr_lbl": ddr_lbls[0],
    }


# ============================================================
# Time-Based ADR Selection
# ============================================================
# Pre-built index of available ADR epochs (partition → start_time, sclk)
_ADR_EPOCHS = [
    {"partition": "1", "sclk": "0000000000", "start_time": datetime(1980, 1, 1)},
    {"partition": "5", "sclk": "0924300803", "start_time": datetime(2009, 4, 15, 22, 13, 1)},
    {"partition": "9", "sclk": "0947778566", "start_time": datetime(2010, 1, 12, 15, 48, 55)},
]

# Default volcano scan column (same as original inference_pipeline.py)
_DEFAULT_VS_COLUMN = "11D87"


def resolve_best_adr(obs_start_time: datetime = None, vs_column: str = None) -> str:
    """
    Select the best ADR file based on observation acquisition time.

    Strategy: pick the latest ADR epoch whose start_time <= observation time.
    Falls back to the newest epoch if no match or no time provided.

    Args:
        obs_start_time: Observation acquisition time (from TRR3 label START_TIME)
        vs_column: Volcano scan column ID (default: 11D87)

    Returns: Absolute path to the selected ADR .IMG file
    """
    col = vs_column or _DEFAULT_VS_COLUMN

    if obs_start_time is None:
        # No time info — use the latest epoch
        epoch = _ADR_EPOCHS[-1]
    else:
        # Find latest epoch with start_time <= observation time
        epoch = _ADR_EPOCHS[0]  # fallback to earliest
        for e in _ADR_EPOCHS:
            if e["start_time"] <= obs_start_time:
                epoch = e

    filename = f"ADR{epoch['partition']}{epoch['sclk']}_{col}_VS00L_8.IMG"
    path = os.path.join(_ADR_DIR, filename)

    if os.path.exists(path):
        return path

    # Fallback: try the hardcoded default
    if os.path.exists(ADR_IMG_PATH):
        return ADR_IMG_PATH

    raise FileNotFoundError(f"ADR file not found: {path}")


def get_obs_start_time(trr_lbl: str) -> datetime:
    """Extract observation start time from TRR3 label."""
    meta = parse_pds3_label(trr_lbl)
    start_str = meta.get("START_TIME", "")
    if not start_str or start_str == "NULL":
        return None
    try:
        # Handle formats like "2009-10-03T08:42:17.123"
        if "T" in start_str:
            # Strip trailing whitespace and possible 'Z'
            start_str = start_str.strip().rstrip("Z")
            return datetime.fromisoformat(start_str)
    except (ValueError, TypeError):
        pass
    return None
