#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_crism_quickviews.py
- CRISM MTR3 PDS3 (.img/.lbl) quickview builder
- NO rasterio / NO GDAL dependency
- Reads raw PDS3 IMG via numpy memmap using LBL metadata

Input:
  ./crism_data/<product>.img
  ./crism_data/<product>.lbl

Output:
  ./crism_quickview/<product>.jpg

Usage:
  python scripts/build_crism_quickviews.py
"""

import os
import re
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
from PIL import Image

# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)

CRISM_DATA_DIR = os.path.join(BACKEND_DIR, "crism_data")
OUT_DIR = os.path.join(BACKEND_DIR, "crism_quickview")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# LBL parsing (PDS3)
# =========================================================
_keyval_re = re.compile(r"^\s*([A-Z0-9_:\-]+)\s*=\s*(.+?)\s*$")


def _strip_comment(line: str) -> str:
    # PDS3 labels often use /* ... */ comments; handle single-line cases
    # We'll remove trailing /* ... */ if present.
    if "/*" in line:
        line = line.split("/*", 1)[0]
    return line.strip()


def _parse_value(raw: str) -> Any:
    raw = raw.strip()

    # quoted string
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]

    # tuple like: (a, b) or ("file", 123)
    if raw.startswith("(") and raw.endswith(")"):
        inner = raw[1:-1].strip()
        # split by comma not inside quotes (simple but ok for our use)
        parts: List[str] = []
        buf = ""
        in_quote = False
        quote_char = ""
        for ch in inner:
            if ch in ("'", '"'):
                if not in_quote:
                    in_quote = True
                    quote_char = ch
                elif quote_char == ch:
                    in_quote = False
            if ch == "," and not in_quote:
                parts.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return tuple(_parse_value(p) for p in parts)

    # integer
    if re.fullmatch(r"[-+]?\d+", raw):
        try:
            return int(raw)
        except Exception:
            pass

    # float
    if re.fullmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", raw) or re.fullmatch(
        r"[-+]?\d+([eE][-+]?\d+)", raw
    ):
        try:
            return float(raw)
        except Exception:
            pass

    return raw


def parse_pds3_lbl(lbl_path: str) -> Dict[str, Any]:
    """
    Parse a PDS3 .lbl as a flat dict of KEY=VALUE lines.
    Enough for:
      - RECORD_BYTES
      - ^IMAGE pointer
      - LINES, LINE_SAMPLES, BANDS
      - SAMPLE_BITS, SAMPLE_TYPE
      - BAND_STORAGE_TYPE or INTERCHANGE_FORMAT (for interleave)
    """
    meta: Dict[str, Any] = {}
    with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = _strip_comment(line)
            if not line:
                continue
            m = _keyval_re.match(line)
            if not m:
                continue
            k = m.group(1).strip().upper()
            v = _parse_value(m.group(2).strip())
            meta[k] = v
    return meta


# =========================================================
# PDS3 image reading helpers
# =========================================================
def _infer_dtype(sample_type: str, sample_bits: int) -> np.dtype:
    st = (sample_type or "").upper()

    # Common PDS3 types:
    #   "MSB_INTEGER", "LSB_INTEGER", "MSB_UNSIGNED_INTEGER", "LSB_UNSIGNED_INTEGER"
    #   "IEEE_REAL", "PC_REAL"
    # CRISM MTR3 often float32
    if "REAL" in st:
        if sample_bits == 32:
            return np.dtype(">f4") if "MSB" in st else np.dtype("<f4")
        if sample_bits == 64:
            return np.dtype(">f8") if "MSB" in st else np.dtype("<f8")
        # fallback
        return np.float32

    signed = "UNSIGNED" not in st
    if sample_bits == 8:
        return np.int8 if signed else np.uint8
    if sample_bits == 16:
        if "MSB" in st:
            return np.dtype(">i2") if signed else np.dtype(">u2")
        if "LSB" in st:
            return np.dtype("<i2") if signed else np.dtype("<u2")
        return np.int16 if signed else np.uint16
    if sample_bits == 32:
        if "MSB" in st:
            return np.dtype(">i4") if signed else np.dtype(">u4")
        if "LSB" in st:
            return np.dtype("<i4") if signed else np.dtype("<u4")
        return np.int32 if signed else np.uint32

    # fallback
    return np.float32


def _get_image_offset_bytes(meta: Dict[str, Any]) -> int:
    """
    PDS3 pointer convention:
      ^IMAGE = <record_number>
      ^IMAGE = ("filename", <record_number>)
    record_number is 1-based record index.
    offset_bytes = (record_number - 1) * RECORD_BYTES

    If not found, return 0.
    """
    rec_bytes = meta.get("RECORD_BYTES", None)
    if rec_bytes is None:
        rec_bytes = meta.get("RECORD_BYTE", None)
    if rec_bytes is None:
        rec_bytes = meta.get("RECORD_BYTES".upper(), None)

    try:
        rec_bytes_int = int(rec_bytes)
    except Exception:
        rec_bytes_int = 0

    ptr = meta.get("^IMAGE", None) or meta.get("^IMAGE".upper(), None)
    if ptr is None:
        return 0

    record_number: Optional[int] = None

    if isinstance(ptr, int):
        record_number = ptr
    elif isinstance(ptr, tuple) and len(ptr) >= 2:
        # ("file", 123)
        try:
            record_number = int(ptr[1])
        except Exception:
            record_number = None
    elif isinstance(ptr, str):
        # sometimes it is just "123"
        try:
            record_number = int(ptr)
        except Exception:
            record_number = None

    if record_number is None or rec_bytes_int <= 0:
        return 0

    return max(0, (record_number - 1) * rec_bytes_int)


def _infer_interleave(meta: Dict[str, Any]) -> str:
    """
    CRISM products are usually BSQ, but we support BIL/BIP if label says so.
    Priority:
      BAND_STORAGE_TYPE
      INTERCHANGE_FORMAT (rare)
    """
    bst = meta.get("BAND_STORAGE_TYPE", None)
    if isinstance(bst, str):
        s = bst.upper()
        if "BSQ" in s:
            return "BSQ"
        if "BIL" in s:
            return "BIL"
        if "BIP" in s:
            return "BIP"

    icf = meta.get("INTERCHANGE_FORMAT", None)
    if isinstance(icf, str):
        s = icf.upper()
        if "BSQ" in s:
            return "BSQ"
        if "BIL" in s:
            return "BIL"
        if "BIP" in s:
            return "BIP"

    # default safe assumption for CRISM cubes
    return "BSQ"


def _get_cube_shape(meta: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Return (bands, lines, samples)
    """
    # Standard PDS keys for cubes
    bands = int(meta.get("BANDS"))
    lines = int(meta.get("LINES"))
    samples = int(meta.get("LINE_SAMPLES"))
    return bands, lines, samples


def open_crism_cube_memmap(img_path: str, lbl_meta: Dict[str, Any]) -> Tuple[np.memmap, str]:
    """
    Returns:
      memmap array (raw layout depending on interleave)
      interleave: 'BSQ'|'BIL'|'BIP'
    """
    interleave = _infer_interleave(lbl_meta)

    sample_bits = int(lbl_meta.get("SAMPLE_BITS", 32))
    sample_type = str(lbl_meta.get("SAMPLE_TYPE", "IEEE_REAL"))

    dtype = _infer_dtype(sample_type, sample_bits)
    offset = _get_image_offset_bytes(lbl_meta)

    bands, lines, samples = _get_cube_shape(lbl_meta)

    if interleave == "BSQ":
        shape = (bands, lines, samples)
    elif interleave == "BIL":
        shape = (lines, bands, samples)
    else:  # BIP
        shape = (lines, samples, bands)

    arr = np.memmap(img_path, dtype=dtype, mode="r", offset=offset, shape=shape)
    return arr, interleave


def extract_band(cube: np.memmap, interleave: str, band_index: int) -> np.ndarray:
    """
    Returns 2D band image as float32 array (lines, samples)
    """
    if interleave == "BSQ":
        band = cube[band_index, :, :]
    elif interleave == "BIL":
        band = cube[:, band_index, :]
    else:  # BIP
        band = cube[:, :, band_index]

    # ensure contiguous float32
    return np.array(band, dtype=np.float32, copy=False)


# =========================================================
# Image scaling for quickview
# =========================================================
def robust_scale_to_uint8(img: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """
    Robust contrast stretch using percentiles -> uint8
    """
    x = img.astype(np.float32, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(img))
        hi = float(np.nanmax(img))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    y = (img - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def downsample_max(img_u8: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Downsample to keep largest dimension <= max_size.
    Uses simple slicing for speed (not fancy resampling).
    """
    h, w = img_u8.shape[:2]
    scale = max(h, w) / float(max_size)
    if scale <= 1.0:
        return img_u8
    step = int(np.ceil(scale))
    return img_u8[::step, ::step]


# =========================================================
# Main: build quickviews
# =========================================================
def find_products(crism_dir: str) -> List[str]:
    """
    returns product base names that have both .img and .lbl
    """
    files = os.listdir(crism_dir)
    imgs = {f[:-4] for f in files if f.lower().endswith(".img")}
    lbls = {f[:-4] for f in files if f.lower().endswith(".lbl")}
    products = sorted(list(imgs & lbls))
    return products


def build_quickview(product: str) -> None:
    img_path = os.path.join(CRISM_DATA_DIR, f"{product}.img")
    lbl_path = os.path.join(CRISM_DATA_DIR, f"{product}.lbl")
    out_path = os.path.join(OUT_DIR, f"{product}.jpg")

    print(f"[CRISM] processing {product}")

    meta = parse_pds3_lbl(lbl_path)

    cube, interleave = open_crism_cube_memmap(img_path, meta)
    bands, lines, samples = _get_cube_shape(meta)

    # 대표 밴드 선택: 중앙 밴드(대부분 안전)
    band_index = 10
    band = extract_band(cube, interleave, band_index=band_index)

    # scale -> uint8
    img_u8 = robust_scale_to_uint8(band, 2, 98)
    img_u8 = downsample_max(img_u8, max_size=1024)

    im = Image.fromarray(img_u8, mode="L")
    im.save(out_path, quality=90, optimize=True)

    print(f"  -> saved {out_path} (band={band_index}, interleave={interleave}, shape={lines}x{samples}, bands={bands})")


def main():
    if not os.path.isdir(CRISM_DATA_DIR):
        raise RuntimeError(f"CRISM_DATA_DIR not found: {CRISM_DATA_DIR}")

    products = find_products(CRISM_DATA_DIR)
    if not products:
        print("[CRISM] no products found (need .img + .lbl pairs) in:", CRISM_DATA_DIR)
        return

    for p in products:
        try:
            build_quickview(p)
        except Exception as e:
            print(f"[CRISM][ERROR] {p}: {e}")


if __name__ == "__main__":
    main()
