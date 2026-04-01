"""SPICE kernel management: download, meta-kernel generation, and initialization."""

import time
import re
from pathlib import Path

import requests
import spiceypy as spice

from .config import (
    SPICE_DIR,
    NAIF_BASE,
    MAX_RETRIES,
    DOWNLOAD_TIMEOUT,
    CHUNK_SIZE,
)

session = requests.Session()
session.headers.update({"User-Agent": "mastcam-z-coregister/1.0 (research)"})

# PDS4 NAIF archive for Mars 2020 SPICE kernels
M2020_SPICE_BASE = f"{NAIF_BASE}/pds/pds4/mars2020/mars2020_spice/spice_kernels"

# Static kernels (sol-independent)
STATIC_KERNELS = {
    "lsk": {
        "url": f"{M2020_SPICE_BASE}/lsk/naif0012.tls",
        "subdir": "lsk",
    },
    "pck": {
        "url": f"{M2020_SPICE_BASE}/pck/pck00010.tpc",
        "subdir": "pck",
    },
    "spk_planets": {
        "url": f"{M2020_SPICE_BASE}/spk/de438s.bsp",
        "subdir": "spk",
    },
    "m2020_fk": {
        "url": f"{M2020_SPICE_BASE}/fk/m2020_v04.tf",
        "subdir": "fk",
    },
    "m2020_ik": {
        "url": f"{M2020_SPICE_BASE}/ik/m2020_struct_v00.ti",
        "subdir": "ik",
    },
}


def _download(url: str, dest: Path) -> Path:
    """Download a file with retry. Returns path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name}")
        return dest

    print(f"  Downloading {dest.name} ...")
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
            return dest
        except requests.RequestException as e:
            if dest.exists():
                dest.unlink()
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Kernel download failed: {url} ({e})") from e
            time.sleep(2 ** attempt)


def _list_naif_dir(url: str) -> list[str]:
    """List file hrefs in a NAIF HTML directory listing."""
    r = session.get(url, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    # Extract hrefs that look like filenames (not .xml labels)
    all_links = re.findall(r'href="([^"?][^"]*)"', r.text)
    return [l for l in all_links if not l.startswith("/") and not l.startswith("http") and not l.startswith("mailto")]


def download_static_kernels() -> list[Path]:
    """Download all static (sol-independent) SPICE kernels."""
    print("Downloading static SPICE kernels...")
    paths = []
    for name, info in STATIC_KERNELS.items():
        subdir = SPICE_DIR / info["subdir"]
        filename = info["url"].rsplit("/", 1)[-1]
        path = _download(info["url"], subdir / filename)
        paths.append(path)
    return paths


def download_sol_kernels(sol: int) -> list[Path]:
    """Download sol-specific CK, SPK, and SCLK kernels."""
    print(f"Downloading sol-specific kernels for sol {sol}...")
    paths = []

    # SCLK — get latest version
    sclk_url = f"{M2020_SPICE_BASE}/sclk/"
    try:
        sclk_files = _list_naif_dir(sclk_url)
        sclk_matches = sorted([f for f in sclk_files if f.endswith(".tsc") and "refit" in f])
        if not sclk_matches:
            sclk_matches = sorted([f for f in sclk_files if f.endswith(".tsc")])
        if sclk_matches:
            sclk_file = sclk_matches[-1]
            path = _download(sclk_url + sclk_file, SPICE_DIR / "sclk" / sclk_file)
            paths.append(path)
            print(f"  SCLK: {sclk_file}")
    except Exception as e:
        print(f"  WARNING: SCLK download failed: {e}")

    # CK (camera/instrument pointing)
    ck_url = f"{M2020_SPICE_BASE}/ck/"
    try:
        ck_files = _list_naif_dir(ck_url)
        # Rover pointing: m2020_surf_rover_tlm_SSSS_EEEE_vV.bc
        ck_candidates = []
        for f in ck_files:
            m = re.match(r'm2020_surf_rover_tlm_(\d+)_(\d+)_v\d+\.bc', f)
            if m:
                s, e = int(m.group(1)), int(m.group(2))
                if s <= sol <= e:
                    ck_candidates.append(f)

        if ck_candidates:
            ck_file = sorted(ck_candidates)[-1]
            path = _download(ck_url + ck_file, SPICE_DIR / "ck" / ck_file)
            paths.append(path)
            print(f"  CK (rover): {ck_file}")
        else:
            print(f"  WARNING: No rover CK found for sol {sol}")

        # Also get RSM (remote sensing mast) pointing if available
        rsm_candidates = []
        for f in ck_files:
            m = re.match(r'm2020_surf_rsm_tlm_(\d+)_(\d+)_v\d+\.bc', f)
            if m:
                s, e = int(m.group(1)), int(m.group(2))
                if s <= sol <= e:
                    rsm_candidates.append(f)
        if rsm_candidates:
            rsm_file = sorted(rsm_candidates)[-1]
            path = _download(ck_url + rsm_file, SPICE_DIR / "ck" / rsm_file)
            paths.append(path)
            print(f"  CK (RSM): {rsm_file}")

    except Exception as e:
        print(f"  WARNING: CK download failed: {e}")

    # SPK (rover position on Mars surface)
    spk_url = f"{M2020_SPICE_BASE}/spk/"
    try:
        spk_files = _list_naif_dir(spk_url)

        # Rover surface location
        spk_candidates = []
        for f in spk_files:
            m = re.match(r'm2020_surf_rover_loc_(\d+)_(\d+)_v\d+\.bsp', f)
            if m:
                s, e = int(m.group(1)), int(m.group(2))
                if s <= sol <= e:
                    spk_candidates.append(f)

        if spk_candidates:
            spk_file = sorted(spk_candidates)[-1]
            path = _download(spk_url + spk_file, SPICE_DIR / "spk" / spk_file)
            paths.append(path)
            print(f"  SPK (rover): {spk_file}")
        else:
            print(f"  WARNING: No rover SPK found for sol {sol}")

        # Also need the Mars-to-Sun SPK and landing site SPK
        for pattern in [r'm2020_ls_ops.*\.bsp', r'm2020_atls_ops.*\.bsp']:
            matches = sorted([f for f in spk_files if re.match(pattern, f) and f.endswith('.bsp')])
            if matches:
                extra = matches[-1]
                path = _download(spk_url + extra, SPICE_DIR / "spk" / extra)
                paths.append(path)
                print(f"  SPK (extra): {extra}")

    except Exception as e:
        print(f"  WARNING: SPK download failed: {e}")

    return paths


def generate_metakernel(kernel_paths: list[Path]) -> Path:
    """Generate a SPICE meta-kernel (.tm) file.

    Uses PATH_SYMBOLS to keep lines under SPICE's 80-char limit.
    """
    mk_path = SPICE_DIR / "m2020_coregister.tm"

    # Use $K as short alias for SPICE_DIR
    kernel_lines = []
    for p in kernel_paths:
        # Use relative path from SPICE_DIR
        try:
            rel = p.relative_to(SPICE_DIR)
            kernel_lines.append(f"  '$K/{rel}'")
        except ValueError:
            kernel_lines.append(f"  '{p}'")

    kernels_block = "\n".join(kernel_lines)

    content = f"""KPL/MK

\\begindata

  PATH_VALUES = ( '{SPICE_DIR}' )
  PATH_SYMBOLS = ( 'K' )

  KERNELS_TO_LOAD = (
{kernels_block}
  )

\\begintext
"""

    mk_path.parent.mkdir(parents=True, exist_ok=True)
    mk_path.write_text(content)
    print(f"Meta-kernel written: {mk_path}")
    return mk_path


def init_spice(sol: int) -> Path:
    """Full SPICE initialization: download kernels, generate meta-kernel, furnsh."""
    static_paths = download_static_kernels()
    sol_paths = download_sol_kernels(sol)
    all_paths = static_paths + sol_paths

    mk_path = generate_metakernel(all_paths)

    spice.kclear()
    spice.furnsh(str(mk_path))
    print(f"SPICE initialized with {len(all_paths)} kernels")

    return mk_path


def cleanup_spice():
    """Unload all SPICE kernels."""
    spice.kclear()
