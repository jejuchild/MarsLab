"""
Shared helpers for MarsLab data downloaders.

Used by:
  - download_bulk_highres.py   (interactive bulk downloader)
  - background_fetch.py        (unattended nightly fetcher)

Contents:
  * small parsing / geometry helpers (lon conversion, WKT, point-in-polygon,
    point-to-track distance)
  * aria2c batch download + JP2 -> GeoTIFF conversion
  * "is this product already local?" checks per instrument
  * index.geojson updaters (same schema the existing scripts write)

Keep this module free of heavy imports (numpy/rasterio are imported lazily)
so unit tests stay fast.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ODE_REST = "https://oderest.rsl.wustl.edu/live2"

MARS_RADIUS_KM = 3389.5


# ═════════════════════════════════════════════════════════════════════════════
# Data layout (relative to a "data root", normally backend/)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataPaths:
    """All data / index locations below one data root (normally backend/)."""
    root: Path

    @property
    def sharad_dir(self) -> Path:
        return self.root / "sharad_highres"

    @property
    def sharad_index(self) -> Path:
        return self.root / "sharad_highres_data" / "index.geojson"

    @property
    def crism_dir(self) -> Path:
        return self.root / "mineral_cnn_data"

    @property
    def crism_index(self) -> Path:
        return self.crism_dir / "index.geojson"

    @property
    def hirise_dir(self) -> Path:
        return self.root / "hirise_data"

    @property
    def hirise_index(self) -> Path:
        return self.hirise_dir / "index.geojson"

    @property
    def ctx_dir(self) -> Path:
        return self.root / "ctx_edr_data"

    @property
    def ctx_index(self) -> Path:
        return self.ctx_dir / "index.geojson"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


# ═════════════════════════════════════════════════════════════════════════════
# Small helpers
# ═════════════════════════════════════════════════════════════════════════════

def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def lon_to_180(lon: float) -> float:
    """Convert 0-360 lon to -180..180."""
    return ((lon + 180) % 360) - 180


def lon_to_360(lon: float) -> float:
    """Convert -180..180 lon to 0..360 (ODE query convention)."""
    return lon % 360


def parse_footprint_wkt(wkt: str) -> Optional[List]:
    """Flatten every 'x y' pair of a WKT string into [[x, y], ...]."""
    numbers = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', str(wkt))
    if not numbers:
        return None
    return [[float(n[0]), float(n[1])] for n in numbers]


def parse_wkt_parts(wkt: str) -> List[List[List[float]]]:
    """Split a WKT geometry into its innermost coordinate lists.

    POLYGON((..)) -> 1 ring, MULTIPOLYGON(((..)),((..))) -> 2 rings,
    LINESTRING(..) -> 1 line. Holes are returned as separate parts (fine for
    our coverage tests: footprints have no holes).
    """
    parts = []
    for grp in re.findall(r'\(([^()]+)\)', str(wkt or "")):
        pts = parse_footprint_wkt(grp)
        if pts:
            parts.append(pts)
    return parts


def ode_products(data) -> List[Dict]:
    """Extract the Product list from an ODE JSON response (always a list)."""
    if not data:
        return []
    prods = data.get("ODEResults", {}).get("Products", {})
    if not isinstance(prods, dict):
        return []  # ODE returns "No Products Found" as a string
    prods = prods.get("Product", [])
    if isinstance(prods, dict):
        prods = [prods]
    return prods


def product_files(product: Dict) -> List[Dict]:
    pf = product.get("Product_files", {})
    if not isinstance(pf, dict):
        return []
    pf = pf.get("Product_file", [])
    if isinstance(pf, dict):
        pf = [pf]
    return pf


def extract_base_key(pdsid: str) -> str:
    """frt0001fd76_07_if166l_trr3 -> frt0001fd76_07 (same as download_crism_trr3)."""
    parts = pdsid.lower().split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else pdsid.lower()


def derive_ddr_pdsid(trr_pdsid: str) -> str:
    """frt00009312_07_if166l_trr3 -> frt00009312_07_de166l_ddr1 (same as download_crism_trr3)."""
    parts = trr_pdsid.lower().split("_")
    if len(parts) >= 4:
        activity = parts[2]
        return f"{parts[0]}_{parts[1]}_de{activity[2:]}_ddr1"
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# Geometry (site point vs. product footprint)
# ═════════════════════════════════════════════════════════════════════════════

def _unwrap(lon: float, ref: float) -> float:
    """Shift lon by ±360 so it is within 180° of ref."""
    while lon - ref > 180:
        lon -= 360
    while lon - ref < -180:
        lon += 360
    return lon


def point_in_ring(lon: float, lat: float, ring: List[List[float]]) -> bool:
    """Ray-casting point-in-polygon. Handles rings crossing the antimeridian."""
    if len(ring) < 3:
        return False
    pts = [(_unwrap(x, lon), y) for x, y in ring]
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > lat) != (yj > lat):
            x_cross = xi + (lat - yi) * (xj - xi) / (yj - yi)
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def footprint_contains(wkt: str, lon: float, lat: float) -> bool:
    return any(point_in_ring(lon, lat, r) for r in parse_wkt_parts(wkt) if len(r) >= 3)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * MARS_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def point_to_polyline_km(lon: float, lat: float, line: List[List[float]]) -> float:
    """Approximate min distance (km) from a point to a lon/lat polyline.

    Uses a local equirectangular projection centred on the point, which is
    accurate to well under 1% for the few-km distances we care about.
    """
    if not line:
        return float("inf")
    kx = math.radians(1) * MARS_RADIUS_KM * max(math.cos(math.radians(lat)), 0.01)
    ky = math.radians(1) * MARS_RADIUS_KM
    xy = [((_unwrap(x, lon) - lon) * kx, (y - lat) * ky) for x, y in line]
    if len(xy) == 1:
        return math.hypot(*xy[0])
    best = float("inf")
    for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, -(x1 * dx + y1 * dy) / seg2))
        best = min(best, math.hypot(x1 + t * dx, y1 + t * dy))
    return best


def footprint_distance_km(wkt: str, lon: float, lat: float) -> float:
    """Distance from point to the nearest footprint vertex/edge (0 if inside)."""
    parts = parse_wkt_parts(wkt)
    if not parts:
        return float("inf")
    if footprint_contains(wkt, lon, lat):
        return 0.0
    return min(point_to_polyline_km(lon, lat, p) for p in parts)


def km_to_deg(lat: float, km: float) -> Tuple[float, float]:
    """(dlat, dlon) in degrees for a distance in km at a given latitude."""
    dlat = math.degrees(km / MARS_RADIUS_KM)
    dlon = dlat / max(math.cos(math.radians(lat)), 0.05)
    return dlat, min(dlon, 180.0)


def ode_bboxes(lat: float, lon: float, half_km: float) -> List[Tuple[float, float, float, float]]:
    """ODE query bboxes (minlat, maxlat, westlon360, eastlon360) around a point.

    Returns two boxes when the box straddles the 0/360 meridian.
    """
    dlat, dlon = km_to_deg(lat, half_km)
    minlat, maxlat = max(-90.0, lat - dlat), min(90.0, lat + dlat)
    c = lon_to_360(lon)
    w, e = c - dlon, c + dlon
    r = lambda v: round(v, 5)  # noqa: E731
    if w < 0:
        return [(r(minlat), r(maxlat), r(w + 360), 360.0), (r(minlat), r(maxlat), 0.0, r(e))]
    if e > 360:
        return [(r(minlat), r(maxlat), r(w), 360.0), (r(minlat), r(maxlat), 0.0, r(e - 360))]
    return [(r(minlat), r(maxlat), r(w), r(e))]


# ═════════════════════════════════════════════════════════════════════════════
# Downloading
# ═════════════════════════════════════════════════════════════════════════════

def run_aria2(download_list: List[Tuple[str, str, str]], concurrent: int = 4,
              label: str = "") -> Tuple[int, int]:
    """
    Batch download via aria2c (interactive; used by download_bulk_highres).

    download_list: [(url, output_dir, filename), ...]
    Returns: (ok_count, fail_count)
    """
    if not download_list:
        return 0, 0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     prefix="aria2_bulk_") as f:
        input_path = f.name
        for url, out_dir, filename in download_list:
            f.write(f"{url}\n")
            f.write(f"  dir={out_dir}\n")
            f.write(f"  out={filename}\n\n")

    print(f"\n  aria2c: {len(download_list)} files, {concurrent} concurrent {label}")

    cmd = [
        "aria2c",
        "-i", input_path,
        "-j", str(concurrent),
        "-x", "16", "-s", "16", "-k", "1M",
        "--continue=true",
        "--file-allocation=trunc",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--retry-wait=3",
        "--max-tries=5",
        "--timeout=600",
        "--connect-timeout=30",
        "--summary-interval=15",
    ]

    try:
        result = subprocess.run(cmd)
        os.remove(input_path)
        if result.returncode == 0:
            return len(download_list), 0
        ok = sum(1 for _, d, fn in download_list if os.path.exists(os.path.join(d, fn)))
        return ok, len(download_list) - ok
    except FileNotFoundError:
        print("  ERROR: aria2c not found! Install: conda install -c conda-forge aria2")
        os.remove(input_path)
        return 0, len(download_list)


def aria2_product(files: List[Tuple[str, str, str]], connections: int = 4,
                  timeout_s: Optional[float] = None) -> Tuple[bool, str]:
    """Download one product's files with aria2c, quietly.

    files: [(url, out_dir, filename), ...]
    Returns (ok, message). Raises subprocess.TimeoutExpired if timeout_s hits
    (the child is killed by subprocess.run first).
    """
    if not files:
        return True, "nothing to do"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     prefix="aria2_bg_") as f:
        input_path = f.name
        for url, out_dir, filename in files:
            f.write(f"{url}\n  dir={out_dir}\n  out={filename}\n\n")
    cmd = [
        "aria2c", "-i", input_path,
        "-j", "2",
        "-x", str(connections), "-s", str(connections), "-k", "1M",
        "--continue=true",
        "--file-allocation=trunc",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--retry-wait=10",
        "--max-tries=5",
        "--timeout=300",
        "--connect-timeout=30",
        "--summary-interval=0",
        "--console-log-level=warn",
        "--download-result=hide",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        return False, "aria2c not found"
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass
    if res.returncode != 0:
        tail = " | ".join((res.stdout + res.stderr).strip().splitlines()[-3:])
        return False, f"aria2c exit {res.returncode}: {tail[:400]}"
    missing = [fn for _, d, fn in files
               if not os.path.exists(os.path.join(d, fn))
               or os.path.exists(os.path.join(d, fn + ".aria2"))]
    if missing:
        return False, f"incomplete files: {missing}"
    return True, "ok"


def cleanup_partial(files: Iterable[Tuple[str, str, str]]) -> int:
    """Remove partial downloads (file + .aria2 control file). Returns bytes removed.

    Only files that still have an .aria2 sidecar (i.e. are incomplete) are
    removed, plus orphaned sidecars. Complete files are left alone.
    """
    removed = 0
    for _, d, fn in files:
        p = Path(d) / fn
        ctl = Path(str(p) + ".aria2")
        if ctl.exists():
            if p.exists():
                removed += p.stat().st_size
                p.unlink(missing_ok=True)
            ctl.unlink(missing_ok=True)
    return removed


def convert_jp2_to_tif(jp2_path: Path, skip_gdal: bool = False) -> Optional[Path]:
    """Convert JP2 to GeoTIFF using rasterio (windowed read for large files)."""
    if skip_gdal:
        return None
    tif_path = jp2_path.with_suffix(".tif")
    if tif_path.exists():
        return tif_path
    try:
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(str(jp2_path)) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                compress="lzw",
                tiled=True,
                blockxsize=512,
                blockysize=512,
                bigtiff="IF_SAFER",
            )
            with rasterio.open(str(tif_path), "w", **profile) as dst:
                tile = 2048
                for row_off in range(0, src.height, tile):
                    for col_off in range(0, src.width, tile):
                        win = Window(
                            col_off, row_off,
                            min(tile, src.width - col_off),
                            min(tile, src.height - row_off),
                        )
                        dst.write(src.read(window=win), window=win)
        return tif_path
    except ImportError:
        print("    rasterio not installed, skipping JP2 conversion")
    except Exception as e:
        print(f"    Conversion error: {e}")
        if tif_path.exists():
            tif_path.unlink(missing_ok=True)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# "Already local?" checks
# ═════════════════════════════════════════════════════════════════════════════

def _complete(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0 and not Path(str(p) + ".aria2").exists()


def sharad_local_ids(sharad_dir: Path) -> set:
    """Upper-case SHARAD product ids with a complete .dat on disk."""
    out = set()
    if sharad_dir.exists():
        for f in sharad_dir.iterdir():
            if f.suffix == ".dat" and _complete(f):
                out.add(f.stem.upper())
    return out


def hirise_is_local(hirise_dir: Path, base_id: str) -> bool:
    """RED JP2 (complete) or RED GeoTIFF present, top-level or in <base_id>/."""
    for d in (hirise_dir, hirise_dir / base_id):
        for ext in (".JP2", ".jp2", ".tif"):
            if _complete(d / f"{base_id}_RED{ext}"):
                return True
    return False


_L_TRR3_RE = re.compile(r"_if\d+l_trr3\.img$", re.IGNORECASE)


def crism_is_local(crism_dir: Path, base_key: str) -> bool:
    """Same rule as download_crism_trr3: L-sensor TRR3 IMG + DDR1 IMG present."""
    d = crism_dir / base_key.lower()
    if not d.is_dir():
        return False
    names = os.listdir(d)
    if any(n.endswith(".aria2") for n in names):
        return False
    has_l_trr3 = any(_L_TRR3_RE.search(n) for n in names)
    has_ddr = any(n.lower().endswith("_ddr1.img") for n in names)
    return has_l_trr3 and has_ddr


def ctx_is_local(ctx_dir: Path, pid: str) -> bool:
    return _complete(ctx_dir / f"{pid.upper()}.IMG")


# ═════════════════════════════════════════════════════════════════════════════
# Index updates (schemas identical to the existing writers)
# ═════════════════════════════════════════════════════════════════════════════

def load_index(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}


def save_index(path: Path, index: Dict) -> None:
    """Atomic write (tmp file + rename) so a running server never reads half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".index_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(index, f, indent=2)
        # mkstemp creates 0600; keep the old file's mode (or a umask-based default)
        if path.exists():
            mode = path.stat().st_mode & 0o777
        else:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_sharad_index(tracks: List[Dict], sharad_dir: Path, index_path: Path) -> Tuple[int, int]:
    """Add SHARAD tracks (dicts with pid, lat, lon, footprint_wkt) to the index.

    Returns (added, total). Schema matches download_bulk_highres. Existing
    entries whose dat_file is null are filled in when the file now exists.
    """
    index = load_index(index_path)
    by_id = {feat["properties"]["product_id"].upper(): feat for feat in index["features"]}

    added = 0
    changed = False
    for t in tracks:
        pid = t["pid"]
        dat_path = sharad_dir / f"{pid.lower()}.dat"
        if not dat_path.exists():
            continue

        coords = parse_footprint_wkt(t.get("footprint_wkt", ""))
        if coords and len(coords) >= 2:
            geometry = {"type": "LineString", "coordinates": coords}
        else:
            lon = lon_to_180(t["lon"]) if t.get("lon") else 0
            geometry = {"type": "Point", "coordinates": [lon, t.get("lat") or 0]}

        existing = by_id.get(pid)
        if existing is not None:
            props = existing["properties"]
            if not props.get("dat_file"):
                props["dat_file"] = f"{pid.lower()}.dat"
                props["lbl_file"] = f"{pid.lower()}.lbl"
                if existing.get("geometry", {}).get("type") == "Point":
                    existing["geometry"] = geometry
                changed = True
            continue

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "instrument": "SHARAD_HIGHRES",
                "product_id": pid,
                "dat_file": f"{pid.lower()}.dat",
                "lbl_file": f"{pid.lower()}.lbl",
            }
        }
        index["features"].append(feature)
        by_id[pid] = feature
        added += 1

    if added or changed or not index_path.exists():
        save_index(index_path, index)
    return added, len(index["features"])


def update_hirise_index(new_items: List[Dict], hirise_dir: Path, index_path: Path) -> Tuple[int, int]:
    """Add HiRISE products (base_id, jp2_fn, lat, lon, footprint_wkt). Returns (added, total).

    Schema matches download_bulk_highres. Existing entries whose red_tif is
    null get it filled in once the GeoTIFF exists.
    """
    index = load_index(index_path)
    by_id = {feat["properties"].get("product_id", ""): feat for feat in index["features"]}

    added = 0
    changed = False
    for meta in new_items:
        base_id = meta["base_id"]
        jp2_path = hirise_dir / meta["jp2_fn"]
        if not jp2_path.exists():
            continue

        tif_name = meta["jp2_fn"].replace(".JP2", ".tif").replace(".jp2", ".tif")
        tif_path = hirise_dir / tif_name

        existing = by_id.get(base_id)
        if existing is not None:
            props = existing["properties"]
            if props.get("red_tif") is None and tif_path.exists():
                props["red_tif"] = tif_name
                changed = True
            continue

        coords = parse_footprint_wkt(meta.get("footprint_wkt", ""))
        if coords and len(coords) >= 4:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            west, east = min(lons), max(lons)
            south, north = min(lats), max(lats)
            ring = [[west, south], [east, south], [east, north],
                    [west, north], [west, south]]
            geometry = {"type": "Polygon", "coordinates": [ring]}
        else:
            lat = meta["lat"] or 0
            lon = lon_to_180(meta["lon"]) if meta["lon"] else 0
            geometry = {"type": "Point", "coordinates": [lon, lat]}

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "instrument": "HIRISE",
                "product_id": base_id,
                "quicklook": f"/hirise/quickview/{base_id}.jpg",
                "red_tif": tif_name if tif_path.exists() else None,
                "red_jp2": meta["jp2_fn"],
                "center_latitude": meta["lat"],
                "center_longitude": meta["lon"],
            }
        }
        index["features"].append(feature)
        by_id[base_id] = feature
        added += 1

    if added or changed or not index_path.exists():
        save_index(index_path, index)
    return added, len(index["features"])


def _crism_geometry_from_ddr(obs_dir: Path) -> Optional[Dict]:
    """Footprint from DDR lat/lon bands, exactly as build_trr3_index.py does."""
    ddr_lbl = ddr_img = None
    for f in obs_dir.iterdir():
        u = f.name.upper()
        if u.endswith("_DDR1.LBL"):
            ddr_lbl = f
        elif u.endswith("_DDR1.IMG"):
            ddr_img = f
    if not ddr_lbl or not ddr_img:
        return None
    try:
        import build_trr3_index as bti  # lazy: pulls numpy
        info = bti.parse_ddr_label(ddr_lbl)
        if not all(k in info for k in ("lines", "samples", "bands")):
            return None
        lat_band = bti.read_ddr_band(ddr_img, info, 3)
        lon_band = bti.read_ddr_band(ddr_img, info, 4)
        return bti.extract_footprint(lat_band, lon_band)
    except Exception:
        return None


def update_crism_index(items: List[Dict], crism_dir: Path, index_path: Path) -> Tuple[int, int]:
    """Add CRISM observations (base_key, footprint_wkt) to mineral_cnn_data/index.geojson.

    Schema matches build_trr3_index.py: product_id/obs_dir = base_key,
    trr3_product_id from the TRR3 label, instrument CRISM_TRR3. Geometry from
    the DDR (preferred) or the ODE footprint polygon.
    """
    index = load_index(index_path)
    existing_ids = {feat["properties"].get("product_id", "") for feat in index["features"]}

    added = 0
    for it in items:
        bk = it["base_key"].lower()
        if bk in existing_ids:
            continue
        obs_dir = crism_dir / bk
        if not crism_is_local(crism_dir, bk):
            continue

        trr3_product_id = None
        for f in obs_dir.iterdir():
            if f.name.upper().endswith("_TRR3.LBL") and _L_TRR3_RE.search(f.name[:-4] + ".img"):
                m = re.search(r'PRODUCT_ID\s*=\s*"?([^"\s]+)"?', f.read_text(errors="ignore"))
                if m:
                    trr3_product_id = m.group(1)
                break

        geometry = _crism_geometry_from_ddr(obs_dir)
        if geometry is None:
            parts = [p for p in parse_wkt_parts(it.get("footprint_wkt", "")) if len(p) >= 3]
            if parts:
                ring = [[lon_to_180(x) if x > 180 else x, y] for x, y in parts[0]]
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                geometry = {"type": "Polygon", "coordinates": [ring]}
            else:
                continue

        index["features"].append({
            "type": "Feature",
            "properties": {
                "product_id": bk,
                "trr3_product_id": trr3_product_id,
                "instrument": "CRISM_TRR3",
                "obs_dir": bk,
            },
            "geometry": geometry,
        })
        existing_ids.add(bk)
        added += 1

    if added or not index_path.exists():
        save_index(index_path, index)
    return added, len(index["features"])


def update_ctx_index(items: List[Dict], ctx_dir: Path, index_path: Path) -> Tuple[int, int]:
    """CTX EDR index (new; opt-in instrument). Polygon footprint + file name."""
    index = load_index(index_path)
    existing_ids = {feat["properties"].get("product_id", "") for feat in index["features"]}
    added = 0
    for it in items:
        pid = it["pid"].upper()
        if pid in existing_ids or not ctx_is_local(ctx_dir, pid):
            continue
        parts = [p for p in parse_wkt_parts(it.get("footprint_wkt", "")) if len(p) >= 3]
        if parts:
            geometry = {"type": "Polygon", "coordinates": [parts[0]]}
        else:
            geometry = {"type": "Point", "coordinates": [lon_to_180(it["lon"] or 0), it["lat"] or 0]}
        index["features"].append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "instrument": "CTX_EDR",
                "product_id": pid,
                "img_file": f"{pid}.IMG",
                "center_latitude": it.get("lat"),
                "center_longitude": it.get("lon"),
            },
        })
        existing_ids.add(pid)
        added += 1
    if added or not index_path.exists():
        save_index(index_path, index)
    return added, len(index["features"])
