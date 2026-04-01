"""PDS data fetcher for Mastcam-Z and HiRISE products.

Uses PDS Imaging Node directory crawling for Mastcam-Z stereo/calibrated data,
and ODE REST API (WashU mirror) for HiRISE DTM search.
"""

import re
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

from .config import (
    PDS_CACHE,
    JEZERO_LON,
    JEZERO_LAT,
    JEZERO_SEARCH_RADIUS_KM,
    MAX_RETRIES,
    DOWNLOAD_TIMEOUT,
    CHUNK_SIZE,
)

session = requests.Session()
session.headers.update({"User-Agent": "mastcam-z-coregister/1.0 (research)"})

# PDS Imaging Node base URLs for Mars 2020 Mastcam-Z
PDS_IMG_BASE = "https://pds-imaging.jpl.nasa.gov/data/mars2020"
MCAMZ_STEREO_BASE = f"{PDS_IMG_BASE}/mars2020_mastcamz_ops_stereo/data/sol"
MCAMZ_CALIB_BASE = f"{PDS_IMG_BASE}/mars2020_mastcamz_ops_calibrated/data/sol"

# ODE REST API (WashU mirror — pds.nasa.gov DNS broken as of 2026)
ODE_API = "https://oderest.rsl.wustl.edu/live2"


class _LinkParser(HTMLParser):
    """Extract href links from an HTML directory listing."""
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v and not v.startswith("?") and not v.startswith("/") and not v.startswith("http") and not v.startswith("mailto"):
                    self.links.append(v)


def _list_dir(url: str) -> list[str]:
    """List files/dirs from a PDS HTTP directory listing."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=DOWNLOAD_TIMEOUT)
            r.raise_for_status()
            parser = _LinkParser()
            parser.feed(r.text)
            # Deduplicate (PDS often lists each file twice)
            seen = set()
            unique = []
            for link in parser.links:
                if link not in seen:
                    seen.add(link)
                    unique.append(link)
            return unique
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Directory listing failed: {url} ({e})") from e
            time.sleep(2 ** attempt)


def download_file(url: str, dest: Path) -> Path:
    """Download a file with retry. Returns dest path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name}")
        return dest

    print(f"  Downloading {dest.name} ...")
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        if downloaded % (CHUNK_SIZE * 128) < CHUNK_SIZE:
                            print(f"    {pct:.0f}% ({downloaded // 1024} KB)", end="\r")
            if total > 0:
                print(f"    100% ({total // 1024} KB)      ")
            return dest
        except requests.RequestException as e:
            if dest.exists():
                dest.unlink()
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Download failed: {url} ({e})") from e
            time.sleep(2 ** attempt)


# ── Mastcam-Z Products (PDS Imaging Node) ────────────────────────────

def list_available_sols(bundle: str = "stereo") -> list[int]:
    """List available sol numbers for a Mastcam-Z bundle."""
    base = MCAMZ_STEREO_BASE if bundle == "stereo" else MCAMZ_CALIB_BASE
    entries = _list_dir(base + "/")
    sols = []
    for e in entries:
        m = re.match(r"(\d+)/", e)
        if m:
            sols.append(int(m.group(1)))
    return sorted(sols)


def search_mastcamz_products(sol: int, product_type: str = "XYZ") -> list[dict]:
    """Search Mastcam-Z products by sol via PDS Imaging Node directory crawling.

    product_type: 'XYZ' for 3D point clouds, 'RAS' for radiometric images, etc.
    For stereo products (XYZ, DSP, DFF, etc.), uses the stereo bundle.
    For calibrated (RAS, IOF), uses the calibrated bundle.
    """
    stereo_types = {"XYZ", "DSP", "DSR", "DFF", "RNE", "RNG", "UVW"}
    if product_type.upper() in stereo_types:
        base = MCAMZ_STEREO_BASE
    else:
        base = MCAMZ_CALIB_BASE

    sol_url = f"{base}/{sol:05d}/ids/rdr/zcam/"
    try:
        entries = _list_dir(sol_url)
    except RuntimeError:
        return []

    products = []
    pt_upper = product_type.upper()

    # Find matching IMG files (unique, with XML labels)
    img_files = sorted(set(e for e in entries if e.upper().endswith(".IMG") and pt_upper in e.upper()))

    for img_name in img_files:
        xml_name = img_name.rsplit(".", 1)[0] + ".xml"
        products.append({
            "product_id": img_name.rsplit(".", 1)[0],
            "data_url": sol_url + img_name,
            "label_url": sol_url + xml_name if xml_name in entries else "",
            "sol": sol,
        })

    return products


def download_mastcamz_product(product: dict, dest_dir: Path = None) -> dict:
    """Download a Mastcam-Z product (label + data). Returns dict with paths."""
    sol = product.get("sol", 0)
    if dest_dir is None:
        dest_dir = PDS_CACHE / "mastcamz" / f"sol{sol:05d}"

    result = {}
    pid = product["product_id"]

    if product.get("label_url"):
        result["label_path"] = download_file(
            product["label_url"], dest_dir / f"{pid}.xml"
        )

    if product.get("data_url"):
        result["data_path"] = download_file(
            product["data_url"], dest_dir / f"{pid}.IMG"
        )

    return result


# ── HiRISE Products (ODE REST API — WashU mirror) ───────────────────

def _ode_query(params: dict) -> dict:
    """Execute an ODE REST API query."""
    params.setdefault("output", "JSON")
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(ODE_API, params=params, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"ODE API query failed: {e}") from e
            time.sleep(2 ** attempt)


def search_hirise_dtm(
    lon: float = JEZERO_LON,
    lat: float = JEZERO_LAT,
    radius_km: float = JEZERO_SEARCH_RADIUS_KM,
) -> list[dict]:
    """Search for HiRISE DTM products covering a given area."""
    radius_deg_lat = radius_km / 59.2
    radius_deg_lon = radius_km / (59.2 * __import__("math").cos(__import__("math").radians(lat)))

    params = {
        "query": "product",
        "results": "m",
        "output": "JSON",
        "target": "Mars",
        "ihid": "MRO",
        "iid": "HIRISE",
        "pt": "DTM",
        "westernlon": lon - radius_deg_lon,
        "easternlon": lon + radius_deg_lon,
        "minlat": lat - radius_deg_lat,
        "maxlat": lat + radius_deg_lat,
    }

    data = _ode_query(params)
    products = []

    ode_response = data.get("ODEResults", {})
    status = ode_response.get("Status", "")
    if status == "ERROR":
        print(f"  ODE API error: {ode_response.get('Error', 'unknown')}")
        return products

    product_list = ode_response.get("Products", {})
    if not product_list:
        return products

    items = product_list.get("Product", [])
    if isinstance(items, dict):
        items = [items]

    for item in items:
        # HiRISE DTM ODE responses use different field names
        pid = (item.get("pdsid", "") or item.get("Product_id", "")
               or item.get("Comment", "unknown"))
        data_url = item.get("LabelURL", "")  # confusingly named, it's the IMG file
        label_url = ""
        files_url = item.get("FilesURL", "")

        # Extract product ID from the IMG URL if not set
        if not pid and data_url:
            pid = data_url.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        # Also try Product_files if available
        files = item.get("Product_files", {}).get("Product_file", [])
        if isinstance(files, dict):
            files = [files]
        for f in files:
            url = f.get("URL", "")
            fname = f.get("FileName", "").upper()
            if fname.endswith(".XML") or fname.endswith(".LBL"):
                label_url = url
            elif fname.endswith(".IMG") or fname.endswith(".TIF") or fname.endswith(".JP2"):
                data_url = url

        # Build a useful product ID from the URL
        if data_url and not pid:
            pid = data_url.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        products.append({
            "product_id": pid,
            "label_url": label_url,
            "data_url": data_url,
            "files_url": files_url,
            "lon_range": [float(item.get("Westernmost_longitude", 0)),
                          float(item.get("Easternmost_longitude", 0))],
            "lat_range": [float(item.get("Minimum_latitude", 0)),
                          float(item.get("Maximum_latitude", 0))],
            "map_scale_m": float(item.get("Map_scale", 0)),
            "metadata": item,
        })

    return products


def download_hirise_dtm(product: dict, dest_dir: Path = None) -> dict:
    """Download a HiRISE DTM product. Returns dict with paths."""
    if dest_dir is None:
        dest_dir = PDS_CACHE / "hirise"

    result = {}
    pid = product["product_id"]
    safe_pid = re.sub(r'[^\w\-.]', '_', pid)

    if product.get("label_url"):
        ext = Path(product["label_url"]).suffix or ".lbl"
        result["label_path"] = download_file(
            product["label_url"], dest_dir / f"{safe_pid}{ext}"
        )

    if product.get("data_url"):
        ext = Path(product["data_url"]).suffix or ".img"
        result["data_path"] = download_file(
            product["data_url"], dest_dir / f"{safe_pid}{ext}"
        )

    return result
