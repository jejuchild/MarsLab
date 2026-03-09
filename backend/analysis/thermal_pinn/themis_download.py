"""
THEMIS IRBTR data download pipeline.

Queries ODE REST API for THEMIS IR Brightness Temperature Record (IRBTR)
products, downloads PDS3 IMG files, and parses Band 9 (12.57 µm) brightness
temperature arrays.

Product: IRBTR (IR Brightness Temperature Record)
PDS ID:  ODY-M-THM-3-IRBTR-V1.0
Band:    IR-9 (12.57 µm, atmospheric window → BT ≈ T_surface)
Format:  PDS3 IMG with attached label, 8-bit unsigned DN
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pvl
import requests

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = _BACKEND_DIR / "data" / "themis_irbtr"

ODE_BASE = "https://oderest.rsl.wustl.edu/live2"

# Arcadia Planitia bounding box (default)
DEFAULT_BBOX = {
    "minlat": 35.0,
    "maxlat": 55.0,
    "westernlon": 160.0,   # 0-360 East-positive
    "easternlon": 230.0,
}


@dataclass
class IRBTRProduct:
    """Metadata for a single THEMIS IRBTR product."""
    product_id: str
    img_url: str
    center_lat: float
    center_lon: float
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
    solar_lon: float        # Ls (season, 0-360°)
    local_time: float       # local solar time (hours)
    utc_start: str
    orbit: int


@dataclass
class IRBTRImage:
    """Parsed IRBTR image with brightness temperature."""
    product_id: str
    bt: np.ndarray              # (lines, samples) in Kelvin
    center_lat: float
    center_lon: float
    solar_lon: float            # Ls
    local_time: float           # hours
    band_number: int
    band_center_um: float
    bt_min: float
    bt_max: float
    lines: int
    samples: int
    orbit: int
    utc_start: str
    scaling_factor: float
    offset: float


# ── ODE REST API ──────────────────────────────────────────────


def query_ode_products(
    minlat: float = DEFAULT_BBOX["minlat"],
    maxlat: float = DEFAULT_BBOX["maxlat"],
    westernlon: float = DEFAULT_BBOX["westernlon"],
    easternlon: float = DEFAULT_BBOX["easternlon"],
    max_products: int = 5000,
    page_size: int = 500,
) -> list[IRBTRProduct]:
    """
    Query ODE REST API for THEMIS IRBTR products in a bounding box.
    Paginates automatically.
    """
    all_products: list[IRBTRProduct] = []
    offset = 0

    while True:
        params = {
            "target": "mars",
            "query": "product",
            "results": "cm",
            "output": "JSON",
            "ihid": "ODY",
            "iid": "THEMIS",
            "pt": "IRBTR",
            "minlat": minlat,
            "maxlat": maxlat,
            "westernlon": westernlon,
            "easternlon": easternlon,
            "loc": "b",
            "limit": page_size,
            "offset": offset,
        }

        logger.info("Querying ODE: offset=%d ...", offset)
        resp = requests.get(ODE_BASE, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        results = data["ODEResults"]
        total = int(results["Count"])
        logger.info("ODE total: %d products (fetched offset=%d)", total, offset)

        if total == 0:
            break

        raw_products = results["Products"]["Product"]
        if isinstance(raw_products, dict):
            raw_products = [raw_products]

        for p in raw_products:
            try:
                prod = IRBTRProduct(
                    product_id=_extract_product_id(p),
                    img_url=p.get("LabelURL", ""),
                    center_lat=float(p.get("Center_latitude", 0)),
                    center_lon=float(p.get("Center_longitude", 0)),
                    min_lat=float(p.get("Minimum_latitude", 0)),
                    max_lat=float(p.get("Maximum_latitude", 0)),
                    west_lon=float(p.get("Westernmost_longitude", 0)),
                    east_lon=float(p.get("Easternmost_longitude", 0)),
                    solar_lon=float(p.get("Solar_longitude", 0)),
                    local_time=float(p.get("Solar_time", 0)),
                    utc_start=p.get("UTC_start_time", ""),
                    orbit=int(p.get("Start_orbit_number", 0)),
                )
                all_products.append(prod)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping malformed product: %s", exc)

        offset += page_size
        if offset >= total or len(all_products) >= max_products:
            break
        time.sleep(0.5)  # polite rate limit

    logger.info("Retrieved %d IRBTR products", len(all_products))
    return all_products[:max_products]


def _extract_product_id(raw: dict) -> str:
    """Extract product ID from ODE response."""
    # Try ProductURL first
    purl = raw.get("ProductURL", "")
    if "product_id=" in purl:
        return purl.split("product_id=")[1].split("&")[0]
    # Fall back to LabelURL filename
    label_url = raw.get("LabelURL", "")
    if label_url:
        return Path(label_url).stem
    return raw.get("pdsid", "UNKNOWN")


# ── Nighttime filter ─────────────────────────────────────────


def filter_nighttime(products: list[IRBTRProduct]) -> list[IRBTRProduct]:
    """
    Keep only nighttime observations (local_time < 6 or > 18).
    Nighttime ATI is preferred for thermal inertia work — suppresses
    albedo effects and isolates thermophysical properties.
    """
    night = [p for p in products if p.local_time < 6.0 or p.local_time > 18.0]
    logger.info("Nighttime filter: %d / %d kept", len(night), len(products))
    return night


# ── Download ─────────────────────────────────────────────────


def download_img(product: IRBTRProduct, out_dir: Optional[Path] = None) -> Path:
    """Download a single IRBTR IMG file. Returns local path."""
    if out_dir is None:
        out_dir = DATA_DIR / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{product.product_id}.IMG"
    out_path = out_dir / filename

    if out_path.exists():
        logger.debug("Already downloaded: %s", out_path.name)
        return out_path

    logger.info("Downloading %s (%.2f°N, %.2f°E, Ls=%.1f°) ...",
                product.product_id, product.center_lat,
                product.center_lon, product.solar_lon)

    resp = requests.get(product.img_url, timeout=300, stream=True)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)

    logger.info("Saved: %s (%d bytes)", out_path.name, out_path.stat().st_size)
    return out_path


# ── PDS3 IMG parsing ─────────────────────────────────────────


def parse_irbtr_img(img_path: Path) -> IRBTRImage:
    """
    Parse a THEMIS IRBTR PDS3 IMG file.
    Returns IRBTRImage with brightness temperature array in Kelvin.
    """
    raw = img_path.read_bytes()

    # Parse PDS3 label (text portion before binary)
    label_text = raw.split(b"END\r\n")[0] + b"END\r\n"
    label = pvl.loads(label_text.decode("ascii", errors="replace"))

    record_bytes = int(label["RECORD_BYTES"])
    image_ptr = int(label["^IMAGE"])  # 1-indexed record number

    img_obj = label["IMAGE"]
    lines = int(img_obj["LINES"])
    samples = int(img_obj["LINE_SAMPLES"])
    scaling = float(img_obj["SCALING_FACTOR"])
    offset_val = float(img_obj["OFFSET"])

    # Read binary image data
    img_start = (image_ptr - 1) * record_bytes
    n_bytes = lines * samples  # 8-bit = 1 byte/pixel

    if img_start + n_bytes > len(raw):
        raise ValueError(
            f"Image data exceeds file size: need {img_start + n_bytes}, "
            f"have {len(raw)} bytes"
        )

    dn = np.frombuffer(raw[img_start: img_start + n_bytes], dtype=np.uint8)
    dn = dn.reshape((lines, samples))

    # Convert DN → Brightness Temperature (Kelvin)
    bt = dn.astype(np.float32) * scaling + offset_val

    # Extract metadata
    band_num = int(label.get("BAND_NUMBER", 9))
    band_center = label.get("BAND_CENTER", None)
    if band_center is not None:
        # pvl may return a Quantity object or a string with units
        try:
            band_center = float(band_center)
        except (TypeError, ValueError):
            # Extract numeric value from Quantity or string repr
            import re
            match = re.search(r'[\d.]+', str(band_center))
            band_center = float(match.group()) if match else 12.57
    else:
        band_center = 12.57  # default Band 9

    return IRBTRImage(
        product_id=img_path.stem,
        bt=bt,
        center_lat=float(label.get("CENTER_LATITUDE", 0)),
        center_lon=float(label.get("CENTER_LONGITUDE", 0)),
        solar_lon=float(label.get("SOLAR_LONGITUDE", 0)),
        local_time=float(label.get("LOCAL_TIME", 0)),
        band_number=band_num,
        band_center_um=band_center,
        bt_min=float(label.get("MINIMUM_BRIGHTNESS_TEMPERATURE", bt.min())),
        bt_max=float(label.get("MAXIMUM_BRIGHTNESS_TEMPERATURE", bt.max())),
        lines=lines,
        samples=samples,
        orbit=int(label.get("ORBIT_NUMBER", 0)),
        utc_start=str(label.get("START_TIME", "")),
        scaling_factor=scaling,
        offset=offset_val,
    )


# ── Dataset generation ───────────────────────────────────────


@dataclass
class THEMISObservation:
    """Single observation for PINN training."""
    lat: float          # degrees N
    lon: float          # degrees E (0-360)
    solar_lon: float    # Ls (0-360°)
    local_time: float   # hours
    bt_kelvin: float    # brightness temperature
    orbit: int
    utc_start: str


def build_training_dataset(
    products: list[IRBTRProduct],
    raw_dir: Optional[Path] = None,
    spatial_subsample: int = 4,
) -> list[THEMISObservation]:
    """
    Download, parse, and build training observations from IRBTR products.
    
    spatial_subsample: take every Nth pixel to reduce data volume
                      (320×272 → 80×68 = 5440 points per image)
    """
    observations: list[THEMISObservation] = []

    for i, prod in enumerate(products):
        try:
            img_path = download_img(prod, out_dir=raw_dir)
            parsed = parse_irbtr_img(img_path)

            # Validate Band 9
            if parsed.band_number != 9:
                logger.warning("Skipping %s: band %d (not 9)",
                               prod.product_id, parsed.band_number)
                continue

            # Spatial subsampling
            bt_sub = parsed.bt[::spatial_subsample, ::spatial_subsample]

            # Generate approximate lat/lon grid for the image
            # IRBTR covers a strip: lat range = [min_lat, max_lat]
            # lon range = [west_lon, east_lon]
            lat_grid = np.linspace(prod.max_lat, prod.min_lat,
                                   bt_sub.shape[0])
            lon_grid = np.linspace(prod.west_lon, prod.east_lon,
                                   bt_sub.shape[1])

            for r in range(bt_sub.shape[0]):
                for c in range(bt_sub.shape[1]):
                    bt_val = float(bt_sub[r, c])
                    if bt_val < 100.0 or bt_val > 350.0:
                        continue  # skip obviously invalid
                    observations.append(THEMISObservation(
                        lat=float(lat_grid[r]),
                        lon=float(lon_grid[c]),
                        solar_lon=prod.solar_lon,
                        local_time=prod.local_time,
                        bt_kelvin=bt_val,
                        orbit=prod.orbit,
                        utc_start=prod.utc_start,
                    ))

            if (i + 1) % 50 == 0:
                logger.info("Processed %d / %d products (%d obs so far)",
                            i + 1, len(products), len(observations))

        except Exception as exc:
            logger.warning("Failed to process %s: %s", prod.product_id, exc)
            continue

    logger.info("Built dataset: %d observations from %d products",
                len(observations), len(products))
    return observations


def save_dataset(observations: list[THEMISObservation], out_path: Path) -> None:
    """Save observations as .npz for efficient loading."""
    if not observations:
        logger.warning("No observations to save")
        return

    arrays = {
        "lat": np.array([o.lat for o in observations], dtype=np.float32),
        "lon": np.array([o.lon for o in observations], dtype=np.float32),
        "solar_lon": np.array([o.solar_lon for o in observations], dtype=np.float32),
        "local_time": np.array([o.local_time for o in observations], dtype=np.float32),
        "bt_kelvin": np.array([o.bt_kelvin for o in observations], dtype=np.float32),
        "orbit": np.array([o.orbit for o in observations], dtype=np.int32),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    logger.info("Saved dataset: %s (%d obs, %.1f MB)",
                out_path, len(observations),
                out_path.stat().st_size / 1e6)


def load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    """Load saved dataset."""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


# ── CLI entry point ──────────────────────────────────────────


def main():
    """Download Arcadia Planitia THEMIS IRBTR data and build training dataset."""
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Download THEMIS IRBTR data for thermal PINN training")
    parser.add_argument("--minlat", type=float, default=35.0)
    parser.add_argument("--maxlat", type=float, default=55.0)
    parser.add_argument("--westernlon", type=float, default=160.0)
    parser.add_argument("--easternlon", type=float, default=230.0)
    parser.add_argument("--max-products", type=int, default=500)
    parser.add_argument("--nighttime-only", action="store_true", default=True)
    parser.add_argument("--subsample", type=int, default=4)
    parser.add_argument("--out", type=str,
                        default=str(DATA_DIR / "arcadia_bt_dataset.npz"))
    args = parser.parse_args()

    # 1. Query ODE
    products = query_ode_products(
        minlat=args.minlat,
        maxlat=args.maxlat,
        westernlon=args.westernlon,
        easternlon=args.easternlon,
        max_products=args.max_products,
    )

    # 2. Filter nighttime
    if args.nighttime_only:
        products = filter_nighttime(products)

    if not products:
        logger.error("No products found!")
        return

    # Save product catalog
    catalog_path = DATA_DIR / "product_catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w") as f:
        json.dump([asdict(p) for p in products], f, indent=2)
    logger.info("Saved catalog: %s (%d products)", catalog_path, len(products))

    # 3. Download + parse + build dataset
    observations = build_training_dataset(
        products,
        spatial_subsample=args.subsample,
    )

    # 4. Save
    save_dataset(observations, Path(args.out))

    # 5. Summary statistics
    if observations:
        bts = [o.bt_kelvin for o in observations]
        ls_vals = [o.solar_lon for o in observations]
        logger.info("Dataset summary:")
        logger.info("  BT range: %.1f - %.1f K", min(bts), max(bts))
        logger.info("  Ls range: %.1f° - %.1f°", min(ls_vals), max(ls_vals))
        logger.info("  Unique orbits: %d", len(set(o.orbit for o in observations)))


if __name__ == "__main__":
    main()
