#!/usr/bin/env python3
"""
Find HiRISE stereo pair candidates near SHARAD tracks for DTM generation.

Given a SHARAD product ID or a lat/lon bounding box, queries ODE for all
HiRISE RDR images in the region, identifies overlapping pairs with sufficient
emission angle difference for stereo DTM generation, and outputs ranked
candidates.

Usage:
    # By SHARAD product ID (auto-extracts search region from track geometry)
    python find_stereo_pairs.py --sharad R_5496101_001_SS19_700_A --buffer-km 50

    # By bounding box  (lon_min lat_min lon_max lat_max, -180/180)
    python find_stereo_pairs.py --bbox -170.5 39.0 -170.0 39.2

    # Batch: scan all SHARAD tracks in index
    python find_stereo_pairs.py --all-sharad --buffer-km 30

    # Generate ASP processing commands for top pairs
    python find_stereo_pairs.py --sharad R_5496101_001_SS19_700_A --asp-commands

    # Save results as JSON
    python find_stereo_pairs.py --bbox -170 38 -160 42 --output-json results.json
"""

import argparse
import asyncio
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

# ── Paths ────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.parent
SHARAD_INDEX = BACKEND_DIR / "sharad_highres_data" / "index.geojson"
DTM_INDEX = BACKEND_DIR / "hirise_dtm_data" / "index.geojson"

ODE_REST_BASE = "https://oderest.rsl.wustl.edu/live2"
MARS_RADIUS_KM = 3389.5


# ── Data classes ─────────────────────────────────────────

@dataclass
class HiRISEImage:
    product_id: str           # e.g. ESP_060706_2195_RED
    observation_id: str       # e.g. ESP_060706_2195
    center_lat: float
    center_lon: float         # -180/180
    emission_angle: float
    incidence_angle: float
    footprint_bbox: Dict[str, float]  # lat_min, lat_max, lon_min, lon_max
    observation_time: str     # UTC string


@dataclass
class StereoPair:
    image_a: HiRISEImage
    image_b: HiRISEImage
    emission_diff: float
    overlap_fraction: float
    overlap_bbox: Dict[str, float]
    convergence_quality: float
    composite_score: float
    existing_dtm: Optional[str]
    time_sep_days: float


# ── Spatial helpers (from download_sharad_for_dtm.py) ────

def _bbox_from_geometry(geom: Dict) -> Optional[Dict[str, float]]:
    coords = []
    geom_type = geom.get("type", "")
    if geom_type == "Polygon":
        rings = geom.get("coordinates", [])
        if rings:
            coords = rings[0]
    elif geom_type == "LineString":
        coords = geom.get("coordinates", [])
    elif geom_type == "Point":
        c = geom.get("coordinates", [])
        if len(c) >= 2:
            return {"lon_min": c[0], "lon_max": c[0], "lat_min": c[1], "lat_max": c[1]}
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {"lon_min": min(lons), "lon_max": max(lons), "lat_min": min(lats), "lat_max": max(lats)}


def _bboxes_overlap(a: Dict, b: Dict) -> bool:
    if a["lat_max"] < b["lat_min"] or b["lat_max"] < a["lat_min"]:
        return False
    if a["lon_max"] < b["lon_min"] or b["lon_max"] < a["lon_min"]:
        return False
    return True


def _lon_to_360(lon: float) -> float:
    return lon % 360


def _lon_to_180(lon: float) -> float:
    return lon - 360 if lon > 180 else lon


def _km_to_deg(km: float) -> float:
    """Convert km to approximate degrees on Mars surface."""
    return km / (MARS_RADIUS_KM * math.pi / 180)


def _expand_bbox(bbox: Dict, margin_deg: float) -> Dict[str, float]:
    return {
        "lat_min": max(-90, bbox["lat_min"] - margin_deg),
        "lat_max": min(90, bbox["lat_max"] + margin_deg),
        "lon_min": bbox["lon_min"] - margin_deg,
        "lon_max": bbox["lon_max"] + margin_deg,
    }


def _bbox_intersection(a: Dict, b: Dict) -> Optional[Dict[str, float]]:
    lat_min = max(a["lat_min"], b["lat_min"])
    lat_max = min(a["lat_max"], b["lat_max"])
    lon_min = max(a["lon_min"], b["lon_min"])
    lon_max = min(a["lon_max"], b["lon_max"])
    if lat_min >= lat_max or lon_min >= lon_max:
        return None
    return {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}


def _bbox_area(bbox: Dict) -> float:
    return (bbox["lat_max"] - bbox["lat_min"]) * (bbox["lon_max"] - bbox["lon_min"])


# ── WKT parser ───────────────────────────────────────────

def _parse_footprint_wkt(wkt: str) -> Optional[Dict[str, float]]:
    """Parse WKT POLYGON/MULTIPOLYGON into a bounding box."""
    # Extract all coordinate numbers from WKT
    # Pattern: find all groups of "lon lat" pairs
    numbers = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', wkt)
    if not numbers:
        return None
    lons = [float(n[0]) for n in numbers]
    lats = [float(n[1]) for n in numbers]
    return {"lon_min": min(lons), "lon_max": max(lons), "lat_min": min(lats), "lat_max": max(lats)}


# ── ODE query ────────────────────────────────────────────

async def query_hirise_in_bbox(
    bbox: Dict[str, float],
    session: aiohttp.ClientSession,
    max_results: int = 500,
) -> List[HiRISEImage]:
    """Query ODE for HiRISE RDRV11 products in a bounding box."""

    westernlon = _lon_to_360(bbox["lon_min"])
    easternlon = _lon_to_360(bbox["lon_max"])

    url = (
        f"{ODE_REST_BASE}?"
        f"target=mars&ihid=mro&iid=hirise&pt=RDRV11&"
        f"minlat={bbox['lat_min']}&maxlat={bbox['lat_max']}&"
        f"westernlon={westernlon}&easternlon={easternlon}&"
        f"output=json&results=pmf&limit={max_results * 3}"
    )

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as resp:
            if resp.status != 200:
                print(f"  ODE HTTP {resp.status}", flush=True)
                return []
            data = await resp.json()
    except Exception as e:
        print(f"  ODE error: {e}", flush=True)
        return []

    ode_results = data.get("ODEResults", {})
    if ode_results.get("Status") != "Success":
        return []

    products_raw = ode_results.get("Products", {})
    if not isinstance(products_raw, dict):
        return []  # ODE sometimes returns empty string when no products
    product_list = products_raw.get("Product", [])
    if isinstance(product_list, dict):
        product_list = [product_list]

    # Parse and filter to _RED products
    images = []
    seen_obs = set()

    for p in product_list:
        product_id = p.get("pdsid", p.get("Product_id", ""))
        if not product_id:
            continue

        # Only keep RED channel products
        if "_RED" not in product_id.upper():
            continue

        # Extract observation ID (strip _RED suffix and any trailing version)
        obs_id = re.sub(r'_RED.*$', '', product_id, flags=re.IGNORECASE)
        if obs_id in seen_obs:
            continue
        seen_obs.add(obs_id)

        # Parse emission angle
        emission = _safe_float(p.get("Emission_angle") or p.get("emission_angle"))
        if emission is None:
            continue

        incidence = _safe_float(p.get("Incidence_angle") or p.get("incidence_angle")) or 0.0

        # Parse center coordinates
        center_lat = _safe_float(p.get("Center_latitude") or p.get("center_latitude"))
        center_lon = _safe_float(p.get("Center_longitude") or p.get("center_longitude"))
        if center_lat is None or center_lon is None:
            continue
        center_lon = _lon_to_180(center_lon)

        # Parse footprint for bbox
        footprint_wkt = p.get("Footprint_C0_geometry", "")
        if isinstance(footprint_wkt, dict):
            # Sometimes ODE returns geometry as dict
            footprint_wkt = str(footprint_wkt)

        fp_bbox = _parse_footprint_wkt(str(footprint_wkt)) if footprint_wkt else None
        if fp_bbox is None:
            # Fallback: create a small bbox around center
            fp_bbox = {
                "lat_min": center_lat - 0.1,
                "lat_max": center_lat + 0.1,
                "lon_min": center_lon - 0.1,
                "lon_max": center_lon + 0.1,
            }

        # Parse observation time
        obs_time = p.get("UTC_start_time") or p.get("utc_start_time") or ""

        images.append(HiRISEImage(
            product_id=product_id,
            observation_id=obs_id,
            center_lat=center_lat,
            center_lon=center_lon,
            emission_angle=emission,
            incidence_angle=incidence,
            footprint_bbox=fp_bbox,
            observation_time=obs_time,
        ))

    return images[:max_results]


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Stereo pair evaluation ───────────────────────────────

def find_stereo_pairs(
    images: List[HiRISEImage],
    min_emission_diff: float = 5.0,
    max_emission_diff: float = 40.0,
    min_overlap: float = 0.3,
) -> List[StereoPair]:
    """Find all viable stereo pairs from a list of HiRISE images."""

    pairs = []

    for a, b in combinations(images, 2):
        # 1. Bbox overlap check (fast reject)
        if not _bboxes_overlap(a.footprint_bbox, b.footprint_bbox):
            continue

        # 2. Emission angle difference
        em_diff = abs(a.emission_angle - b.emission_angle)
        if em_diff < min_emission_diff or em_diff > max_emission_diff:
            continue

        # 3. Compute overlap
        isect = _bbox_intersection(a.footprint_bbox, b.footprint_bbox)
        if isect is None:
            continue

        isect_area = _bbox_area(isect)
        a_area = _bbox_area(a.footprint_bbox)
        b_area = _bbox_area(b.footprint_bbox)
        smaller_area = min(a_area, b_area)
        if smaller_area <= 0:
            continue

        overlap_frac = min(1.0, isect_area / smaller_area)
        if overlap_frac < min_overlap:
            continue

        # 4. Convergence quality (peaks at ~22.5 deg)
        conv_quality = max(0.0, 1.0 - abs(em_diff - 22.5) / 22.5)

        # 5. Composite score
        score = overlap_frac * conv_quality

        # 6. Time separation
        time_sep = _compute_time_sep(a.observation_time, b.observation_time)

        pairs.append(StereoPair(
            image_a=a,
            image_b=b,
            emission_diff=em_diff,
            overlap_fraction=overlap_frac,
            overlap_bbox=isect,
            convergence_quality=conv_quality,
            composite_score=score,
            existing_dtm=None,
            time_sep_days=time_sep,
        ))

    pairs.sort(key=lambda p: p.composite_score, reverse=True)
    return pairs


def _compute_time_sep(time_a: str, time_b: str) -> float:
    """Compute time separation in days between two UTC time strings."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt_a = datetime.strptime(time_a[:26], fmt)
            dt_b = datetime.strptime(time_b[:26], fmt)
            return abs((dt_a - dt_b).total_seconds()) / 86400
        except (ValueError, TypeError):
            continue
    return 0.0


# ── Existing DTM check ───────────────────────────────────

def load_existing_dtm_pairs() -> Dict[frozenset, str]:
    """Load existing DTM index and extract source observation ID pairs.

    DTM product ID format: DTEEC_060706_2195_060416_2195_A01
    Source images: ESP_060706_2195 and ESP_060416_2195
    (orbit < 10000 → PSP_ prefix, else ESP_)
    """
    if not DTM_INDEX.exists():
        return {}

    with open(DTM_INDEX) as f:
        index = json.load(f)

    pairs = {}
    pattern = re.compile(r'^DTE\w+_(\d{6})_(\d{4})_(\d{6})_(\d{4})_')

    for feat in index["features"]:
        dtm_id = feat.get("properties", {}).get("product_id", "")
        m = pattern.match(dtm_id)
        if not m:
            continue

        orbit_a, target_a = int(m.group(1)), m.group(2)
        orbit_b, target_b = int(m.group(3)), m.group(4)

        prefix_a = "PSP" if orbit_a < 11000 else "ESP"
        prefix_b = "PSP" if orbit_b < 11000 else "ESP"

        obs_a = f"{prefix_a}_{m.group(1)}_{m.group(2)}"
        obs_b = f"{prefix_b}_{m.group(3)}_{m.group(4)}"

        key = frozenset({obs_a, obs_b})
        pairs[key] = dtm_id

    return pairs


def annotate_existing_dtms(stereo_pairs: List[StereoPair], dtm_pairs: Dict[frozenset, str]):
    """Mark stereo pairs that already have DTMs."""
    for sp in stereo_pairs:
        key = frozenset({sp.image_a.observation_id, sp.image_b.observation_id})
        sp.existing_dtm = dtm_pairs.get(key)


# ── SHARAD track loading ─────────────────────────────────

def load_sharad_track(product_id: str) -> Optional[Dict]:
    """Load a SHARAD track from the index by product ID."""
    if not SHARAD_INDEX.exists():
        print(f"Error: SHARAD index not found at {SHARAD_INDEX}", flush=True)
        return None

    with open(SHARAD_INDEX) as f:
        index = json.load(f)

    pid_upper = product_id.upper()
    for feat in index["features"]:
        feat_pid = feat.get("properties", {}).get("product_id", "").upper()
        if feat_pid == pid_upper:
            return feat

    print(f"Error: SHARAD track '{product_id}' not found in index", flush=True)
    return None


def load_all_sharad_tracks() -> List[Dict]:
    """Load all SHARAD tracks from the index."""
    if not SHARAD_INDEX.exists():
        return []
    with open(SHARAD_INDEX) as f:
        index = json.load(f)
    return index.get("features", [])


# ── Output formatting ────────────────────────────────────

def format_results_table(
    pairs: List[StereoPair],
    top_n: int = 20,
    include_existing: bool = False,
) -> str:
    """Format stereo pairs as a text table."""
    display = pairs if include_existing else [p for p in pairs if p.existing_dtm is None]
    display = display[:top_n]

    if not display:
        return "  No viable stereo pairs found."

    lines = []
    header = (
        f"  {'Rank':>4} | {'Image A':<20} | {'Image B':<20} | "
        f"{'dEmis':>5} | {'Qual':>4} | {'Overlap':>7} | {'Score':>5} | "
        f"{'TimeSep':>8} | {'Status':<10}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines.append(header)
    lines.append(sep)

    for i, sp in enumerate(display, 1):
        status = sp.existing_dtm[:20] if sp.existing_dtm else "NEW"
        time_str = f"{sp.time_sep_days:.0f}d" if sp.time_sep_days > 0 else "?"
        lines.append(
            f"  {i:>4} | {sp.image_a.observation_id:<20} | {sp.image_b.observation_id:<20} | "
            f"{sp.emission_diff:>4.1f}° | {sp.convergence_quality:>4.2f} | {sp.overlap_fraction:>6.1%} | "
            f"{sp.composite_score:>5.3f} | {time_str:>8} | {status:<10}"
        )

    return "\n".join(lines)


def generate_asp_commands(pairs: List[StereoPair], max_pairs: int = 10) -> str:
    """Generate Ames Stereo Pipeline processing commands."""
    new_pairs = [p for p in pairs if p.existing_dtm is None][:max_pairs]

    if not new_pairs:
        return "# No new stereo pairs to process."

    lines = [
        "#!/bin/bash",
        "# HiRISE Stereo DTM Generation — Ames Stereo Pipeline (ASP)",
        "# Prerequisites: ISIS3 + ASP installed",
        "# Ref: https://stereopipeline.readthedocs.io/",
        "",
    ]

    for i, sp in enumerate(new_pairs, 1):
        obs_a = sp.image_a.observation_id
        obs_b = sp.image_b.observation_id
        # Derive orbit folder for HiRISE PDS URL
        prefix_a = obs_a.split("_")[0]  # ESP, PSP, or TRA
        prefix_b = obs_b.split("_")[0]
        orb_a = _hirise_orbit_folder(obs_a)
        orb_b = _hirise_orbit_folder(obs_b)
        out_dir = f"stereo_{obs_a}_{obs_b}"

        lines.extend([
            f"# === Pair {i}: {obs_a} + {obs_b} ===",
            f"# Emission diff: {sp.emission_diff:.1f}°  Overlap: {sp.overlap_fraction:.0%}  Score: {sp.composite_score:.3f}",
            f"# Overlap region: lat [{sp.overlap_bbox['lat_min']:.2f}, {sp.overlap_bbox['lat_max']:.2f}], "
            f"lon [{sp.overlap_bbox['lon_min']:.2f}, {sp.overlap_bbox['lon_max']:.2f}]",
            "",
            f"# Step 1: Download EDR products from HiRISE PDS",
            f"wget -r -np -nd -A '*.IMG' 'https://hirise-pds.lpl.arizona.edu/PDS/EDR/{prefix_a}/{orb_a}/{obs_a}/'",
            f"wget -r -np -nd -A '*.IMG' 'https://hirise-pds.lpl.arizona.edu/PDS/EDR/{prefix_b}/{orb_b}/{obs_b}/'",
            "",
            f"# Step 2: ISIS3 preprocessing (stitch CCDs, calibrate, project)",
            f"hiedr2mosaic.py {obs_a}*RED*.IMG -o {obs_a}",
            f"hiedr2mosaic.py {obs_b}*RED*.IMG -o {obs_b}",
            "",
            f"# Step 3: Map-project for better stereo matching",
            f"cam2map4stereo.py {obs_a}.mos_hijitreged.norm.cub {obs_b}.mos_hijitreged.norm.cub",
            "",
            f"# Step 4: Stereo correlation",
            f"mkdir -p {out_dir}",
            f"parallel_stereo \\",
            f"  {obs_a}.map.cub {obs_b}.map.cub \\",
            f"  {out_dir}/run \\",
            f"  --stereo-algorithm asp_mgm \\",
            f"  --subpixel-mode 3 \\",
            f"  --alignment-method none",
            "",
            f"# Step 5: Generate DTM",
            f"point2dem {out_dir}/run-PC.tif \\",
            f"  --dem-spacing 1.0 \\",
            f"  --reference-spheroid mars \\",
            f"  --dem-hole-fill-len 50 \\",
            f"  --nodata-value -32767",
            "",
            f"# Step 6: Orthoimage",
            f"mapproject {out_dir}/run-DEM.tif {obs_a}.map.cub {out_dir}/{obs_a}_ortho.tif",
            "",
            f"echo 'Pair {i} done: {out_dir}/run-DEM.tif'",
            "",
        ])

    return "\n".join(lines)


def _hirise_orbit_folder(obs_id: str) -> str:
    """Convert ESP_060706_2195 → ORB_060700_060799"""
    m = re.match(r'(ESP|PSP|TRA)_(\d{6})_', obs_id)
    if not m:
        return "ORB_000000_000099"
    prefix = m.group(1)
    orbit = int(m.group(2))
    orb_start = (orbit // 100) * 100
    orb_end = orb_start + 99
    return f"ORB_{orb_start:06d}_{orb_end:06d}"


def save_results_json(
    pairs: List[StereoPair],
    output_path: str,
    query_info: Dict,
):
    """Save results as JSON."""
    results = {
        "query": query_info,
        "pairs_count": len(pairs),
        "new_pairs_count": sum(1 for p in pairs if p.existing_dtm is None),
        "pairs": [],
    }

    for i, sp in enumerate(pairs, 1):
        results["pairs"].append({
            "rank": i,
            "image_a": sp.image_a.observation_id,
            "image_b": sp.image_b.observation_id,
            "emission_diff": round(sp.emission_diff, 2),
            "overlap_fraction": round(sp.overlap_fraction, 3),
            "convergence_quality": round(sp.convergence_quality, 3),
            "composite_score": round(sp.composite_score, 3),
            "time_separation_days": round(sp.time_sep_days, 1),
            "existing_dtm": sp.existing_dtm,
            "overlap_bbox": {k: round(v, 4) for k, v in sp.overlap_bbox.items()},
            "image_a_meta": {
                "emission_angle": sp.image_a.emission_angle,
                "incidence_angle": sp.image_a.incidence_angle,
                "center_lat": sp.image_a.center_lat,
                "center_lon": sp.image_a.center_lon,
                "observation_time": sp.image_a.observation_time,
            },
            "image_b_meta": {
                "emission_angle": sp.image_b.emission_angle,
                "incidence_angle": sp.image_b.incidence_angle,
                "center_lat": sp.image_b.center_lat,
                "center_lon": sp.image_b.center_lon,
                "observation_time": sp.image_b.observation_time,
            },
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}", flush=True)


# ── Main logic ───────────────────────────────────────────

async def run_for_bbox(
    bbox: Dict[str, float],
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    label: str = "",
) -> List[StereoPair]:
    """Run stereo pair search for a given bounding box."""

    print(f"\n  Querying ODE for HiRISE images...", flush=True)
    images = await query_hirise_in_bbox(bbox, session, args.max_results)
    print(f"  Found {len(images)} HiRISE images", flush=True)

    if len(images) < 2:
        print("  Need at least 2 images for stereo pairs", flush=True)
        return []

    n_pairs = len(images) * (len(images) - 1) // 2
    print(f"  Evaluating {n_pairs} potential pairs...", flush=True)

    pairs = find_stereo_pairs(
        images,
        min_emission_diff=args.min_emission_diff,
        max_emission_diff=args.max_emission_diff,
        min_overlap=args.min_overlap,
    )

    # Check existing DTMs
    dtm_pairs = load_existing_dtm_pairs()
    annotate_existing_dtms(pairs, dtm_pairs)

    new_count = sum(1 for p in pairs if p.existing_dtm is None)
    existing_count = len(pairs) - new_count
    print(f"  Found {len(pairs)} viable stereo pairs ({new_count} new, {existing_count} have DTMs)", flush=True)

    return pairs


async def _main(args: argparse.Namespace):
    print("=" * 70, flush=True)
    print("  HiRISE Stereo Pair Finder for DTM Generation", flush=True)
    print("=" * 70, flush=True)

    async with aiohttp.ClientSession() as session:
        all_pairs = []

        if args.sharad:
            # Single SHARAD track mode
            feat = load_sharad_track(args.sharad)
            if not feat:
                return

            geom = feat["geometry"]
            track_bbox = _bbox_from_geometry(geom)
            if not track_bbox:
                print("Error: Could not compute bbox from track geometry", flush=True)
                return

            margin = _km_to_deg(args.buffer_km)
            search_bbox = _expand_bbox(track_bbox, margin)

            pid = feat["properties"]["product_id"]
            print(f"\n  SHARAD track: {pid}", flush=True)
            print(f"  Track bbox: lat [{track_bbox['lat_min']:.2f}, {track_bbox['lat_max']:.2f}], "
                  f"lon [{track_bbox['lon_min']:.2f}, {track_bbox['lon_max']:.2f}]", flush=True)

            lat_span = search_bbox["lat_max"] - search_bbox["lat_min"]
            if lat_span > 15:
                # Split into chunks of ~10 deg latitude to avoid ODE timeout
                print(f"  Track spans {lat_span:.1f}° lat — splitting into chunks...", flush=True)
                chunk_size = 10.0
                seen_pair_keys = set()
                lat = search_bbox["lat_min"]
                chunk_idx = 0
                while lat < search_bbox["lat_max"]:
                    chunk_idx += 1
                    chunk_bbox = {
                        "lat_min": lat,
                        "lat_max": min(lat + chunk_size, search_bbox["lat_max"]),
                        "lon_min": search_bbox["lon_min"],
                        "lon_max": search_bbox["lon_max"],
                    }
                    print(f"\n  Chunk {chunk_idx}: lat [{chunk_bbox['lat_min']:.2f}, {chunk_bbox['lat_max']:.2f}]", flush=True)
                    chunk_pairs = await run_for_bbox(chunk_bbox, args, session, label=pid)
                    for p in chunk_pairs:
                        key = frozenset({p.image_a.observation_id, p.image_b.observation_id})
                        if key not in seen_pair_keys:
                            seen_pair_keys.add(key)
                            all_pairs.append(p)
                    lat += chunk_size
                    await asyncio.sleep(0.5)
                all_pairs.sort(key=lambda p: p.composite_score, reverse=True)
            else:
                print(f"  Search bbox (buffer {args.buffer_km} km): "
                      f"lat [{search_bbox['lat_min']:.2f}, {search_bbox['lat_max']:.2f}], "
                      f"lon [{search_bbox['lon_min']:.2f}, {search_bbox['lon_max']:.2f}]", flush=True)
                all_pairs = await run_for_bbox(search_bbox, args, session, label=pid)
            query_info = {
                "mode": "sharad",
                "sharad_product_id": args.sharad,
                "buffer_km": args.buffer_km,
                "search_bbox": {k: round(v, 4) for k, v in search_bbox.items()},
            }

        elif args.bbox:
            lon_min, lat_min, lon_max, lat_max = args.bbox
            search_bbox = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}

            print(f"\n  Search bbox: lat [{lat_min:.2f}, {lat_max:.2f}], "
                  f"lon [{lon_min:.2f}, {lon_max:.2f}]", flush=True)

            all_pairs = await run_for_bbox(search_bbox, args, session)
            query_info = {
                "mode": "bbox",
                "search_bbox": {k: round(v, 4) for k, v in search_bbox.items()},
            }

        elif args.all_sharad:
            tracks = load_all_sharad_tracks()
            print(f"\n  Scanning {len(tracks)} SHARAD tracks (buffer {args.buffer_km} km)...", flush=True)

            # Deduplicate pairs across tracks
            seen_pair_keys = set()
            margin = _km_to_deg(args.buffer_km)

            for i, feat in enumerate(tracks):
                pid = feat.get("properties", {}).get("product_id", "unknown")
                geom = feat.get("geometry", {})
                track_bbox = _bbox_from_geometry(geom)
                if not track_bbox:
                    continue

                # Skip tracks that span > 30 deg lat (too large, would flood ODE)
                if track_bbox["lat_max"] - track_bbox["lat_min"] > 30:
                    continue

                search_bbox = _expand_bbox(track_bbox, margin)
                print(f"\n  [{i+1}/{len(tracks)}] {pid}", flush=True)

                pairs = await run_for_bbox(search_bbox, args, session, label=pid)

                for p in pairs:
                    key = frozenset({p.image_a.observation_id, p.image_b.observation_id})
                    if key not in seen_pair_keys:
                        seen_pair_keys.add(key)
                        all_pairs.append(p)

                # Rate limiting for ODE
                await asyncio.sleep(1.0)

            all_pairs.sort(key=lambda p: p.composite_score, reverse=True)
            query_info = {
                "mode": "all_sharad",
                "buffer_km": args.buffer_km,
                "tracks_scanned": len(tracks),
            }

        # Print results
        print(f"\n{'=' * 70}", flush=True)
        print(f"  RESULTS", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(format_results_table(all_pairs, top_n=args.top, include_existing=args.include_existing), flush=True)

        # ASP commands
        if args.asp_commands:
            print(f"\n{'=' * 70}", flush=True)
            print(f"  ASP PROCESSING COMMANDS", flush=True)
            print(f"{'=' * 70}", flush=True)
            print(generate_asp_commands(all_pairs, max_pairs=args.top), flush=True)

        # JSON output
        if args.output_json:
            save_results_json(all_pairs, args.output_json, query_info)


def main():
    parser = argparse.ArgumentParser(
        description="Find HiRISE stereo pair candidates near SHARAD tracks for DTM generation"
    )

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sharad", type=str,
        help="SHARAD high-res product ID (e.g. R_5496101_001_SS19_700_A)"
    )
    input_group.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Bounding box in -180/180 longitude"
    )
    input_group.add_argument(
        "--all-sharad", action="store_true",
        help="Search near all SHARAD tracks in the index"
    )

    # Search parameters
    parser.add_argument("--buffer-km", type=float, default=30.0,
                        help="Buffer around SHARAD track in km (default: 30)")
    parser.add_argument("--min-emission-diff", type=float, default=5.0,
                        help="Minimum emission angle difference in degrees (default: 5)")
    parser.add_argument("--max-emission-diff", type=float, default=40.0,
                        help="Maximum emission angle difference in degrees (default: 40)")
    parser.add_argument("--min-overlap", type=float, default=0.3,
                        help="Minimum overlap fraction 0-1 (default: 0.3)")
    parser.add_argument("--max-results", type=int, default=500,
                        help="Max HiRISE images to fetch from ODE per query (default: 500)")

    # Output options
    parser.add_argument("--asp-commands", action="store_true",
                        help="Generate ASP processing commands for top pairs")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N pairs (default: 20)")
    parser.add_argument("--include-existing", action="store_true",
                        help="Include pairs that already have DTMs")

    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
