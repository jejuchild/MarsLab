#!/usr/bin/env python3
"""
Download SHARAD surface clutter simulation files from PDS for all
locally-stored SHARAD High-Res RDR products.

For each RDR product (e.g. R_0277201_001_SS19_700_A), this script:
  1. Extracts the 7-digit observation ID (0277201)
  2. Zero-pads to 8 digits (00277201)
  3. Queries ODE for the corresponding sim product (s_00277201_sim)
  4. Downloads s_XXXXXXXX_sim.img and s_XXXXXXXX_sim.xml

Usage:
    python download_cluttergrams.py [--dry-run] [--product PRODUCT_ID]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARAD_HR_DIR = os.path.join(BACKEND_DIR, "sharad_highres")
INDEX_PATH = os.path.join(BACKEND_DIR, "sharad_highres_data", "index.geojson")

ODE_REST_BASE = "https://oderest.rsl.wustl.edu/live2/"


def extract_obs_id(product_id: str) -> str:
    """Extract observation ID from RDR product ID.

    R_0277201_001_SS19_700_A -> 0277201
    """
    parts = product_id.upper().split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""


def obs_id_to_sim_id(obs_id: str) -> str:
    """Zero-pad observation ID to 8 digits for sim product lookup.

    0277201 -> 00277201
    """
    return obs_id.zfill(8)


def sim_id_to_dir(sim_id: str) -> str:
    """Compute PDS directory name from sim ID.

    00277201 -> s_0027xx
    """
    return f"s_{sim_id[:4]}xx"


def query_ode_sim(sim_id: str) -> dict | None:
    """Query ODE for simulation product files.

    Returns dict with 'sim_img_url', 'sim_xml_url', 'sim_img_kb', 'lines', 'samples'
    or None if not found.
    """
    query_url = (
        f"{ODE_REST_BASE}?target=mars&ihid=mro&iid=sharad"
        f"&productid=s_{sim_id}_sim*&output=json&results=pfm&limit=5"
    )

    try:
        with urllib.request.urlopen(query_url, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  [ERROR] ODE query failed: {e}")
        return None

    products_container = data.get("ODEResults", {}).get("Products", {})
    if not isinstance(products_container, dict):
        return None
    products = products_container.get("Product", [])
    if isinstance(products, dict):
        products = [products]
    if not products:
        return None

    p = products[0]
    files = p.get("Product_files", {}).get("Product_file", [])
    if isinstance(files, dict):
        files = [files]

    result = {}
    for f in files:
        fname = f.get("FileName", "").upper()
        url = f.get("URL", "")
        kb = int(f.get("KBytes", 0))

        if fname.endswith("_SIM.IMG"):
            result["sim_img_url"] = url
            result["sim_img_kb"] = kb
        elif fname.endswith("_SIM.XML"):
            result["sim_xml_url"] = url

    if "sim_img_url" not in result:
        return None

    return result


def download_file(url: str, dest_path: str, expected_kb: int = 0) -> bool:
    """Download a file using aria2c for speed (parallel connections)."""
    if os.path.exists(dest_path):
        existing_kb = os.path.getsize(dest_path) // 1024
        if expected_kb > 0 and abs(existing_kb - expected_kb) < 10:
            print(f"  [SKIP] Already exists: {os.path.basename(dest_path)} ({existing_kb} KB)")
            return True

    dest_dir = os.path.dirname(dest_path)
    dest_name = os.path.basename(dest_path)

    try:
        import subprocess
        print(f"  aria2c: {dest_name} ({expected_kb} KB)...")
        result = subprocess.run(
            [
                "aria2c", "-x", "4", "-s", "4",  # 4 connections
                "--dir", dest_dir,
                "--out", dest_name,
                "--continue=true",
                "--max-tries=3",
                "--retry-wait=2",
                "--console-log-level=error",
                "--summary-interval=0",
                url,
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0 and os.path.exists(dest_path):
            actual_kb = os.path.getsize(dest_path) // 1024
            print(f"  Done: {actual_kb} KB")
            return True
        else:
            print(f"  [ERROR] aria2c failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def get_all_rdr_products() -> list[str]:
    """Get all SHARAD HIRES RDR product IDs from .lbl files in the data directory."""
    products = []
    for fname in sorted(os.listdir(SHARAD_HR_DIR)):
        if fname.endswith(".lbl"):
            with open(os.path.join(SHARAD_HR_DIR, fname), "r") as f:
                for line in f:
                    m = re.match(r'\s*PRODUCT_ID\s*=\s*"?([^"\s]+)"?', line)
                    if m:
                        products.append(m.group(1).upper())
                        break
    return products


def main():
    parser = argparse.ArgumentParser(description="Download SHARAD cluttergram sim files")
    parser.add_argument("--dry-run", action="store_true", help="Only check availability, don't download")
    parser.add_argument("--product", type=str, help="Download for specific product ID only")
    args = parser.parse_args()

    if not os.path.isdir(SHARAD_HR_DIR):
        print(f"SHARAD HIRES data directory not found: {SHARAD_HR_DIR}")
        sys.exit(1)

    # Get products
    if args.product:
        products = [args.product.upper()]
    else:
        products = get_all_rdr_products()

    if not products:
        print("No SHARAD HIRES products found.")
        sys.exit(0)

    print(f"Found {len(products)} SHARAD HIRES RDR products\n")

    stats = {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0, "not_found": 0}

    for i, pid in enumerate(products, 1):
        obs_id = extract_obs_id(pid)
        sim_id = obs_id_to_sim_id(obs_id)

        print(f"[{i}/{len(products)}] {pid}")
        print(f"  Observation: {obs_id} -> Sim ID: s_{sim_id}")

        # Query ODE
        info = query_ode_sim(sim_id)
        if not info:
            print(f"  [NOT FOUND] No clutter simulation available for this observation")
            stats["not_found"] += 1
            continue

        stats["found"] += 1
        img_kb = info.get("sim_img_kb", 0)
        print(f"  Found: sim.img ({img_kb} KB)")

        if args.dry_run:
            print(f"  [DRY RUN] Would download to {SHARAD_HR_DIR}/")
            stats["skipped"] += 1
            continue

        # Download sim.img
        img_dest = os.path.join(SHARAD_HR_DIR, f"s_{sim_id}_sim.img")
        img_ok = download_file(info["sim_img_url"], img_dest, img_kb)

        # Download sim.xml
        xml_ok = True
        if "sim_xml_url" in info:
            xml_dest = os.path.join(SHARAD_HR_DIR, f"s_{sim_id}_sim.xml")
            xml_ok = download_file(info["sim_xml_url"], xml_dest)

        if img_ok and xml_ok:
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1

        # Rate limiting
        time.sleep(0.5)

    # Summary
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total products:  {len(products)}")
    print(f"  Sims found:      {stats['found']}")
    print(f"  Downloaded:      {stats['downloaded']}")
    print(f"  Skipped:         {stats['skipped']}")
    print(f"  Failed:          {stats['failed']}")
    print(f"  Not available:   {stats['not_found']}")


if __name__ == "__main__":
    main()
