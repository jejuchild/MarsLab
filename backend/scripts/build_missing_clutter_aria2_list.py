#!/usr/bin/env python3
"""
Build an aria2c URL list for missing SHARAD cluttergram pairs.

Outputs a plain text file containing one URL per line for missing
`s_XXXXXXXX_sim.img` and `s_XXXXXXXX_sim.xml` files corresponding to
local SHARAD RDR products (`r_*.dat`).
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

ODE_REST_BASE = "https://oderest.rsl.wustl.edu/live2/"


def query_ode_sim(sim_id: str) -> dict | None:
    query_url = (
        f"{ODE_REST_BASE}?target=mars&ihid=mro&iid=sharad"
        f"&productid=s_{sim_id}_sim*&output=json&results=pfm&limit=5"
    )
    try:
        with urllib.request.urlopen(query_url, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None

    products_container = data.get("ODEResults", {}).get("Products", {})
    if not isinstance(products_container, dict):
        return None
    products = products_container.get("Product", [])
    if isinstance(products, dict):
        products = [products]
    if not products:
        return None

    files = products[0].get("Product_files", {}).get("Product_file", [])
    if isinstance(files, dict):
        files = [files]

    result: dict[str, str] = {}
    for file_obj in files:
        fname = str(file_obj.get("FileName", "")).upper()
        url = str(file_obj.get("URL", ""))
        if fname.endswith("_SIM.IMG"):
            result["img"] = url
        elif fname.endswith("_SIM.XML"):
            result["xml"] = url
    return result if "img" in result else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sharad-dir",
        default="backend/sharad_highres",
        help="Directory containing SHARAD RDR and clutter files",
    )
    parser.add_argument(
        "--out",
        default="/tmp/sharad_clutter_missing_urls.txt",
        help="Output URL list path for aria2c -i",
    )
    args = parser.parse_args()

    sharad_dir = Path(args.sharad_dir)
    out_path = Path(args.out)

    obs_ids: set[str] = set()
    for dat_path in sorted(sharad_dir.glob("r_*.dat")):
        match = re.match(r"r_(\d+)_", dat_path.name, re.IGNORECASE)
        if match:
            obs_ids.add(match.group(1).zfill(8))

    urls: list[str] = []
    missing_ids: list[str] = []
    not_found: list[str] = []

    for sim_id in sorted(obs_ids):
        img_path = sharad_dir / f"s_{sim_id}_sim.img"
        xml_path = sharad_dir / f"s_{sim_id}_sim.xml"
        need_img = not img_path.exists()
        need_xml = not xml_path.exists()
        if not need_img and not need_xml:
            continue

        missing_ids.append(sim_id)
        info = query_ode_sim(sim_id)
        if not info:
            not_found.append(sim_id)
            continue

        if need_img and "img" in info:
            urls.append(info["img"])
        if need_xml and "xml" in info:
            urls.append(info["xml"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")

    print(f"total_obs={len(obs_ids)}")
    print(f"missing_pairs_or_partial={len(missing_ids)}")
    print(f"not_found_in_ode={len(not_found)}")
    print(f"download_url_count={len(urls)}")
    print(f"out={out_path}")
    if not_found:
        print("not_found_ids=" + ",".join(not_found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
