#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arcadia Planitia CRISM MTRDR browse crawler (STABLE VERSION)

Workflow:
1) Query ODE REST for CRISM MTRDR (pt=MTRDR) products in a bbox
2) For each product:
   - download LBL
   - parse START_TIME -> year / DOY
   - extract observation ID (e.g. frt00003156)
3) Construct PDS browse PNG URLs (HYD / ICE / IC2 / VNIR)
4) Download existing browse images

This version includes:
- requests.Session reuse
- retry + exponential backoff
- ODE-friendly failure handling
"""

import os
import time
import requests
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# CONFIG
# ============================================================

ODE_BASE = "https://oderest.rsl.wustl.edu/live2/"

PDS_BROWSE_ROOT = (
    "https://pds-geosciences.wustl.edu/"
    "mro/mro-m-crism-5-rdr-mptargeted-v1/"
    "mrocr_4001/browse"
)

OUT_DIR = "./arcadia_browse"
LBL_DIR = os.path.join(OUT_DIR, "lbl")
PNG_DIR = os.path.join(OUT_DIR, "browse")

os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

# Arcadia Planitia
BBOX = dict(
    westernlon=-130.0,
    easternlon=150.0,
    minlat=35.0,
    maxlat=70.0,
)

PT = "MTRDR"
LIMIT = 50          # ⬅ ODE 친화적 (100도 가능하지만 50 권장)
SLEEP = 0.3

HEADERS = {
    "User-Agent": "arcadia-crism-browse-crawler/1.0"
}

BROWSE_SUFFIXES = {
    "HYD":  "brhydj",
    "ICE":  "bricej",
    "IC2":  "bric2j",
    "VNIR": "brvnaj",
}

# ============================================================
# SESSION / RETRY
# ============================================================

def make_session():
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    s = requests.Session()
    s.headers.update(HEADERS)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# ============================================================
# HELPERS
# ============================================================

def ode_query(session, limit, offset):
    params = {
        "target": "mars",
        "query": "product",
        "output": "JSON",
        "ihid": "mro",
        "iid": "crism",
        "pt": PT,
        "results": "opmf",
        "limit": limit,
        "offset": offset,
        **BBOX,
    }
    r = session.get(ODE_BASE, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def iter_products(js):
    ode = js.get("ODEResults", {})
    prods = ode.get("Products", {}).get("Product", [])
    if isinstance(prods, dict):
        prods = [prods]
    return prods


def download(session, url, out_path):
    if os.path.exists(out_path):
        return
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for c in r.iter_content(1024 * 1024):
                if c:
                    f.write(c)


def read_start_time(lbl_path):
    with open(lbl_path, "r", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("START_TIME"):
                return line.split("=")[1].strip().strip('"')
    raise RuntimeError("START_TIME not found")


def year_doy(timestr):
    t = datetime.fromisoformat(timestr.replace("Z", ""))
    return t.year, f"{t.timetuple().tm_yday:03d}"


def get_obs_id(product_id):
    return product_id.split("_")[0].lower()


def make_browse_urls(product_id, start_time):
    obsid = get_obs_id(product_id)
    year, doy = year_doy(start_time)

    base_dir = (
        f"{PDS_BROWSE_ROOT}/"
        f"{year}/{year}_{doy}/"
        f"{obsid}/"
    )

    return {
        kind: f"{base_dir}{obsid}_07_{suf}_mtr3.png"
        for kind, suf in BROWSE_SUFFIXES.items()
    }


def url_exists(session, url):
    r = session.head(url, timeout=30)
    return r.status_code == 200

# ============================================================
# MAIN
# ============================================================

def main():
    session = make_session()

    offset = 0
    checked = 0
    downloaded = 0

    while True:
        try:
            js = ode_query(session, LIMIT, offset)
        except Exception as e:
            print(f"[ODE FAIL] offset={offset} → retry after sleep: {e}")
            time.sleep(10)
            continue   # ❗ offset 증가 안 함

        products = iter_products(js)
        if not products:
            break

        for p in products:
            pid = p.get("pdsid")
            if not pid:
                continue
            checked += 1

            # -----------------------------
            # LBL URL
            # -----------------------------
            lbl_url = None
            pf = p.get("Product_files", {}).get("Product_file", [])
            if isinstance(pf, dict):
                pf = [pf]

            for f in pf:
                u = f.get("URL", "")
                if u.lower().endswith(".lbl"):
                    lbl_url = u
                    break

            if lbl_url is None:
                continue

            lbl_path = os.path.join(LBL_DIR, os.path.basename(lbl_url))
            try:
                download(session, lbl_url, lbl_path)
            except Exception as e:
                print(f"[LBL FAIL] {pid}: {e}")
                continue

            # -----------------------------
            # START_TIME
            # -----------------------------
            try:
                start_time = read_start_time(lbl_path)
            except Exception as e:
                print(f"[PARSE FAIL] {pid}: {e}")
                continue

            # -----------------------------
            # browse PNG
            # -----------------------------
            browse_urls = make_browse_urls(pid, start_time)
            any_found = False

            for kind, url in browse_urls.items():
                if url_exists(session, url):
                    out = os.path.join(PNG_DIR, f"{get_obs_id(pid)}_{kind}.png")
                    try:
                        download(session, url, out)
                        print(f"[OK] {pid} ({kind})")
                        downloaded += 1
                        any_found = True
                    except Exception as e:
                        print(f"[PNG FAIL] {pid} ({kind}): {e}")

            if not any_found:
                print(f"[NO BROWSE] {pid}")

            time.sleep(SLEEP)

        offset += LIMIT

    print("\n==============================")
    print(f"Checked products : {checked}")
    print(f"Browse downloaded: {downloaded}")
    print("==============================")

if __name__ == "__main__":
    main()
