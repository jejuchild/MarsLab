#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import requests
from tqdm import tqdm
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup   # ★ 필요: pip install beautifulsoup4

# ============================================================
# 사용자 설정
# ============================================================
DATA_DIR = "../data/crism_ml"
OUT_DIR  = "./ddr"
os.makedirs(OUT_DIR, exist_ok=True)

ODE_PRODUCTFILES = "https://ode.rsl.wustl.edu/mars/productfiles.aspx"
ODE_REST = "https://oderest.rsl.wustl.edu/live2/"
TIMEOUT = 120

# ============================================================
# 1) TRR → DDR product_id
# ============================================================
TRR_RE = re.compile(
    r"(?P<obs>(?:frt|frs|hrl|hrs)[0-9a-f]{8})_(?P<ver>\d{2})_(?P<if>if\d{3}[a-z])_trr\d",
    re.IGNORECASE
)

def collect_ddr_product_ids(data_dir):
    out = set()
    for f in os.listdir(data_dir):
        f = f.lower()
        if "_trr" not in f:
            continue
        m = TRR_RE.search(f)
        if not m:
            continue
        obs = m.group("obs")
        ver = m.group("ver")
        ifc = m.group("if")
        dec = "de" + ifc[2:]
        out.add(f"{obs}_{ver}_{dec}_ddr1")
    return sorted(out)

# ============================================================
# 2) ODE REST: product 존재 확인
# ============================================================
def ode_product_exists(product_id):
    params = {
        "query": "product",
        "results": "cm",
        "output": "JSON",
        "ihid": "MRO",
        "iid": "CRISM",
        "pt": "DDR",
        "productid": product_id,
    }
    r = requests.get(ODE_REST, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    prod = r.json().get("ODEResults", {}).get("Products", {}).get("Product")
    return prod is not None

# ============================================================
# 3) productfiles.aspx HTML 파싱 (핵심)
# ============================================================
def get_ddr_file_links_from_html(product_id):
    params = {
        "product_id": product_id
    }
    r = requests.get(ODE_PRODUCTFILES, params=params, timeout=TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".img") or href.lower().endswith(".lbl"):
            links.append(urljoin(ODE_PRODUCTFILES, href))

    return sorted(set(links))

# ============================================================
# 4) 다운로드
# ============================================================
def download(url):
    fname = os.path.basename(urlparse(url).path)
    out = os.path.join(OUT_DIR, fname)

    if os.path.exists(out):
        return True

    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()

        ctype = r.headers.get("Content-Type", "").lower()
        if "text/html" in ctype:
            print(f"  [SKIP] HTML page: {url}")
            return False

        total = int(r.headers.get("content-length", 0))
        with open(out, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=fname
        ) as pbar:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return True

# ============================================================
# 5) 메인
# ============================================================
def main():
    pids = collect_ddr_product_ids(DATA_DIR)
    print(f"[INFO] Found {len(pids)} DDR candidate(s)")

    ok = 0
    for pid in pids:
        print(f"\n[INFO] {pid}")

        if not ode_product_exists(pid):
            print("  -> Product not found in ODE")
            continue

        links = get_ddr_file_links_from_html(pid)
        if not links:
            print("  -> No file links found on productfiles.aspx")
            continue

        print(f"  -> Found {len(links)} file link(s)")
        success = True
        for url in links:
            if not download(url):
                success = False

        if success:
            ok += 1

    print(f"\n[DONE] Downloaded DDR for {ok} / {len(pids)} products")

if __name__ == "__main__":
    main()
