#!/usr/bin/env python3
"""
MarsLab unattended background fetcher (nightly cron job).

Goal: every priority site (config/fetch_sites.json) should gradually get
local SHARAD, CRISM and HiRISE (optionally CTX) products covering the SAME
point, so the instruments can be compared side by side.

Each run:
  1. Discovery — for every site x instrument, ask PDS ODE which products cover
     the site point:
       SHARAD  : RDR tracks whose ground track passes within 5 km
       CRISM   : targeted L-sensor TRR3 (FRT/FRS/HRL/HRS/ATO/ATU) footprints
                 containing the point
       HiRISE  : RDR RED footprints containing the point
       CTX     : EDR footprints containing the point (opt-in)
     If no footprint contains the point, fall back to products within the
     site's radius_km.  Products already on disk are skipped (same
     completeness rules as the existing downloaders).
  2. Planning — round-robin over site x instrument pairs; within each round
     the pairs with the fewest local products go first (ties: site priority).
     At most --per-pair-cap new products per pair per run.
  3. Download — one product at a time via aria2c, with guards:
       * refuse to start / stop when free space < --min-free-gb
       * stop when bytes downloaded this run > --max-run-gb
       * stop at the wall-clock limit (--max-hours)
       * single instance (fcntl.flock on logs/background_fetch.lock)
       * polite ODE pacing + retries with exponential backoff
       * partial downloads (+ .aria2 control files) removed on failure
     Each finished product is added to its index.geojson immediately (same
     schema as download_bulk_highres.py / build_trr3_index.py).
  4. Best-effort POST /api/repair-index on the local server.

State : <data-root>/logs/background_fetch_state.json
Log   : <data-root>/logs/background_fetch.log   (key=value lines)

Usage:
    python background_fetch.py --dry-run
    python background_fetch.py --site jezero --instrument sharad --max-run-gb 5
    python background_fetch.py --data-root /disk1/cspark/MarsLab/backend --dry-run

Exit codes: 0 ok / stopped by a guard, 1 error, 2 refused (disk), 3 already running.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fetch_lib as fl  # noqa: E402

DEFAULT_SITES_FILE = BACKEND_DIR / "config" / "fetch_sites.json"
INSTRUMENTS = ("sharad", "crism", "hirise", "ctx")
DEFAULT_INSTRUMENTS = ("sharad", "crism", "hirise")

DEFAULT_MIN_FREE_GB = 1000.0
DEFAULT_MAX_RUN_GB = 150.0
DEFAULT_MAX_HOURS = 5.0
DEFAULT_PER_PAIR_CAP = 5
MAX_FAILURES = 3            # give up on a product after this many failed runs
SHARAD_MAX_KM = 5.0         # track must pass within this distance of the site
POINT_HALF_KM = 0.5         # half-size of the "point" query box
ODE_LIMIT = 1000
REPAIR_URL = "http://localhost:8000/api/repair-index"

GB = 1024 ** 3
# Fallback size estimates (ODE KBytes is missing / wrong for some products,
# e.g. CRISM TRR3 reports ~1.6 MB for a ~600 MB file).
NOMINAL_BYTES = {
    "sharad": 65 * 1024 ** 2,
    "crism_full": 630 * 1024 ** 2,   # FRT/FRS/ATO/ATU: TRR3 ~605 MB + DDR ~19 MB
    "crism_half": 300 * 1024 ** 2,   # HRL/HRS
    "hirise": 1024 ** 3,
    "ctx": 60 * 1024 ** 2,
}
CRISM_TARGETED = ("frt", "frs", "hrl", "hrs", "ato", "atu")

log = logging.getLogger("background_fetch")


def kv(event: str, level: int = logging.INFO, **fields) -> None:
    """Structured log line: event=<name> k=v ..."""
    parts = [f"event={event}"]
    for k, v in fields.items():
        if isinstance(v, float):
            v = f"{v:.3f}"
        s = str(v)
        if not s or any(c in s for c in ' ="'):
            s = json.dumps(s)
        parts.append(f"{k}={s}")
    log.log(level, " ".join(parts))


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Interrupted(BaseException):
    """Raised from the SIGTERM/SIGINT handler (BaseException: not swallowed)."""


# ═════════════════════════════════════════════════════════════════════════════
# Data model
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Site:
    id: str
    name: str
    lat: float
    lon: float
    radius_km: float = 10.0
    priority: int = 2


@dataclass
class Candidate:
    instrument: str
    product_id: str
    site_id: str
    est_bytes: int
    distance_km: float
    dest_dir: Path
    files: List[Tuple[str, str]]          # (url, filename) inside dest_dir
    meta: Dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.instrument}:{self.product_id}"


@dataclass
class PairStatus:
    site: Site
    instrument: str
    covering: int = 0
    local: int = 0
    missing: List[Candidate] = field(default_factory=list)
    fallback: bool = False
    error: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.site.id}/{self.instrument}"


def load_sites(path: Path) -> Tuple[List[Site], List[str]]:
    with open(path) as f:
        cfg = json.load(f)
    sites = []
    for s in cfg["sites"]:
        sites.append(Site(
            id=s.get("id") or s["name"].lower().replace(" ", "_"),
            name=s["name"],
            lat=float(s["lat"]),
            lon=float(s["lon"]),
            radius_km=float(s.get("radius_km", 10)),
            priority=int(s.get("priority", 2)),
        ))
    instruments = [i for i in cfg.get("instruments", DEFAULT_INSTRUMENTS) if i in INSTRUMENTS]
    return sites, instruments


def select_sites(sites: List[Site], names: Optional[List[str]]) -> List[Site]:
    if not names:
        return sites
    wanted = {n.lower() for n in names}
    out = [s for s in sites if s.id.lower() in wanted or s.name.lower() in wanted]
    unknown = wanted - {s.id.lower() for s in out} - {s.name.lower() for s in out}
    if unknown:
        raise SystemExit(f"Unknown site(s): {', '.join(sorted(unknown))}. "
                         f"Known: {', '.join(s.id for s in sites)}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# ODE client (polite)
# ═════════════════════════════════════════════════════════════════════════════

class OdeClient:
    """Sequential ODE REST client: min interval between calls, retry + backoff."""

    def __init__(self, base: str = fl.ODE_REST, min_interval: float = 1.0,
                 retries: int = 4, timeout: float = 120.0,
                 sleep: Callable[[float], None] = time.sleep, session=None):
        self.base = base
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self.sleep = sleep
        self._last = 0.0
        self.calls = 0
        if session is None:
            import requests
            session = requests.Session()
            session.headers["User-Agent"] = "MarsLab-background-fetch/1.0 (nightly; polite)"
        self.session = session

    def query(self, **params) -> Optional[Dict]:
        params = {"target": "mars", "output": "json", **params}
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                r = self.session.get(self.base, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    status = data.get("ODEResults", {}).get("Status")
                    if status == "Success":
                        return data
                    err = data.get("ODEResults", {}).get("Error", status)
                    kv("ode_error", logging.WARNING, attempt=attempt + 1, error=err)
                    if "not available" in str(err):
                        return None  # permanent (bad parameter combination)
                else:
                    kv("ode_http", logging.WARNING, attempt=attempt + 1, status=r.status_code)
            except Exception as e:  # network errors, JSON decode errors
                kv("ode_exception", logging.WARNING, attempt=attempt + 1,
                   error=f"{type(e).__name__}: {e}"[:200])
            if attempt < self.retries - 1:
                self.sleep(min(300.0, 5.0 * 2 ** attempt))
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Discovery per instrument
# ═════════════════════════════════════════════════════════════════════════════

def _query_boxes(client: OdeClient, site: Site, half_km: float, **params) -> Tuple[List[Dict], bool]:
    """Query every bbox around the site; returns (products, all_queries_ok)."""
    out, ok = [], True
    for minlat, maxlat, w, e in fl.ode_bboxes(site.lat, site.lon, half_km):
        data = client.query(minlat=minlat, maxlat=maxlat, westernlon=w, easternlon=e,
                            limit=ODE_LIMIT, **params)
        if data is None:
            ok = False
            continue
        out.extend(fl.ode_products(data))
    return out, ok


def _kb(f: Dict) -> int:
    try:
        return int(float(f.get("KBytes", 0)) * 1024)
    except (TypeError, ValueError):
        return 0


def _sharad_candidates(prods: List[Dict], site: Site, paths: fl.DataPaths) -> List[Candidate]:
    out = {}
    for p in prods:
        wkt = p.get("Footprint_C0_geometry") or p.get("Footprint_geometry") or ""
        parts = fl.parse_wkt_parts(wkt)
        if not parts:
            continue
        dist = min(fl.point_to_polyline_km(site.lon, site.lat, line) for line in parts)
        if dist > SHARAD_MAX_KM:
            continue
        label_url = p.get("LabelURL", "")
        pid = (p.get("pdsid") or label_url.rsplit("/", 1)[-1].rsplit(".", 1)[0]).upper()
        if not pid or pid in out:
            continue
        dat_url = lbl_url = None
        est = 0
        for f in fl.product_files(p):
            fn = f.get("FileName", "").upper()
            if fn == f"{pid}.DAT":
                dat_url, est = f.get("URL"), _kb(f)
            elif fn == f"{pid}.LBL":
                lbl_url = f.get("URL")
        if label_url:
            lbl_url = lbl_url or label_url
            dat_url = dat_url or f"{label_url.rsplit('/', 1)[0]}/{pid.lower()}.dat"
        if not dat_url or not lbl_url:
            continue
        out[pid] = Candidate(
            instrument="sharad", product_id=pid, site_id=site.id,
            est_bytes=est or NOMINAL_BYTES["sharad"], distance_km=dist,
            dest_dir=paths.sharad_dir,
            files=[(dat_url, f"{pid.lower()}.dat"), (lbl_url, f"{pid.lower()}.lbl")],
            meta={"pid": pid, "lat": fl._float(p.get("Center_latitude")),
                  "lon": fl._float(p.get("Center_longitude")), "footprint_wkt": wkt},
        )
    return list(out.values())


def _hirise_candidate(p: Dict, site: Site, paths: fl.DataPaths, dist: float) -> Optional[Candidate]:
    pdsid = p.get("pdsid", "").upper()
    if not pdsid.endswith("_RED"):
        return None
    base_id = pdsid[:-4]
    jp2 = lbl = None
    for f in fl.product_files(p):
        fn = f.get("FileName", "")
        u = fn.upper()
        if u == f"{base_id}_RED.JP2":
            jp2 = (f.get("URL"), fn, _kb(f))
        elif u == f"{base_id}_RED.LBL":
            lbl = (f.get("URL"), fn)
    if not jp2:
        return None
    files = [(jp2[0], jp2[1])] + ([lbl] if lbl else [])
    return Candidate(
        instrument="hirise", product_id=base_id, site_id=site.id,
        est_bytes=jp2[2] or NOMINAL_BYTES["hirise"], distance_km=dist,
        dest_dir=paths.hirise_dir, files=files,
        meta={"base_id": base_id, "jp2_fn": jp2[1],
              "lat": fl._float(p.get("Center_latitude")),
              "lon": fl._float(p.get("Center_longitude")),
              "footprint_wkt": p.get("Footprint_C0_geometry", "")},
    )


def _crism_candidate(p: Dict, site: Site, paths: fl.DataPaths, dist: float) -> Optional[Candidate]:
    pdsid = p.get("pdsid", "").lower()
    if not pdsid.startswith(CRISM_TARGETED) or not fl._L_TRR3_RE.search(pdsid + ".img"):
        return None
    bk = fl.extract_base_key(pdsid)
    img = lbl = None
    for f in fl.product_files(p):
        u = f.get("FileName", "").upper()
        if u.endswith("_TRR3.IMG"):
            img = (f.get("URL"), f.get("FileName"), _kb(f))
        elif u.endswith("_TRR3.LBL"):
            lbl = (f.get("URL"), f.get("FileName"))
    if not img or not lbl:
        return None
    nominal = NOMINAL_BYTES["crism_half" if pdsid.startswith(("hrl", "hrs")) else "crism_full"]
    return Candidate(
        instrument="crism", product_id=bk, site_id=site.id,
        est_bytes=max(img[2], nominal), distance_km=dist,
        dest_dir=paths.crism_dir / bk, files=[(img[0], img[1]), lbl],
        meta={"base_key": bk, "trr3_pdsid": pdsid,
              "footprint_wkt": p.get("Footprint_C0_geometry", "")},
    )


def _ctx_candidate(p: Dict, site: Site, paths: fl.DataPaths, dist: float) -> Optional[Candidate]:
    pid = p.get("pdsid", "").upper()
    if not pid:
        return None
    img = None
    for f in fl.product_files(p):
        if f.get("FileName", "").upper() == f"{pid}.IMG":
            img = (f.get("URL"), f"{pid}.IMG", _kb(f))
    if not img:
        return None
    return Candidate(
        instrument="ctx", product_id=pid, site_id=site.id,
        est_bytes=img[2] or NOMINAL_BYTES["ctx"], distance_km=dist,
        dest_dir=paths.ctx_dir, files=[(img[0], img[1])],
        meta={"pid": pid, "lat": fl._float(p.get("Center_latitude")),
              "lon": fl._float(p.get("Center_longitude")),
              "footprint_wkt": p.get("Footprint_C0_geometry", "")},
    )


FOOTPRINT_QUERIES = {
    "hirise": (dict(ihid="mro", iid="hirise", pt="RDRV11", results="fpm"), _hirise_candidate),
    "crism": (dict(ihid="mro", iid="crism", pt="TRDR", results="fpm", productid="*IF*L_TRR3"),
              _crism_candidate),
    "ctx": (dict(ihid="mro", iid="ctx", pt="EDR", results="fpm"), _ctx_candidate),
}


def _footprint_candidates(prods: List[Dict], site: Site, paths: fl.DataPaths, inst: str,
                          max_km: float) -> List[Candidate]:
    make = FOOTPRINT_QUERIES[inst][1]
    out: Dict[str, Candidate] = {}
    for p in prods:
        wkt = p.get("Footprint_C0_geometry") or p.get("Footprint_geometry") or ""
        dist = fl.footprint_distance_km(wkt, site.lon, site.lat)
        if dist > max_km:
            continue
        c = make(p, site, paths, dist)
        if c and (c.product_id not in out or dist < out[c.product_id].distance_km):
            out[c.product_id] = c
    return list(out.values())


def discover(client: OdeClient, site: Site, inst: str, paths: fl.DataPaths) -> Tuple[List[Candidate], bool, bool]:
    """Products covering the site. Returns (candidates sorted by distance, fallback_used, ok)."""
    if inst == "sharad":
        prods, ok = _query_boxes(client, site, SHARAD_MAX_KM,
                                 ihid="mro", iid="sharad", pt="RDR", results="fpm")
        cands, fallback = _sharad_candidates(prods, site, paths), False
    else:
        params = FOOTPRINT_QUERIES[inst][0]
        prods, ok = _query_boxes(client, site, POINT_HALF_KM, **params)
        cands, fallback = _footprint_candidates(prods, site, paths, inst, 0.0), False
        if not cands and ok:
            prods, ok = _query_boxes(client, site, site.radius_km, **params)
            cands = _footprint_candidates(prods, site, paths, inst, site.radius_km)
            fallback = bool(cands)
    cands.sort(key=lambda c: (c.distance_km, c.product_id))
    return cands, fallback, ok


class LocalChecker:
    """Answers "is this product already complete on disk?" per instrument."""

    def __init__(self, paths: fl.DataPaths):
        self.paths = paths
        self._sharad: Optional[set] = None

    def is_local(self, c: Candidate) -> bool:
        if c.instrument == "sharad":
            if self._sharad is None:
                self._sharad = fl.sharad_local_ids(self.paths.sharad_dir)
            return c.product_id in self._sharad
        if c.instrument == "hirise":
            return fl.hirise_is_local(self.paths.hirise_dir, c.product_id)
        if c.instrument == "crism":
            return fl.crism_is_local(self.paths.crism_dir, c.product_id)
        if c.instrument == "ctx":
            return fl.ctx_is_local(self.paths.ctx_dir, c.product_id)
        return False

    def mark_local(self, c: Candidate) -> None:
        if c.instrument == "sharad" and self._sharad is not None:
            self._sharad.add(c.product_id)


# ═════════════════════════════════════════════════════════════════════════════
# Planning (fairness)
# ═════════════════════════════════════════════════════════════════════════════

def plan_downloads(pairs: List[PairStatus], per_pair_cap: int,
                   instrument_order: Tuple[str, ...] = INSTRUMENTS) -> List[Candidate]:
    """Round-robin plan over site x instrument pairs.

    Round r gives each pair (that still has missing products and has taken
    fewer than per_pair_cap) one product. Inside a round, pairs are ordered
    by fewest local products, then site priority, instrument order, site id.
    A product that covers several sites is only planned once.
    """
    def order(p: PairStatus):
        inst_rank = instrument_order.index(p.instrument) if p.instrument in instrument_order else 99
        return (p.local, p.site.priority, inst_rank, p.site.id)

    active = sorted([p for p in pairs if p.missing], key=order)
    ptr = {p.key: 0 for p in active}
    taken = {p.key: 0 for p in active}
    seen = set()
    plan: List[Candidate] = []
    while True:
        progressed = False
        for p in active:
            if taken[p.key] >= per_pair_cap:
                continue
            while ptr[p.key] < len(p.missing):
                c = p.missing[ptr[p.key]]
                ptr[p.key] += 1
                if c.key in seen:
                    continue
                seen.add(c.key)
                plan.append(c)
                taken[p.key] += 1
                progressed = True
                break
        if not progressed:
            return plan


# ═════════════════════════════════════════════════════════════════════════════
# State
# ═════════════════════════════════════════════════════════════════════════════

def load_state(path: Path) -> Dict:
    try:
        with open(path) as f:
            st = json.load(f)
        if isinstance(st, dict):
            st.setdefault("pairs", {})
            st.setdefault("failures", {})
            st.setdefault("history", [])
            return st
    except (OSError, ValueError):
        pass
    return {"pairs": {}, "failures": {}, "history": []}


def save_state(path: Path, state: Dict) -> None:
    state["updated"] = utcnow()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fetcher
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Settings:
    min_free_bytes: float = DEFAULT_MIN_FREE_GB * GB
    max_run_bytes: float = DEFAULT_MAX_RUN_GB * GB
    max_seconds: float = DEFAULT_MAX_HOURS * 3600
    per_pair_cap: int = DEFAULT_PER_PAIR_CAP
    connections: int = 4
    skip_gdal: bool = False
    dry_run: bool = False


class Fetcher:
    def __init__(self, paths: fl.DataPaths, client: OdeClient, settings: Settings,
                 state: Dict, downloader=fl.aria2_product,
                 free_bytes: Optional[Callable[[Path], int]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.paths = paths
        self.client = client
        self.s = settings
        self.state = state
        self.downloader = downloader
        self.free_bytes = free_bytes or (lambda p: shutil.disk_usage(p).free)
        self.clock = clock
        self.t0 = clock()
        self.checker = LocalChecker(paths)
        self.run_bytes = 0
        self.summary = {"downloaded": 0, "failed": 0, "skipped_cap": 0, "bytes": 0,
                        "per_instrument": {}, "stop_reason": None}
        self.current: Optional[Candidate] = None

    # ── guards ─────────────────────────────────────────────────────────────
    def remaining_s(self) -> float:
        return self.s.max_seconds - (self.clock() - self.t0)

    def _free(self, path: Path) -> int:
        p = path
        while not p.exists() and p != p.parent:
            p = p.parent
        return self.free_bytes(p)

    def disk_ok(self, extra: float = 0) -> bool:
        return self._free(self.paths.root) - extra >= self.s.min_free_bytes

    # ── discovery ──────────────────────────────────────────────────────────
    def discover_all(self, sites: List[Site], instruments: List[str]) -> List[PairStatus]:
        failures = self.state.get("failures", {})
        pairs = []
        for site in sites:
            for inst in instruments:
                if self.remaining_s() <= 0:
                    kv("discovery_timeout", logging.WARNING, site=site.id, instrument=inst)
                    return pairs
                ps = PairStatus(site=site, instrument=inst)
                cands, ps.fallback, ok = discover(self.client, site, inst, self.paths)
                if not ok:
                    ps.error = "ode_query_failed"
                ps.covering = len(cands)
                for c in cands:
                    if self.checker.is_local(c):
                        ps.local += 1
                    elif failures.get(c.key, {}).get("count", 0) >= MAX_FAILURES:
                        continue
                    else:
                        ps.missing.append(c)
                kv("discovered", site=site.id, instrument=inst, covering=ps.covering,
                   local=ps.local, missing=len(ps.missing), fallback=ps.fallback,
                   error=ps.error or "-")
                pairs.append(ps)
                if not self.s.dry_run:
                    prev = self.state["pairs"].get(ps.key, {})
                    self.state["pairs"][ps.key] = {
                        "covering": ps.covering, "local": ps.local,
                        "missing": len(ps.missing), "fallback": ps.fallback,
                        "error": ps.error, "last_checked": utcnow(),
                        "downloaded_total": prev.get("downloaded_total", 0),
                    }
        return pairs

    # ── download ───────────────────────────────────────────────────────────
    def _resolve_crism_ddr(self, c: Candidate) -> bool:
        ddr = fl.derive_ddr_pdsid(c.meta["trr3_pdsid"])
        if not ddr:
            return False
        data = self.client.query(ihid="mro", iid="crism", productid=ddr, results="fp", limit=1)
        img = lbl = None
        for p in fl.ode_products(data)[:1]:
            for f in fl.product_files(p):
                u = f.get("FileName", "").upper()
                if u.endswith("_DDR1.IMG"):
                    img = (f.get("URL"), f.get("FileName"))
                elif u.endswith("_DDR1.LBL"):
                    lbl = (f.get("URL"), f.get("FileName"))
        if not img or not lbl:
            return False
        c.files = [f for f in c.files if "_DDR1." not in f[1].upper()] + [img, lbl]
        return True

    def _update_index(self, c: Candidate) -> int:
        p = self.paths
        if c.instrument == "sharad":
            return fl.update_sharad_index([c.meta], p.sharad_dir, p.sharad_index)[0]
        if c.instrument == "hirise":
            return fl.update_hirise_index([c.meta], p.hirise_dir, p.hirise_index)[0]
        if c.instrument == "crism":
            return fl.update_crism_index([c.meta], p.crism_dir, p.crism_index)[0]
        if c.instrument == "ctx":
            return fl.update_ctx_index([c.meta], p.ctx_dir, p.ctx_index)[0]
        return 0

    def _record_failure(self, c: Candidate, err: str) -> None:
        f = self.state["failures"].setdefault(c.key, {"count": 0})
        f["count"] += 1
        f["last_error"] = err[:300]
        f["last_attempt"] = utcnow()
        self.summary["failed"] += 1
        kv("download_failed", logging.WARNING, site=c.site_id, instrument=c.instrument,
           product=c.product_id, attempts=f["count"], error=err[:300])

    def download_one(self, c: Candidate) -> bool:
        self.current = c
        triples: List[Tuple[str, str, str]] = []
        try:
            if c.instrument == "crism" and not self._resolve_crism_ddr(c):
                self._record_failure(c, "no DDR1 product found on ODE")
                return False
            c.dest_dir.mkdir(parents=True, exist_ok=True)
            triples = [(url, str(c.dest_dir), fn) for url, fn in c.files
                       if not fl._complete(c.dest_dir / fn)]
            kv("download_start", site=c.site_id, instrument=c.instrument, product=c.product_id,
               files=len(triples), est_mb=round(c.est_bytes / 1024 ** 2))
            t = time.monotonic()
            try:
                ok, msg = self.downloader(triples, connections=self.s.connections,
                                          timeout_s=max(1.0, self.remaining_s()))
            except subprocess.TimeoutExpired:
                ok, msg = False, "wall-clock limit reached during download"
            nbytes = sum((c.dest_dir / fn).stat().st_size for _, _, fn in triples
                         if (c.dest_dir / fn).exists())
            self.run_bytes += nbytes
            if not ok:
                fl.cleanup_partial(triples)
                if c.instrument == "crism" and c.dest_dir.exists() and not any(c.dest_dir.iterdir()):
                    c.dest_dir.rmdir()
                self._record_failure(c, msg)
                return False

            if c.instrument == "hirise" and not self.s.skip_gdal:
                jp2 = c.dest_dir / c.meta["jp2_fn"]
                try:
                    tif = fl.convert_jp2_to_tif(jp2)
                except BaseException:
                    jp2.with_suffix(".tif").unlink(missing_ok=True)
                    fl.update_hirise_index([c.meta], self.paths.hirise_dir, self.paths.hirise_index)
                    raise
                kv("converted", product=c.product_id, tif=bool(tif))

            added = self._update_index(c)
            self.checker.mark_local(c)
            self.state["failures"].pop(c.key, None)
            pair = self.state["pairs"].setdefault(f"{c.site_id}/{c.instrument}", {})
            pair["downloaded_total"] = pair.get("downloaded_total", 0) + 1
            pi = self.summary["per_instrument"].setdefault(c.instrument, {"n": 0, "bytes": 0})
            pi["n"] += 1
            pi["bytes"] += nbytes
            self.summary["downloaded"] += 1
            kv("download_ok", site=c.site_id, instrument=c.instrument, product=c.product_id,
               mb=round(nbytes / 1024 ** 2, 1), secs=round(time.monotonic() - t),
               index_added=added, run_gb=round(self.run_bytes / GB, 2))
            return True
        except Interrupted:
            removed = fl.cleanup_partial(triples)
            kv("download_interrupted", logging.WARNING, site=c.site_id, instrument=c.instrument,
               product=c.product_id, partial_mb_removed=round(removed / 1024 ** 2, 1))
            raise
        finally:
            self.current = None
            self.summary["bytes"] = self.run_bytes

    def execute(self, plan: List[Candidate]) -> None:
        for c in plan:
            if self.remaining_s() < 60:
                self.summary["stop_reason"] = "time_limit"
                break
            if self.run_bytes >= self.s.max_run_bytes:
                self.summary["stop_reason"] = "run_size_cap"
                break
            if not self.disk_ok(c.est_bytes):
                self.summary["stop_reason"] = "disk_free_below_min"
                kv("guard_disk", logging.WARNING,
                   free_gb=round(self._free(self.paths.root) / GB, 1),
                   min_free_gb=round(self.s.min_free_bytes / GB, 1))
                break
            if self.run_bytes + c.est_bytes > self.s.max_run_bytes:
                self.summary["skipped_cap"] += 1
                kv("skip_would_exceed_cap", product=c.product_id, instrument=c.instrument,
                   est_mb=round(c.est_bytes / 1024 ** 2))
                continue
            if self.checker.is_local(c):  # e.g. fetched for another site earlier
                continue
            self.download_one(c)
        if self.summary["stop_reason"]:
            kv("stopped", logging.WARNING, reason=self.summary["stop_reason"],
               run_gb=round(self.run_bytes / GB, 2))


# ═════════════════════════════════════════════════════════════════════════════
# Lock / reporting / main
# ═════════════════════════════════════════════════════════════════════════════

class SingleInstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            return False
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"{os.getpid()}\n")
        self.fh.flush()
        return True

    def release(self) -> None:
        if self.fh:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


def print_dry_run(pairs: List[PairStatus], plan: List[Candidate], settings: Settings) -> None:
    print("\nCoverage per site x instrument (ODE products covering the site point):")
    print(f"  {'site':<12} {'instr':<7} {'covering':>8} {'local':>6} {'missing':>8}  note")
    for p in pairs:
        note = "fallback radius" if p.fallback else ""
        if p.error:
            note = (note + " " + p.error).strip()
        print(f"  {p.site.id:<12} {p.instrument:<7} {p.covering:>8} {p.local:>6} "
              f"{len(p.missing):>8}  {note}")
    budget = settings.max_run_bytes
    run, fit = 0, []
    for c in plan:
        if run + c.est_bytes <= budget:
            run += c.est_bytes
            fit.append(c)
    print(f"\nPlan (per-pair cap {settings.per_pair_cap}): {len(plan)} products, "
          f"{sum(c.est_bytes for c in plan) / GB:.1f} GB estimated; "
          f"{len(fit)} fit in the {budget / GB:.0f} GB run cap ({run / GB:.1f} GB)")
    by_inst: Dict[str, List[int]] = {}
    for c in fit:
        by_inst.setdefault(c.instrument, [0, 0])
        by_inst[c.instrument][0] += 1
        by_inst[c.instrument][1] += c.est_bytes
    for inst, (n, b) in sorted(by_inst.items()):
        print(f"  {inst:<7} {n:>4} products  ~{b / GB:7.2f} GB")
    print("\nDownload order (first 40):")
    for i, c in enumerate(fit[:40], 1):
        print(f"  {i:>3}. {c.site_id:<11} {c.instrument:<7} {c.product_id:<28} "
              f"{c.est_bytes / 1024 ** 2:>8.0f} MB  d={c.distance_km:.1f} km")


def post_repair_index(url: str = REPAIR_URL) -> None:
    try:
        import requests
        r = requests.post(url, timeout=600)
        kv("repair_index", status=r.status_code, body=r.text[:200])
    except Exception as e:
        kv("repair_index_skipped", error=f"{type(e).__name__}"[:100])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MarsLab unattended background data fetcher")
    p.add_argument("--dry-run", action="store_true", help="Discover + plan only; write nothing")
    p.add_argument("--site", action="append", help="Limit to site id/name (repeatable)")
    p.add_argument("--instrument", action="append", choices=INSTRUMENTS,
                   help="Limit to instrument (repeatable; 'ctx' is opt-in)")
    p.add_argument("--max-run-gb", type=float, default=DEFAULT_MAX_RUN_GB)
    p.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    p.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    p.add_argument("--per-pair-cap", type=int, default=DEFAULT_PER_PAIR_CAP,
                   help="Max new products per site x instrument per run")
    p.add_argument("--data-root", default=os.environ.get("MARSLAB_DATA_ROOT", str(BACKEND_DIR)),
                   help="Directory holding hirise_data/, sharad_highres/, mineral_cnn_data/ ... "
                        "(env MARSLAB_DATA_ROOT; default: this backend/)")
    p.add_argument("--sites-file", default=str(DEFAULT_SITES_FILE))
    p.add_argument("--skip-gdal", action="store_true", help="Skip HiRISE JP2 -> GeoTIFF")
    p.add_argument("--connections", type=int, default=4, help="aria2 connections per file")
    p.add_argument("--no-repair", action="store_true", help="Don't POST /api/repair-index")
    return p.parse_args(argv)


def setup_logging(log_file: Optional[Path]) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z")
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    if not log_file or sys.stdout.isatty():
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)


def main(argv=None, client: Optional[OdeClient] = None, downloader=None,
         free_bytes=None) -> int:
    args = parse_args(argv)
    paths = fl.DataPaths(Path(args.data_root).resolve())
    settings = Settings(
        min_free_bytes=args.min_free_gb * GB, max_run_bytes=args.max_run_gb * GB,
        max_seconds=args.max_hours * 3600, per_pair_cap=args.per_pair_cap,
        connections=args.connections, skip_gdal=args.skip_gdal, dry_run=args.dry_run,
    )
    # Dry run never writes into the data root (no log file, lock or state).
    setup_logging(None if args.dry_run else paths.logs_dir / "background_fetch.log")

    sites, cfg_instruments = load_sites(Path(args.sites_file))
    sites = select_sites(sites, args.site)
    instruments = args.instrument or cfg_instruments

    state_path = paths.logs_dir / "background_fetch_state.json"
    state = load_state(state_path)
    lock = SingleInstanceLock(paths.logs_dir / "background_fetch.lock")
    if not args.dry_run and not lock.acquire():
        kv("already_running", logging.WARNING, lock=str(lock.path))
        return 3

    fetcher = Fetcher(paths, client or OdeClient(), settings, state,
                      downloader=downloader or fl.aria2_product, free_bytes=free_bytes)
    run = {"started": utcnow(), "dry_run": args.dry_run, "sites": [s.id for s in sites],
           "instruments": instruments}
    kv("run_start", data_root=str(paths.root), sites=",".join(run["sites"]),
       instruments=",".join(instruments), dry_run=args.dry_run,
       free_gb=round(fetcher._free(paths.root) / GB, 1), min_free_gb=args.min_free_gb,
       max_run_gb=args.max_run_gb, max_hours=args.max_hours)

    def _on_signal(signum, _frame):
        raise Interrupted(signal.Signals(signum).name)

    old = {s: signal.signal(s, _on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    rc = 0
    try:
        if not fetcher.disk_ok():
            kv("refused_disk_free", logging.ERROR,
               free_gb=round(fetcher._free(paths.root) / GB, 1), min_free_gb=args.min_free_gb)
            if not args.dry_run:
                run["status"] = "refused_disk_free"
                return 2
            print("WARNING: free space below --min-free-gb; a real run would refuse to start.")
        pairs = fetcher.discover_all(sites, instruments)
        plan = plan_downloads(pairs, settings.per_pair_cap, tuple(instruments))
        kv("planned", products=len(plan), est_gb=round(sum(c.est_bytes for c in plan) / GB, 2),
           ode_calls=fetcher.client.calls)
        if args.dry_run:
            print_dry_run(pairs, plan, settings)
            run["status"] = "dry_run"
            return 0
        fetcher.execute(plan)
        run["status"] = "stopped:" + fetcher.summary["stop_reason"] if fetcher.summary["stop_reason"] else "ok"
    except Interrupted as e:
        kv("interrupted", logging.WARNING, signal=str(e),
           product=fetcher.current.key if fetcher.current else "-")
        run["status"] = f"interrupted:{e}"
        rc = 0
    except Exception as e:
        log.exception("unhandled error")
        run["status"] = f"error:{type(e).__name__}: {e}"[:300]
        rc = 1
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        run.update(fetcher.summary, finished=utcnow(), ode_calls=fetcher.client.calls)
        if not args.dry_run:
            state["last_run"] = run
            state["history"] = (state.get("history", []) + [run])[-30:]
            try:
                save_state(state_path, state)
            except OSError as e:
                kv("state_save_failed", logging.ERROR, error=str(e)[:200])
            lock.release()
        kv("run_end", status=run.get("status"), downloaded=fetcher.summary["downloaded"],
           failed=fetcher.summary["failed"], gb=round(fetcher.run_bytes / GB, 2))

    if fetcher.summary["downloaded"] and not args.no_repair:
        post_repair_index()
    return rc


if __name__ == "__main__":
    sys.exit(main())
