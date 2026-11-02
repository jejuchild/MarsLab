"""Unit tests for scripts/background_fetch.py (no network, tmp data root)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

import background_fetch as bf  # noqa: E402
import fetch_lib as fl  # noqa: E402

GB = 1024 ** 3
JEZERO = bf.Site(id="jezero", name="Jezero Crater", lat=18.41, lon=77.69, radius_km=10, priority=1)
GALE = bf.Site(id="gale", name="Gale Crater", lat=-5.335, lon=137.65, radius_km=15, priority=1)


# ─── fake ODE ────────────────────────────────────────────────────────────────

def _square(lon, lat, d=0.05):
    return (f"POLYGON (({lon - d} {lat - d}, {lon + d} {lat - d}, {lon + d} {lat + d}, "
            f"{lon - d} {lat + d}, {lon - d} {lat - d}))")


def sharad_product(pid, lon, lat, offset_km=0.0):
    x = lon + fl.km_to_deg(lat, offset_km)[1]
    return {
        "pdsid": pid.upper(),
        "LabelURL": f"https://pds.example/rdr/{pid.lower()}.lbl",
        "Center_latitude": str(lat), "Center_longitude": str(lon),
        "Footprint_C0_geometry": f"LINESTRING ({x} {lat - 2}, {x} {lat + 2})",
        "Product_files": {"Product_file": [
            {"FileName": f"{pid.upper()}.DAT", "KBytes": "60000", "URL": f"https://pds.example/{pid.lower()}.dat"},
            {"FileName": f"{pid.upper()}.LBL", "KBytes": "6", "URL": f"https://pds.example/{pid.lower()}.lbl"},
        ]},
    }


def hirise_product(base, lon, lat, kb=500000):
    return {
        "pdsid": f"{base}_RED", "Center_latitude": str(lat), "Center_longitude": str(lon % 360),
        "Footprint_C0_geometry": _square(lon, lat),
        "Product_files": {"Product_file": [
            {"FileName": f"{base}_RED.JP2", "KBytes": str(kb), "URL": f"https://hirise.example/{base}_RED.JP2"},
            {"FileName": f"{base}_RED.LBL", "KBytes": "9", "URL": f"https://hirise.example/{base}_RED.LBL"},
            {"FileName": f"{base}_RED.QLOOK.JP2", "KBytes": "9", "URL": "https://x/q.jp2"},
        ]},
    }


def crism_product(pdsid, lon, lat):
    up = pdsid.upper()
    return {
        "pdsid": pdsid.lower(), "Footprint_C0_geometry": _square(lon, lat),
        "Product_files": {"Product_file": [
            {"FileName": f"{up}.IMG", "KBytes": "1644", "URL": f"https://crism.example/{up}.IMG"},
            {"FileName": f"{up}.LBL", "KBytes": "10", "URL": f"https://crism.example/{up}.LBL"},
        ]},
    }


def ddr_product(pdsid):
    up = pdsid.upper()
    return {"pdsid": pdsid, "Product_files": {"Product_file": [
        {"FileName": f"{up}.IMG", "KBytes": "19000", "URL": f"https://crism.example/{up}.IMG"},
        {"FileName": f"{up}.LBL", "KBytes": "5", "URL": f"https://crism.example/{up}.LBL"},
    ]}}


def wrap(products):
    return {"ODEResults": {"Status": "Success", "Products": {"Product": products}}}


class FakeClient:
    """Routes queries by iid (geometry filtering is the code under test)."""

    def __init__(self, by_iid):
        self.by_iid = by_iid
        self.calls = 0
        self.queries = []

    def query(self, **params):
        self.calls += 1
        self.queries.append(params)
        if "productid" in params and params.get("iid") == "crism" and "pt" not in params:
            return wrap([ddr_product(params["productid"])])
        return wrap(list(self.by_iid.get(params["iid"], [])))


def fake_downloader(written, size=1000):
    def _dl(triples, connections=4, timeout_s=None):
        for url, d, fn in triples:
            Path(d).mkdir(parents=True, exist_ok=True)
            with open(Path(d) / fn, "wb") as fh:
                fh.truncate(size)  # sparse: "big" files cost nothing
            written.append(fn)
        return True, "ok"
    return _dl


def make_fetcher(tmp_path, client, free=5000 * GB, **settings):
    paths = fl.DataPaths(tmp_path)
    s = bf.Settings(**settings)
    written = []
    f = bf.Fetcher(paths, client, s, bf.load_state(tmp_path / "state.json"),
                   downloader=fake_downloader(written), free_bytes=lambda p: free)
    return f, written


def pair(site, inst, local, n, est=GB):
    ps = bf.PairStatus(site=site, instrument=inst, local=local)
    ps.missing = [bf.Candidate(inst, f"{site.id}-{inst}-{i}", site.id, est, 0.0, Path("/x"), [])
                  for i in range(n)]
    return ps


# ─── geometry ────────────────────────────────────────────────────────────────

def test_point_in_polygon_and_antimeridian():
    assert fl.footprint_contains(_square(77.69, 18.41), 77.69, 18.41)
    assert not fl.footprint_contains(_square(77.69, 18.41), 78.0, 18.41)
    ring = "POLYGON ((179.9 49, -179.9 49, -179.9 49.2, 179.9 49.2, 179.9 49))"
    assert fl.footprint_contains(ring, -179.95, 49.1)
    assert fl.footprint_contains(ring, 179.95, 49.1)


def test_sharad_track_distance_filter():
    near = sharad_product("R_0000001_001_SS19_700_A", 77.69, 18.41, offset_km=3)
    far = sharad_product("R_0000002_001_SS19_700_A", 77.69, 18.41, offset_km=8)
    cands = bf._sharad_candidates([near, far], JEZERO, fl.DataPaths(Path("/tmp/x")))
    assert [c.product_id for c in cands] == ["R_0000001_001_SS19_700_A"]
    assert 2.5 < cands[0].distance_km < 3.5
    assert cands[0].files[0][1] == "r_0000001_001_ss19_700_a.dat"


def test_ode_bbox_splits_at_prime_meridian():
    boxes = fl.ode_bboxes(2.39, -0.01, 5)
    assert len(boxes) == 2 and boxes[0][3] == 360.0 and boxes[1][2] == 0.0


# ─── planning / fairness ─────────────────────────────────────────────────────

def test_plan_round_robin_fewest_local_first():
    pairs = [pair(JEZERO, "hirise", local=10, n=3),
             pair(GALE, "hirise", local=0, n=3),
             pair(JEZERO, "sharad", local=2, n=1)]
    plan = bf.plan_downloads(pairs, per_pair_cap=2)
    ids = [c.product_id for c in plan]
    # round 1: fewest local first; round 2: remaining pairs again
    assert ids == ["gale-hirise-0", "jezero-sharad-0", "jezero-hirise-0",
                   "gale-hirise-1", "jezero-hirise-1"]


def test_plan_priority_tiebreak_and_dedup():
    a = pair(JEZERO, "crism", local=0, n=2)
    low = bf.Site(id="nili", name="Nili", lat=22, lon=76.9, priority=3)
    b = pair(low, "crism", local=0, n=0)
    b.missing = [a.missing[0], bf.Candidate("crism", "nili-only", "nili", GB, 0, Path("/x"), [])]
    plan = bf.plan_downloads([b, a], per_pair_cap=5)
    assert [c.product_id for c in plan] == ["jezero-crism-0", "nili-only", "jezero-crism-1"]


# ─── discovery / skip local ──────────────────────────────────────────────────

def test_discovery_skips_already_local(tmp_path):
    hp = [hirise_product("ESP_000001_1985", 77.69, 18.41), hirise_product("ESP_000002_1985", 77.69, 18.41)]
    (tmp_path / "hirise_data").mkdir()
    (tmp_path / "hirise_data" / "ESP_000001_1985_RED.JP2").write_bytes(b"j")
    # incomplete download (aria2 control file present) is NOT local
    (tmp_path / "hirise_data" / "ESP_000002_1985_RED.JP2").write_bytes(b"j")
    (tmp_path / "hirise_data" / "ESP_000002_1985_RED.JP2.aria2").write_bytes(b"c")
    sp = sharad_product("R_0000001_001_SS19_700_A", 77.69, 18.41)
    (tmp_path / "sharad_highres").mkdir()
    (tmp_path / "sharad_highres" / "r_0000001_001_ss19_700_a.dat").write_bytes(b"d")
    f, _ = make_fetcher(tmp_path, FakeClient({"hirise": hp, "sharad": [sp]}))
    pairs = {p.key: p for p in f.discover_all([JEZERO], ["hirise", "sharad"])}
    assert pairs["jezero/hirise"].covering == 2 and pairs["jezero/hirise"].local == 1
    assert [c.product_id for c in pairs["jezero/hirise"].missing] == ["ESP_000002_1985"]
    assert pairs["jezero/sharad"].local == 1 and not pairs["jezero/sharad"].missing


def test_discovery_fallback_radius(tmp_path):
    off = hirise_product("ESP_000003_1985", 77.80, 18.41)  # ~6 km east, not containing
    f, _ = make_fetcher(tmp_path, FakeClient({"hirise": [off]}))
    (p,) = f.discover_all([JEZERO], ["hirise"])
    assert p.fallback and len(p.missing) == 1 and p.missing[0].distance_km > 0


def test_crism_filters_untargeted_and_s_sensor(tmp_path):
    prods = [crism_product("frt000047a3_07_if156l_trr3", 77.69, 18.41),
             crism_product("frt000047a3_07_if156s_trr3", 77.69, 18.41),
             crism_product("msp00003e2f_01_if214l_trr3", 77.69, 18.41)]
    f, _ = make_fetcher(tmp_path, FakeClient({"crism": prods}))
    (p,) = f.discover_all([JEZERO], ["crism"])
    assert [c.product_id for c in p.missing] == ["frt000047a3_07"]
    assert p.missing[0].est_bytes >= bf.NOMINAL_BYTES["crism_full"]  # ODE KBytes ignored


# ─── downloads + index shape ─────────────────────────────────────────────────

def test_download_updates_indexes_with_existing_schema(tmp_path):
    client = FakeClient({
        "hirise": [hirise_product("ESP_000001_1985", 77.69, 18.41)],
        "sharad": [sharad_product("R_0000001_001_SS19_700_A", 77.69, 18.41)],
        "crism": [crism_product("frt000047a3_07_if156l_trr3", 77.69, 18.41)],
    })
    f, written = make_fetcher(tmp_path, client, skip_gdal=True)
    plan = bf.plan_downloads(f.discover_all([JEZERO], ["sharad", "crism", "hirise"]), 5)
    f.execute(plan)
    assert f.summary["downloaded"] == 3 and f.summary["failed"] == 0

    sh = json.loads((tmp_path / "sharad_highres_data" / "index.geojson").read_text())
    (feat,) = sh["features"]
    assert feat["geometry"]["type"] == "LineString"
    assert feat["properties"] == {"instrument": "SHARAD_HIGHRES", "product_id": "R_0000001_001_SS19_700_A",
                                  "dat_file": "r_0000001_001_ss19_700_a.dat",
                                  "lbl_file": "r_0000001_001_ss19_700_a.lbl"}

    hi = json.loads((tmp_path / "hirise_data" / "index.geojson").read_text())
    (feat,) = hi["features"]
    assert feat["geometry"]["type"] == "Polygon"
    assert set(feat["properties"]) == {"instrument", "product_id", "quicklook", "red_tif", "red_jp2",
                                       "center_latitude", "center_longitude"}
    assert feat["properties"]["red_jp2"] == "ESP_000001_1985_RED.JP2"

    cr = json.loads((tmp_path / "mineral_cnn_data" / "index.geojson").read_text())
    (feat,) = cr["features"]
    assert feat["properties"]["instrument"] == "CRISM_TRR3"
    assert feat["properties"]["product_id"] == feat["properties"]["obs_dir"] == "frt000047a3_07"
    assert feat["geometry"]["type"] == "Polygon"
    assert "FRT000047A3_07_DE156L_DDR1.IMG" in written

    # second pass: everything is local now → nothing missing
    f2, written2 = make_fetcher(tmp_path, client, skip_gdal=True)
    pairs = f2.discover_all([JEZERO], ["sharad", "crism", "hirise"])
    assert all(p.local == 1 and not p.missing for p in pairs)


def test_failed_download_cleans_partials_and_records_failure(tmp_path):
    client = FakeClient({"hirise": [hirise_product("ESP_000001_1985", 77.69, 18.41)]})
    f, _ = make_fetcher(tmp_path, client, skip_gdal=True)

    def broken(triples, connections=4, timeout_s=None):
        for _, d, fn in triples:
            Path(d, fn).write_bytes(b"partial")
            Path(d, fn + ".aria2").write_bytes(b"ctl")
        return False, "aria2c exit 1"

    f.downloader = broken
    f.execute(bf.plan_downloads(f.discover_all([JEZERO], ["hirise"]), 5))
    assert f.summary["failed"] == 1
    assert list((tmp_path / "hirise_data").iterdir()) == []
    assert f.state["failures"]["hirise:ESP_000001_1985"]["count"] == 1
    # after MAX_FAILURES the product is no longer planned
    f.state["failures"]["hirise:ESP_000001_1985"]["count"] = bf.MAX_FAILURES
    (p,) = f.discover_all([JEZERO], ["hirise"])
    assert p.missing == []


def test_timeout_during_download_is_cleaned(tmp_path):
    client = FakeClient({"hirise": [hirise_product("ESP_000001_1985", 77.69, 18.41)]})
    f, _ = make_fetcher(tmp_path, client, skip_gdal=True)

    def slow(triples, connections=4, timeout_s=None):
        for _, d, fn in triples:
            Path(d, fn).write_bytes(b"partial")
            Path(d, fn + ".aria2").write_bytes(b"ctl")
        raise subprocess.TimeoutExpired("aria2c", timeout_s)

    f.downloader = slow
    f.execute(bf.plan_downloads(f.discover_all([JEZERO], ["hirise"]), 5))
    assert f.summary["failed"] == 1 and list((tmp_path / "hirise_data").iterdir()) == []


# ─── guards ──────────────────────────────────────────────────────────────────

def test_guard_run_size_cap(tmp_path):
    prods = [hirise_product(f"ESP_00000{i}_1985", 77.69, 18.41, kb=400 * 1024) for i in range(5)]
    f, written = make_fetcher(tmp_path, FakeClient({"hirise": prods}), skip_gdal=True,
                              max_run_bytes=1.0 * GB)
    f.downloader = fake_downloader(written, size=200 * 1024 ** 2)  # JP2+LBL = 400 MB
    f.execute(bf.plan_downloads(f.discover_all([JEZERO], ["hirise"]), 5))
    # 400 MB each → after 2 (800 MB) the next would exceed 1 GB → skipped
    assert f.summary["downloaded"] == 2 and f.summary["skipped_cap"] == 3


def test_guard_stops_when_cap_reached(tmp_path):
    prods = [hirise_product(f"ESP_00000{i}_1985", 77.69, 18.41, kb=1) for i in range(3)]
    f, _ = make_fetcher(tmp_path, FakeClient({"hirise": prods}), skip_gdal=True, max_run_bytes=1500)
    f.execute(bf.plan_downloads(f.discover_all([JEZERO], ["hirise"]), 5))
    assert f.summary["stop_reason"] == "run_size_cap" and f.summary["downloaded"] == 1


def test_guard_disk_free_stops_downloads(tmp_path):
    prods = [hirise_product("ESP_000001_1985", 77.69, 18.41)]
    f, written = make_fetcher(tmp_path, FakeClient({"hirise": prods}), free=1000 * GB + 100 * 1024 ** 2,
                              min_free_bytes=1000 * GB, skip_gdal=True)
    f.execute(bf.plan_downloads(f.discover_all([JEZERO], ["hirise"]), 5))
    assert f.summary["stop_reason"] == "disk_free_below_min" and written == []


def test_guard_time_limit(tmp_path):
    prods = [hirise_product("ESP_000001_1985", 77.69, 18.41)]
    f, written = make_fetcher(tmp_path, FakeClient({"hirise": prods}), skip_gdal=True, max_seconds=30)
    f.execute(bf.plan_downloads(f.discover_all([JEZERO], ["hirise"]), 5))
    assert f.summary["stop_reason"] == "time_limit" and written == []


def _sites_file(tmp_path):
    p = tmp_path / "sites.json"
    p.write_text(json.dumps({"instruments": ["hirise"], "sites": [
        {"id": "jezero", "name": "Jezero Crater", "lat": 18.41, "lon": 77.69}]}))
    return str(p)


def test_main_refuses_when_disk_low(tmp_path):
    rc = bf.main(["--data-root", str(tmp_path), "--sites-file", _sites_file(tmp_path),
                  "--min-free-gb", "1000", "--no-repair"],
                 client=FakeClient({}), free_bytes=lambda p: 10 * GB)
    assert rc == 2
    state = json.loads((tmp_path / "logs" / "background_fetch_state.json").read_text())
    assert state["last_run"]["status"] == "refused_disk_free"
    assert "refused_disk_free" in (tmp_path / "logs" / "background_fetch.log").read_text()


def test_main_single_instance_lock(tmp_path):
    lock = bf.SingleInstanceLock(tmp_path / "logs" / "background_fetch.lock")
    assert lock.acquire()
    try:
        rc = bf.main(["--data-root", str(tmp_path), "--sites-file", _sites_file(tmp_path), "--no-repair"],
                     client=FakeClient({}), free_bytes=lambda p: 5000 * GB)
        assert rc == 3
    finally:
        lock.release()
    rc = bf.main(["--data-root", str(tmp_path), "--sites-file", _sites_file(tmp_path), "--no-repair"],
                 client=FakeClient({}), free_bytes=lambda p: 5000 * GB)
    assert rc == 0


def test_main_end_to_end_and_dry_run_writes_nothing(tmp_path):
    client = FakeClient({"hirise": [hirise_product("ESP_000001_1985", 77.69, 18.41)]})
    dl = []
    rc = bf.main(["--data-root", str(tmp_path), "--sites-file", _sites_file(tmp_path),
                  "--dry-run", "--no-repair"], client=client, downloader=fake_downloader(dl),
                 free_bytes=lambda p: 5000 * GB)
    assert rc == 0 and dl == [] and not (tmp_path / "logs").exists()

    rc = bf.main(["--data-root", str(tmp_path), "--sites-file", _sites_file(tmp_path),
                  "--skip-gdal", "--no-repair"], client=client, downloader=fake_downloader(dl),
                 free_bytes=lambda p: 5000 * GB)
    assert rc == 0 and "ESP_000001_1985_RED.JP2" in dl
    state = json.loads((tmp_path / "logs" / "background_fetch_state.json").read_text())
    assert state["last_run"]["downloaded"] == 1 and state["last_run"]["status"] == "ok"
    assert state["pairs"]["jezero/hirise"]["downloaded_total"] == 1


def test_ode_client_retries_with_backoff():
    class Resp:
        def __init__(self, code, data=None):
            self.status_code, self._d = code, data

        def json(self):
            return self._d

    class Sess:
        def __init__(self):
            self.n = 0

        def get(self, url, params=None, timeout=None):
            self.n += 1
            return Resp(503) if self.n < 3 else Resp(200, wrap([]))

    sleeps = []
    c = bf.OdeClient(min_interval=0, retries=4, sleep=sleeps.append, session=Sess())
    assert c.query(iid="hirise") is not None
    assert c.calls == 3 and sleeps[:2] == [5.0, 10.0]


def test_bulk_downloader_still_importable():
    import importlib
    mod = importlib.import_module("download_bulk_highres")
    assert callable(mod.run_aria2) and callable(mod._update_hirise_index)
