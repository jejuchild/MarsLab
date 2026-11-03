"""SHARAD hi-res router: file guards, cache keying and endpoint behaviour on
tiny synthetic RDR products (no real data needed)."""

import json
import os

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.sharad_highres_router as R
from analysis.shared.sharad_math import (
    IncompleteProductError,
    check_dat_complete,
    read_rwot,
)

PID = "R_9999901_001_SS19_700_A"
N_ROWS = 64
SURF_BIN = 150


def _write_product(d, pid=PID, n_rows=N_ROWS, surf_bin=SURF_BIN, label_rows=None, rwot0=9000.0,
                   lat_step=0.01, extra=None):
    """Synthetic RDR: one bright surface echo per trace at ``surf_bin`` (window
    relative), RWOT ramping by 1 count per row, lat 18.0→18.63 N, lon 77.45 E."""
    rows = np.zeros((n_rows, R.ROW_BYTES), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for i in range(n_rows):
        re = (rng.normal(0, 1, R.N_RANGE_BINS)).astype("<f4")
        im = (rng.normal(0, 1, R.N_RANGE_BINS)).astype("<f4")
        re[surf_bin] = 100.0
        re[surf_bin + 40] = 10.0  # weaker subsurface echo
        if extra is not None:
            for b, amp in extra(i).items():
                re[b] = amp
        rows[i, R.ECHO_REAL_OFFSET:R.ECHO_REAL_OFFSET + R.ECHO_BYTES] = np.frombuffer(re.tobytes(), np.uint8)
        rows[i, R.ECHO_IMAG_OFFSET:R.ECHO_IMAG_OFFSET + R.ECHO_BYTES] = np.frombuffer(im.tobytes(), np.uint8)
        rows[i, R.RWOT_OFFSET:R.RWOT_OFFSET + 4] = np.frombuffer(np.array([rwot0 + i], "<f4").tobytes(), np.uint8)
        rows[i, R.LAT_OFFSET:R.LAT_OFFSET + 8] = np.frombuffer(np.array([18.0 + lat_step * i], "<f8").tobytes(), np.uint8)
        rows[i, R.LON_OFFSET:R.LON_OFFSET + 8] = np.frombuffer(np.array([77.45], "<f8").tobytes(), np.uint8)
        rows[i, R.ALT_OFFSET:R.ALT_OFFSET + 8] = np.frombuffer(np.array([270.0], "<f8").tobytes(), np.uint8)
    dat = os.path.join(d, pid.lower() + ".dat")
    rows.tofile(dat)
    with open(os.path.join(d, pid.lower() + ".lbl"), "w") as f:
        f.write(f"PDS_VERSION_ID = PDS3\n  FILE_RECORDS = {label_rows or n_rows}\nEND\n")
    return dat


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "sharad_highres"
    d.mkdir()
    monkeypatch.setattr(R, "SHARAD_HR_DIR", str(d))
    monkeypatch.setattr(R, "SHARAD_CACHE_DIR", str(d / ".cache"))
    R._product_caches.clear()
    R._png_cache.clear()
    R._percentile_cache.clear()
    yield d
    R._product_caches.clear()


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(R.router)
    return TestClient(app)


# ── incomplete-file guard (C5) ────────────────────────────────────────

def test_check_dat_complete_ok(tmp_path):
    dat = _write_product(str(tmp_path))
    assert check_dat_complete(dat, dat[:-4] + ".lbl", R.ROW_BYTES) == N_ROWS


def test_check_dat_complete_aria2(tmp_path):
    dat = _write_product(str(tmp_path))
    open(dat + ".aria2", "w").close()
    with pytest.raises(IncompleteProductError, match="still downloading"):
        check_dat_complete(dat, dat[:-4] + ".lbl", R.ROW_BYTES)


def test_check_dat_complete_truncated(tmp_path):
    dat = _write_product(str(tmp_path), label_rows=N_ROWS * 4)
    with pytest.raises(IncompleteProductError, match="truncated"):
        check_dat_complete(dat, dat[:-4] + ".lbl", R.ROW_BYTES)


def test_metadata_409_while_downloading(data_dir, client):
    dat = _write_product(str(data_dir))
    open(dat + ".aria2", "w").close()
    r = client.get("/api/sharad_highres/metadata", params={"product_id": PID})
    assert r.status_code == 409
    assert "still downloading" in r.json()["detail"]


def test_metadata_404_missing(data_dir, client):
    r = client.get("/api/sharad_highres/metadata", params={"product_id": "R_1111101_001_SS19_700_A"})
    assert r.status_code == 404
    assert r.json()["detail"]


def test_cache_invalidated_when_source_changes(data_dir, client):
    dat = _write_product(str(data_dir))
    r = client.get("/api/sharad_highres/metadata", params={"product_id": PID})
    assert r.status_code == 200 and r.json()["rows"] == N_ROWS
    cache_dir = data_dir / ".cache" / PID
    stamp = json.loads((cache_dir / "source.json").read_text())
    assert stamp["size"] == N_ROWS * R.ROW_BYTES
    # re-download with more rows → stamp mismatch → caches rebuilt
    os.remove(dat)
    _write_product(str(data_dir), n_rows=N_ROWS * 2)
    st = os.stat(dat)
    os.utime(dat, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    r = client.get("/api/sharad_highres/metadata", params={"product_id": PID})
    assert r.status_code == 200 and r.json()["rows"] == N_ROWS * 2


def test_legacy_cache_older_than_dat_is_dropped(data_dir):
    dat = _write_product(str(data_dir))
    cache_dir = data_dir / ".cache" / PID
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / "power.npy", np.zeros((3, 667), np.float32))
    old = os.path.getmtime(dat) - 100
    os.utime(cache_dir / "power.npy", (old, old))
    R._resolve_product(PID)
    assert not (cache_dir / "power.npy").exists()
    assert (cache_dir / "source.json").exists()


# ── receive-window parsing (C3) ───────────────────────────────────────

def test_read_rwot(tmp_path):
    dat = _write_product(str(tmp_path))
    rw = read_rwot(dat, N_ROWS, R.ROW_BYTES)
    assert rw.shape == (N_ROWS,)
    assert rw[0] == 9000.0 and rw[-1] == 9000.0 + N_ROWS - 1


# ── latitude convention (C4) ──────────────────────────────────────────

def test_latitudes_are_planetocentric(data_dir, client):
    _write_product(str(data_dir))
    m = client.get("/api/sharad_highres/metadata", params={"product_id": PID}).json()
    assert m["start_lat"] == pytest.approx(18.0)
    rm = client.get("/api/sharad_highres/radargram_meta", params={"product_id": PID, "downsample": 1}).json()
    assert rm["lats"][10] == pytest.approx(18.1, abs=1e-4)
    tr = client.get("/api/sharad_highres/track", params={"product_id": PID, "downsample": 1}).json()
    assert tr["lats"][63] == pytest.approx(18.63, abs=1e-4)


# ── hyperbola fit end-to-end (C2) ─────────────────────────────────────

def test_hyperbola_endpoint_on_synthetic_product(data_dir, client):
    import math
    from analysis.shared.constants import SPEED_OF_LIGHT as C
    eps, depth, n, apex = 4.0, 40.0, 301, 150
    v = C / math.sqrt(eps)
    dlat = 18.1 / 59152.0            # ≈18.1 m per trace (SS19 spacing)
    dt = R.SHARAD_SAMPLE_INTERVAL_US * 1e-6

    def hyp(i):
        x = (i - apex) * 18.1
        t = math.sqrt((2 * depth / v) ** 2 + (2 * x / v) ** 2)
        b = SURF_BIN + t / dt
        return {int(round(b)): 30.0} if b < SURF_BIN + 200 else {}

    _write_product(str(data_dir), n_rows=n, lat_step=dlat, extra=hyp)
    from api.epsilon_router import router as eps_router
    client.app.include_router(eps_router)
    r = client.get("/api/epsilon/hyperbola_fit", params={
        "product_id": PID, "trace_idx0": apex, "bin_idx0": SURF_BIN + 15,
        "window_traces": 200, "window_bins": 200,
    })
    assert r.status_code == 200
    j = r.json()
    assert j["success"] and j["experimental"]
    assert j["epsilon_r_est"] == pytest.approx(eps, rel=0.1)
    assert j["depth_m"] == pytest.approx(depth, rel=0.1)
    assert "experimental" in j["flags"]
    assert j["diagnostics"]["trace_spacing_m"] == pytest.approx(18.1, rel=0.02)
    assert j["overlay"] and all("trace" in p and "bin" in p for p in j["overlay"])


# ── concurrency (U2) ──────────────────────────────────────────────────

def test_handlers_are_sync_and_parse_once_under_concurrency(data_dir, monkeypatch):
    import inspect
    import threading
    for name in ("get_metadata", "get_radargram", "get_radargram_meta", "get_surface",
                 "depth_conversion", "get_mola_profile", "get_reflectors"):
        assert not inspect.iscoroutinefunction(getattr(R, name)), name
    _write_product(str(data_dir))
    calls = []
    orig = R._parse_and_cache

    def counting(cache):
        calls.append(1)
        return orig(cache)

    monkeypatch.setattr(R, "_parse_and_cache", counting)
    errs = []

    def worker():
        try:
            R._get_power(PID)
        except Exception as e:  # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(calls) == 1


# ── depth conversion endpoint (incl. two-layer + uncertainty) ─────────

def test_depth_conversion_endpoint(data_dir, client):
    from analysis.shared.constants import SPEED_OF_LIGHT as C
    _write_product(str(data_dir))
    q = {"product_id": PID, "trace_idx": 5, "cursor_bin": SURF_BIN + 100, "downsample": 2,
         "epsilon_r1": 4.0, "epsilon_r2": 4.0, "boundary_m": 0}
    j = client.get("/api/sharad_highres/depth_conversion", params=q).json()
    assert j["surface_bin"] == SURF_BIN and j["full_trace_idx"] == 10
    dt = 100 * 37.5e-9
    assert j["depth_m"] == pytest.approx(C * dt / 4, abs=0.1)
    assert j["depth_uncertainty_m"] == pytest.approx(C * 37.5e-9 / 4, abs=0.01)
    assert j["vertical_resolution_m"] == pytest.approx(7.49, abs=0.01)
    assert j["lat"] == pytest.approx(18.1, abs=1e-4)
    # two-layer: 50 m of εr=4 then εr=9
    q.update(epsilon_r2=9.0, boundary_m=50)
    j2 = client.get("/api/sharad_highres/depth_conversion", params=q).json()
    t1 = 2 * 50 * 2 / C
    assert j2["depth_m"] == pytest.approx(50 + C * (dt - t1) / 6, abs=0.1)
    assert j2["depth_uncertainty_m"] == pytest.approx(C * 37.5e-9 / 6, abs=0.01)
    # manual surface override
    q.update(surface_bin_override=SURF_BIN + 50, boundary_m=0, epsilon_r2=4.0)
    j3 = client.get("/api/sharad_highres/depth_conversion", params=q).json()
    assert j3["delta_bins"] == 50 and j3["surface_pick"] == "manual"
