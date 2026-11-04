"""SHARAD x HiRISE crossing helpers and the imageless-SHARAD filter."""

import numpy as np

from api import sharad_quickview_index as qv
from api.crossings_router import _lon180, _runs
from api.sharad_quickview_index import keep_viewable_sharad


def test_runs_finds_inclusive_true_ranges():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert _runs(m) == [(1, 2), (5, 5), (7, 9)]
    assert _runs(np.zeros(4, dtype=bool)) == []


def test_lon180_wraps_0_360():
    assert np.allclose(_lon180(np.array([0, 179.5, 180.5, 359, -181])), [0, 179.5, -179.5, -1, 179])


def test_imageless_standard_sharad_tracks_are_dropped(tmp_path, monkeypatch):
    (tmp_path / "s_00000101_thm.jpg").write_bytes(b"x")
    monkeypatch.setattr(qv, "QUICKVIEW_DIR", str(tmp_path))
    qv._state.update(mtime=None, ids=frozenset())
    feats = [{"properties": {"product_id": "S_00000101_THM"}}, {"properties": {"product_id": "S_00000202_THM"}}]
    assert [f["properties"]["product_id"] for f in keep_viewable_sharad(feats)] == ["S_00000101_THM"]
