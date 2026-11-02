"""At-point matching: products must actually cover the inspected point."""

from api.inspector_router import _search_cache_at_point


def _poly(pid, west, south, east, north):
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Feature", "properties": {"product_id": pid},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def _line(pid, coords):
    return {"type": "Feature", "properties": {"product_id": pid},
            "geometry": {"type": "LineString", "coordinates": coords}}


def test_polygon_containing_point_covers_with_zero_distance():
    res = _search_cache_at_point("hirise", [_poly("A", 77.4, 18.4, 77.5, 18.5)], 18.45, 77.45, 10)
    assert [(p.product_id, p.covers, p.distance_km) for p in res] == [("A", True, 0.0)]


def test_long_sharad_track_through_point_is_found():
    # Track runs 600 km north-south through the point; its centre is ~300 km away.
    track = _line("S1", [[77.45, 13.4], [77.45, 23.4]])
    res = _search_cache_at_point("sharad", [track], 18.45, 77.45, 10)
    assert len(res) == 1 and res[0].covers and res[0].distance_km < 0.5


def test_large_ctx_frame_with_far_centre_is_found():
    # 0.6 x 2.0 deg frame (~35 x 120 km); point near its north end.
    res = _search_cache_at_point("ctx", [_poly("C", 77.2, 17.0, 77.8, 19.0)], 18.9, 77.5, 10)
    assert len(res) == 1 and res[0].covers


def test_nearby_but_not_covering_is_listed_after_covering():
    near = _poly("NEAR", 77.55, 18.4, 77.6, 18.5)   # ~3 km east of the point
    cover = _poly("COVER", 77.4, 18.4, 77.5, 18.5)
    res = _search_cache_at_point("hirise", [near, cover], 18.45, 77.45, 10)
    assert [p.product_id for p in res] == ["COVER", "NEAR"]
    assert res[1].covers is False and 0 < res[1].distance_km < 10


def test_far_product_excluded():
    res = _search_cache_at_point("hirise", [_poly("FAR", 79.0, 18.4, 79.1, 18.5)], 18.45, 77.45, 10)
    assert res == []


def test_0_360_longitudes_are_handled():
    res = _search_cache_at_point("crism", [_poly("E", 257.4, 18.4, 257.5, 18.5)], 18.45, -102.55, 10)
    assert len(res) == 1 and res[0].covers
