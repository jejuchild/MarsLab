"""Fast unit tests for api.image_pipeline (no network, no repo data files)."""
import os

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from api import image_pipeline as ip


def _decode(png_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)


def _write_tif(path, data, nodata=None):
    h, w = data.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype=data.dtype, transform=from_origin(0, h, 1, 1), nodata=nodata,
    ) as dst:
        dst.write(data, 1)


# ------------------------------------------------------------------
# Polar stereographic reprojection
# ------------------------------------------------------------------
class TestPolarStereo:
    def test_crs_hemisphere(self):
        assert "+lat_0=90 " in ip.polar_stereo_crs(87.0)
        assert "+lat_0=-90 " in ip.polar_stereo_crs(-87.0)
        assert f"+a={ip.MARS_A}" in ip.polar_stereo_crs(87.0)

    def test_pole_maps_to_origin_and_round_trip(self):
        from pyproj import Transformer

        fwd = Transformer.from_crs(ip.MARS_GEO_CRS, ip.polar_stereo_crs(88.0), always_xy=True)
        inv = Transformer.from_crs(ip.polar_stereo_crs(88.0), ip.MARS_GEO_CRS, always_xy=True)

        x, y = fwd.transform(0.0, 90.0)
        assert abs(x) < 1e-6 and abs(y) < 1e-6

        # lon_0=0 north polar: lon=0 lies on the -y axis, lon=90 on +x
        x0, y0 = fwd.transform(0.0, 85.0)
        assert abs(x0) < 1e-6 and y0 < 0
        x90, y90 = fwd.transform(90.0, 85.0)
        assert x90 > 0 and abs(y90) < 1e-6
        assert abs(x90 - abs(y0)) < 1e-6  # same radius at same latitude

        lons = np.array([0.0, 45.0, 120.0, -170.0])
        lats = np.array([85.0, 86.5, 89.0, 87.2])
        xs, ys = fwd.transform(lons, lats)
        lon2, lat2 = inv.transform(xs, ys)
        np.testing.assert_allclose(lat2, lats, atol=1e-9)
        np.testing.assert_allclose(lon2, lons, atol=1e-9)

    def test_reproject_polar_browse_shape_and_content(self):
        gray = np.full((100, 80), 200, dtype=np.uint8)
        feature = {
            "geometry": {"coordinates": [[[10, 86], [20, 86], [20, 87], [10, 87], [10, 86]]]},
            "properties": {"proj_center_lat": 86.5},
        }
        out = ip.reproject_polar_browse(gray, feature)
        assert out.shape == (512, 512)  # max(src, 512)
        assert out.dtype == np.uint8
        # footprint centre maps inside the source image → filled
        assert out[256, 256] == 200

    def test_reproject_keeps_larger_source_size(self):
        gray = np.full((600, 700), 50, dtype=np.uint8)
        feature = {
            "geometry": {"coordinates": [[[-100, -88], [-80, -88], [-80, -86], [-100, -86]]]},
            "properties": {"proj_center_lat": -87.0},
        }
        assert ip.reproject_polar_browse(gray, feature).shape == (600, 700)


# ------------------------------------------------------------------
# GeoTIFF → PNG
# ------------------------------------------------------------------
class TestGeotiffToPng:
    def test_ortho_size_and_zero_is_transparent(self, tmp_path):
        data = np.arange(1, 20 * 30 + 1, dtype=np.uint16).reshape(20, 30)
        data[0:5, 0:5] = 0
        path = str(tmp_path / "ortho.tif")
        _write_tif(path, data)

        img = _decode(ip.geotiff_to_png(path, max_size=2048))
        assert img.shape == (20, 30, 4)
        assert (img[0:5, 0:5, 3] == 0).all()
        assert (img[10:, 10:, 3] == 255).all()

    def test_downsamples_to_max_size(self, tmp_path):
        data = np.full((100, 200), 7, dtype=np.uint16)
        path = str(tmp_path / "big.tif")
        _write_tif(path, data)
        img = _decode(ip.geotiff_to_png(path, max_size=50))
        assert img.shape[:2] == (25, 50)

    def test_hillshade_nodata_is_transparent(self, tmp_path):
        yy, xx = np.mgrid[0:40, 0:40]
        dem = (xx * 2.0 + yy).astype(np.float32) + 100
        dem[:8, :8] = -9999.0
        path = str(tmp_path / "dem.tif")
        _write_tif(path, dem, nodata=-9999.0)

        img = _decode(ip.geotiff_to_png(path, max_size=2048, use_hillshade=True))
        assert img.shape == (40, 40, 4)
        assert (img[:8, :8, 3] == 0).all()
        assert (img[20:, 20:, 3] == 255).all()

    def test_hillshade_default_nodata_sentinel(self):
        dem = np.full((10, 10), 50.0, dtype=np.float32)
        dem[0, 0] = -32768
        rgba = ip.raster_to_rgba(dem, None, use_hillshade=True)
        assert rgba[0, 0, 3] == 0 and rgba[5, 5, 3] == 255


# ------------------------------------------------------------------
# Overlay cache eviction
# ------------------------------------------------------------------
class TestCleanupOverlayCache:
    NOW = 1_700_000_000.0

    def _mk(self, d, name, size, age_s):
        p = d / name
        p.write_bytes(b"x" * size)
        t = self.NOW - age_s
        os.utime(p, (t, t))
        return p

    def test_ttl_eviction(self, tmp_path):
        old = self._mk(tmp_path, "old.png", 10, 31 * 86400)
        new = self._mk(tmp_path, "new.png", 10, 1 * 86400)
        ip.cleanup_overlay_cache(str(tmp_path), max_mb=500, ttl_days=30, now=self.NOW)
        assert not old.exists() and new.exists()

    def test_lru_evicts_oldest_to_80_percent(self, tmp_path):
        max_mb = 1000 / (1024 * 1024)  # 1000-byte budget → evict to <= 800
        files = [self._mk(tmp_path, f"f{i}.png", 300, (10 - i) * 60) for i in range(5)]  # f0 oldest
        ip.cleanup_overlay_cache(str(tmp_path), max_mb=max_mb, ttl_days=30, now=self.NOW)
        remaining = [f.exists() for f in files]
        # 1500 → 1200 → 900 → 600 (<= 800): three oldest removed
        assert remaining == [False, False, False, True, True]

    def test_under_limit_and_subdirs_untouched(self, tmp_path):
        sub = tmp_path / "hirise_dtm"
        sub.mkdir()
        f = self._mk(tmp_path, "a.png", 100, 60)
        ip.cleanup_overlay_cache(str(tmp_path), max_mb=1, ttl_days=30, now=self.NOW)
        assert f.exists() and sub.is_dir()


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def test_gray_to_rgba_threshold():
    gray = np.array([[0, 3, 4, 255]], dtype=np.uint8)
    rgba = ip.gray_to_rgba(gray, 3)
    assert rgba[0, :, 3].tolist() == [0, 0, 255, 255]
    assert (rgba[0, :, 0] == gray[0]).all()


def test_encode_rgba_png_round_trip():
    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # red
    rgba[..., 3] = 255
    bgra = _decode(ip.encode_rgba_png(rgba, [cv2.IMWRITE_PNG_COMPRESSION, 4]))
    assert bgra.shape == (4, 5, 4)
    assert (bgra[..., 2] == 255).all() and (bgra[..., 0] == 0).all()


@pytest.mark.parametrize("shape,lines,samples,expected", [
    ((100, 100), 200, 100, (100, 50)),   # too wide → crop width
    ((100, 100), 100, 200, (50, 100)),   # too tall → crop height
    ((100, 100), 1000, 1005, (100, 100)),  # within 1% → untouched
])
def test_crop_to_aspect(shape, lines, samples, expected):
    assert ip.crop_to_aspect(np.zeros(shape, np.uint8), lines, samples).shape == expected


def test_downscale_to_width():
    assert ip.downscale_to_width(np.zeros((300, 2048), np.uint8), 1024).shape == (150, 1024)
    assert ip.downscale_to_width(np.zeros((300, 500), np.uint8), 1024).shape == (300, 500)


def test_crism_image_to_rgba_variants():
    gray = np.array([[0, 10]], dtype=np.uint8)
    assert ip.crism_image_to_rgba(gray)[0, :, 3].tolist() == [0, 255]
    bgr = np.zeros((1, 2, 3), np.uint8)
    bgr[0, 1] = (50, 60, 70)
    rgba = ip.crism_image_to_rgba(bgr)
    assert rgba[0, :, 3].tolist() == [0, 255] and rgba[0, 1, :3].tolist() == [50, 60, 70]
    four = np.ones((2, 2, 4), np.uint8)
    assert ip.crism_image_to_rgba(four) is four


def test_blank_png_is_tile_sized():
    assert _decode(ip.blank_png()).shape == (ip.TILE_SIZE, ip.TILE_SIZE)
