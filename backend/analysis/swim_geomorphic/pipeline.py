from typing import Callable, ClassVar, Literal, cast

from ..swim_common import SwimGeoTIFF, load_swim_geotiff, validate_swim_bounds
from ..swim_common import tile_renderer

from .models import GeomorphicPointResponse, GeomorphicRegionResponse

_render_tile = cast(Callable[[SwimGeoTIFF, int, int, int], bytes | None], tile_renderer.render_consistency_tile)


class SwimGeomorphicPipeline:
    _DEPTH_SHALLOW: ClassVar[Literal["0-1m"]] = "0-1m"
    _DEPTH_INTERMEDIATE: ClassVar[Literal["1-5m"]] = "1-5m"
    _DEPTH_DEEP: ClassVar[Literal["5m-plus"]] = "5m-plus"

    _DEPTH_LANDFORMS: ClassVar[dict[str, list[str]]] = {
        _DEPTH_SHALLOW: [
            "thermal contraction crack polygons",
            "sublimation pits",
            "smooth mantling material",
            "dissected/pitted mantle",
        ],
        _DEPTH_INTERMEDIATE: [
            "scalloped terrain",
            "viscous flow features (VFF)",
        ],
        _DEPTH_DEEP: [
            "lobate debris aprons (LDA)",
            "lineated valley fill (LVF)",
            "concentric crater fill (CCF)",
            "glacier-like forms (GLF)",
        ],
    }

    def __init__(self) -> None:
        self._geomorphology_0_1m: SwimGeoTIFF | None = None
        self._geomorphology_1_5m: SwimGeoTIFF | None = None
        self._geomorphology_5m_plus: SwimGeoTIFF | None = None

    def _ensure_loaded(self) -> None:
        if self._geomorphology_0_1m is None:
            self._geomorphology_0_1m = load_swim_geotiff(
                "geomorphology_0_1m.tif",
                name="Geomorphology 0-1m",
            )
        if self._geomorphology_1_5m is None:
            self._geomorphology_1_5m = load_swim_geotiff(
                "geomorphology_1_5m.tif",
                name="Geomorphology 1-5m",
            )
        if self._geomorphology_5m_plus is None:
            self._geomorphology_5m_plus = load_swim_geotiff(
                "geomorphology_5m_plus.tif",
                name="Geomorphology >5m",
            )

    def _get_geotiff_for_depth(self, depth: Literal["0-1m", "1-5m", "5m-plus"]) -> SwimGeoTIFF:
        self._ensure_loaded()
        if depth == self._DEPTH_SHALLOW:
            if self._geomorphology_0_1m is None:
                raise RuntimeError("geomorphology 0-1m dataset is not available")
            return self._geomorphology_0_1m
        if depth == self._DEPTH_INTERMEDIATE:
            if self._geomorphology_1_5m is None:
                raise RuntimeError("geomorphology 1-5m dataset is not available")
            return self._geomorphology_1_5m
        if self._geomorphology_5m_plus is None:
            raise RuntimeError("geomorphology 5m-plus dataset is not available")
        return self._geomorphology_5m_plus

    def _detect_landforms(
        self,
        consistency_shallow: float | None,
        consistency_intermediate: float | None,
        consistency_deep: float | None,
    ) -> list[str]:
        detected: list[str] = []
        if consistency_shallow is not None and consistency_shallow > 0:
            detected.extend(self._DEPTH_LANDFORMS[self._DEPTH_SHALLOW])
        if consistency_intermediate is not None and consistency_intermediate > 0:
            detected.extend(self._DEPTH_LANDFORMS[self._DEPTH_INTERMEDIATE])
        if consistency_deep is not None and consistency_deep > 0:
            detected.extend(self._DEPTH_LANDFORMS[self._DEPTH_DEEP])
        return detected

    def query_point(self, lat: float, lon: float) -> GeomorphicPointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        if not valid:
            return GeomorphicPointResponse(
                lat=lat,
                lon=lon,
                consistency_shallow=None,
                consistency_intermediate=None,
                consistency_deep=None,
                landforms_detected=[],
                hirise_classification=None,
            )

        consistency_shallow = self._get_geotiff_for_depth(self._DEPTH_SHALLOW).sample_point(lat, lon)
        consistency_intermediate = self._get_geotiff_for_depth(self._DEPTH_INTERMEDIATE).sample_point(lat, lon)
        consistency_deep = self._get_geotiff_for_depth(self._DEPTH_DEEP).sample_point(lat, lon)

        return GeomorphicPointResponse(
            lat=lat,
            lon=lon,
            consistency_shallow=consistency_shallow,
            consistency_intermediate=consistency_intermediate,
            consistency_deep=consistency_deep,
            landforms_detected=self._detect_landforms(
                consistency_shallow,
                consistency_intermediate,
                consistency_deep,
            ),
            hirise_classification=None,
        )

    def query_region(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        depth: Literal["0-1m", "1-5m", "5m-plus"],
    ) -> GeomorphicRegionResponse:
        geotiff = self._get_geotiff_for_depth(depth)
        sample_region = cast(Callable[[float, float, float, float], dict[str, object]], geotiff.sample_region)
        region_data = sample_region(north, south, east, west)
        stats: dict[str, float] = {}
        stats_source = region_data.get("stats")
        if isinstance(stats_source, dict):
            typed_stats_source = cast(dict[str, object], stats_source)
            for key, value in typed_stats_source.items():
                if isinstance(value, (int, float)):
                    stats[key] = float(value)

        bounds = {"north": north, "south": south, "east": east, "west": west}
        bounds_source = region_data.get("bounds")
        if isinstance(bounds_source, dict):
            typed_bounds_source = cast(dict[str, object], bounds_source)
            for key in bounds:
                value = typed_bounds_source.get(key)
                if isinstance(value, (int, float)):
                    bounds[key] = float(value)

        return GeomorphicRegionResponse(
            bounds=bounds,
            stats=stats,
            depth=depth,
            tile_url=f"/api/swim-ice/geomorphic/tile/{{z}}/{{x}}/{{y}}.png?depth={depth}",
        )

    def get_tile(self, z: int, x: int, y: int, depth: Literal["0-1m", "1-5m", "5m-plus"]) -> bytes | None:
        geotiff = self._get_geotiff_for_depth(depth)
        return _render_tile(geotiff, z, x, y)
