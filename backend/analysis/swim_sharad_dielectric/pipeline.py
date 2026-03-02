from typing import Callable, ClassVar, Literal, cast

from ..swim_common import SwimGeoTIFF, load_swim_geotiff, validate_swim_bounds
from ..swim_common import tile_renderer

from .models import DielectricPointResponse, DielectricRegionResponse

_render_tile = cast(Callable[[SwimGeoTIFF, int, int, int], bytes | None], tile_renderer.render_consistency_tile)


class SwimSharadDielectricPipeline:
    _DEPTH_1_5M: ClassVar[Literal["1-5m"]] = "1-5m"
    _DEPTH_5M_PLUS: ClassVar[Literal["5m-plus"]] = "5m-plus"
    _DEPTH_RANGES: ClassVar[list[str]] = ["1-5m", ">5m"]

    def __init__(self) -> None:
        self._dielectric_1_5m: SwimGeoTIFF | None = None
        self._dielectric_5m_plus: SwimGeoTIFF | None = None

    def _ensure_loaded(self) -> None:
        if self._dielectric_1_5m is None:
            self._dielectric_1_5m = load_swim_geotiff(
                "radar_dielectric_1_5m.tif",
                name="SHARAD dielectric 1-5m",
            )
        if self._dielectric_5m_plus is None:
            self._dielectric_5m_plus = load_swim_geotiff(
                "radar_dielectric_5m_plus.tif",
                name="SHARAD dielectric >5m",
            )

    @classmethod
    def _epsilon_to_consistency(cls, epsilon: float | None) -> float | None:
        if epsilon is None:
            return None
        if epsilon < 4.5:
            return 1.0
        if epsilon <= 6.0:
            return 0.0
        return -1.0

    def _get_geotiff_for_depth(self, depth: Literal["1-5m", "5m-plus"]) -> SwimGeoTIFF:
        self._ensure_loaded()
        if depth == self._DEPTH_1_5M:
            if self._dielectric_1_5m is None:
                raise RuntimeError("radar dielectric 1-5m dataset is not available")
            return self._dielectric_1_5m
        if self._dielectric_5m_plus is None:
            raise RuntimeError("radar dielectric 5m-plus dataset is not available")
        return self._dielectric_5m_plus

    def query_point(self, lat: float, lon: float) -> DielectricPointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        if not valid:
            return DielectricPointResponse(
                lat=lat,
                lon=lon,
                consistency_score_1_5m=None,
                consistency_score_5m_plus=None,
                estimated_epsilon=None,
                depth_ranges=self._DEPTH_RANGES,
                nearest_track_id=None,
            )

        epsilon_1_5m = self._get_geotiff_for_depth(self._DEPTH_1_5M).sample_point(lat, lon)
        epsilon_5m_plus = self._get_geotiff_for_depth(self._DEPTH_5M_PLUS).sample_point(lat, lon)

        scores = [v for v in (epsilon_1_5m, epsilon_5m_plus) if v is not None]
        estimated_epsilon = round(sum(scores) / len(scores), 4) if scores else None

        return DielectricPointResponse(
            lat=lat,
            lon=lon,
            consistency_score_1_5m=self._epsilon_to_consistency(epsilon_1_5m),
            consistency_score_5m_plus=self._epsilon_to_consistency(epsilon_5m_plus),
            estimated_epsilon=estimated_epsilon,
            depth_ranges=self._DEPTH_RANGES,
            nearest_track_id=None,
        )

    def query_region(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        depth: Literal["1-5m", "5m-plus"],
    ) -> DielectricRegionResponse:
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

        return DielectricRegionResponse(
            bounds=bounds,
            stats=stats,
            depth=depth,
            tile_url=f"/api/swim-ice/radar-dielectric/tile/{{z}}/{{x}}/{{y}}.png?depth={depth}",
        )

    def get_tile(self, z: int, x: int, y: int, depth: Literal["1-5m", "5m-plus"]) -> bytes | None:
        geotiff = self._get_geotiff_for_depth(depth)
        return _render_tile(geotiff, z, x, y)
