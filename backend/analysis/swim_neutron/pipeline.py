from typing import cast

from analysis.swim_common import (
    SwimGeoTIFF,
    load_swim_geotiff,
    render_consistency_tile,
    validate_swim_bounds,
)

from .models import PointResponse, RegionResponse


class SwimNeutronPipeline:
    def __init__(self) -> None:
        self._geotiff: SwimGeoTIFF | None = None

    def _ensure_loaded(self) -> None:
        if self._geotiff is None:
            self._geotiff = load_swim_geotiff("neutron_consistency.tif", name="SWIM Neutron")

    @staticmethod
    def _weh_to_consistency(weh: float) -> float:
        if weh > 10.0:
            return 1.0
        if weh >= 3.0:
            return 0.0
        return -1.0

    @staticmethod
    def _consistency_to_weh(consistency: float) -> float:
        if consistency >= 0.3:
            return 12.0
        if consistency <= -0.3:
            return 2.0
        return 6.0

    def query_point(self, lat: float, lon: float) -> PointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        if not valid:
            return PointResponse(
                lat=lat,
                lon=lon,
                consistency_score=None,
                water_equivalent_h=None,
                depth_range="0-1m",
                data_quality="no_data",
                native_resolution_note="Native neutron footprint is ~300 km despite ~3 km output grid.",
            )

        self._ensure_loaded()
        if not self._geotiff or not self._geotiff.loaded:
            return PointResponse(
                lat=lat,
                lon=lon,
                consistency_score=None,
                water_equivalent_h=None,
                depth_range="0-1m",
                data_quality="no_data",
                native_resolution_note="Native neutron footprint is ~300 km despite ~3 km output grid.",
            )

        sampled = self._geotiff.sample_point(lat, lon)
        if sampled is None:
            return PointResponse(
                lat=lat,
                lon=lon,
                consistency_score=None,
                water_equivalent_h=None,
                depth_range="0-1m",
                data_quality="no_data",
                native_resolution_note="Native neutron footprint is ~300 km despite ~3 km output grid.",
            )

        if sampled > 1.0:
            weh = round(sampled, 3)
            consistency = self._weh_to_consistency(weh)
        else:
            consistency = max(-1.0, min(1.0, sampled))
            weh = self._consistency_to_weh(consistency)

        return PointResponse(
            lat=lat,
            lon=lon,
            consistency_score=round(consistency, 3),
            water_equivalent_h=round(weh, 3),
            depth_range="0-1m",
            data_quality="nominal",
            native_resolution_note="Native neutron footprint is ~300 km despite ~3 km output grid.",
        )

    def query_region(self, north: float, south: float, east: float, west: float) -> RegionResponse:
        self._ensure_loaded()
        if not self._geotiff or not self._geotiff.loaded:
            return RegionResponse(
                bounds={"north": north, "south": south, "east": east, "west": west},
                stats={"available": False, "error": "SWIM neutron data unavailable"},
                tile_url="/api/swim-ice/neutron/tile/{z}/{x}/{y}.png",
            )

        region = cast(
            dict[str, object],
            self._geotiff.sample_region(north=north, south=south, east=east, west=west),
        )

        raw_bounds = region.get("bounds")
        if isinstance(raw_bounds, dict):
            bounds = {
                "north": float(raw_bounds.get("north", north)),
                "south": float(raw_bounds.get("south", south)),
                "east": float(raw_bounds.get("east", east)),
                "west": float(raw_bounds.get("west", west)),
            }
        else:
            bounds = {"north": north, "south": south, "east": east, "west": west}

        raw_stats = region.get("stats")
        if isinstance(raw_stats, dict):
            stats = {
                key: value
                for key, value in raw_stats.items()
                if isinstance(value, (float, int, str, bool)) or value is None
            }
        else:
            stats = {
                key: value
                for key, value in region.items()
                if isinstance(value, (float, int, str, bool)) or value is None
            }

        return RegionResponse(
            bounds=bounds,
            stats={k: float(v) if isinstance(v, int) else v for k, v in stats.items()},
            tile_url="/api/swim-ice/neutron/tile/{z}/{x}/{y}.png",
        )

    def get_tile(self, z: int, x: int, y: int) -> bytes | None:
        self._ensure_loaded()
        if not self._geotiff:
            return None
        return render_consistency_tile(self._geotiff, z=z, x=x, y=y)
