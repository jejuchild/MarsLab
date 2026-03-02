from typing import Literal, cast

from analysis.swim_common import (
    SwimGeoTIFF,
    load_swim_geotiff,
    render_consistency_tile,
    validate_swim_bounds,
)

from .models import PointResponse, RegionResponse


class SwimThermalPipeline:
    def __init__(self) -> None:
        self._geotiff: SwimGeoTIFF | None = None

    def _ensure_loaded(self) -> None:
        if self._geotiff is None:
            self._geotiff = load_swim_geotiff("thermal_consistency.tif", name="SWIM Thermal")

    @staticmethod
    def _interpret(
        consistency_score: float | None,
        ti_value: float | None,
    ) -> Literal["ice_cemented", "ambiguous", "dry_fines"]:
        if consistency_score is not None:
            if consistency_score >= 0.3:
                return "ice_cemented"
            if consistency_score <= -0.3:
                return "dry_fines"
            return "ambiguous"

        if ti_value is None:
            return "ambiguous"
        if ti_value > 600.0:
            return "ice_cemented"
        if ti_value < 200.0:
            return "dry_fines"
        return "ambiguous"

    def query_point(self, lat: float, lon: float) -> PointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        thermal_module = __import__("api.thermal_inertia", fromlist=["get_thermal_inertia"])
        get_thermal_inertia = thermal_module.get_thermal_inertia
        ti_value = get_thermal_inertia(lat, lon)

        if not valid:
            return PointResponse(
                lat=lat,
                lon=lon,
                consistency_score=None,
                thermal_inertia_tiu=round(ti_value, 2) if ti_value is not None else None,
                interpretation=self._interpret(None, ti_value),
                depth_range="0-1m",
                data_quality="no_data",
            )

        self._ensure_loaded()
        if not self._geotiff or not self._geotiff.loaded:
            return PointResponse(
                lat=lat,
                lon=lon,
                consistency_score=None,
                thermal_inertia_tiu=round(ti_value, 2) if ti_value is not None else None,
                interpretation=self._interpret(None, ti_value),
                depth_range="0-1m",
                data_quality="no_data",
            )

        sampled = self._geotiff.sample_point(lat, lon)
        consistency_score = round(max(-1.0, min(1.0, sampled)), 3) if sampled is not None else None

        return PointResponse(
            lat=lat,
            lon=lon,
            consistency_score=consistency_score,
            thermal_inertia_tiu=round(ti_value, 2) if ti_value is not None else None,
            interpretation=self._interpret(consistency_score, ti_value),
            depth_range="0-1m",
            data_quality="nominal" if sampled is not None else "no_data",
        )

    def query_region(self, north: float, south: float, east: float, west: float) -> RegionResponse:
        self._ensure_loaded()
        if not self._geotiff or not self._geotiff.loaded:
            return RegionResponse(
                bounds={"north": north, "south": south, "east": east, "west": west},
                stats={"available": False, "error": "SWIM thermal data unavailable"},
                tile_url="/api/swim-ice/thermal/tile/{z}/{x}/{y}.png",
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
            tile_url="/api/swim-ice/thermal/tile/{z}/{x}/{y}.png",
        )

    def get_tile(self, z: int, x: int, y: int) -> bytes | None:
        self._ensure_loaded()
        if not self._geotiff:
            return None
        return render_consistency_tile(self._geotiff, z=z, x=x, y=y)
