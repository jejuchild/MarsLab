from typing import Optional

from analysis.swim_common import load_swim_geotiff, render_consistency_tile, validate_swim_bounds
from api.thermal_inertia import get_thermal_inertia

from .models import ThermalPointResponse, ThermalRegionResponse


class SwimThermalPipeline:
    def __init__(self) -> None:
        self._geotiff = None

    def _ensure_loaded(self) -> None:
        if self._geotiff is None:
            self._geotiff = load_swim_geotiff("thermal_consistency.tif", name="SWIM Thermal")

    @staticmethod
    def _interpret(consistency_score: Optional[float], ti_value: Optional[float]) -> str:
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

    def query_point(self, lat: float, lon: float) -> ThermalPointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        ti_value = get_thermal_inertia(lat, lon)

        if not valid:
            return ThermalPointResponse(
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
            return ThermalPointResponse(
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

        return ThermalPointResponse(
            lat=lat,
            lon=lon,
            consistency_score=consistency_score,
            thermal_inertia_tiu=round(ti_value, 2) if ti_value is not None else None,
            interpretation=self._interpret(consistency_score, ti_value),
            depth_range="0-1m",
            data_quality="nominal" if sampled is not None else "no_data",
        )

    def query_region(self, north: float, south: float, east: float, west: float) -> ThermalRegionResponse:
        self._ensure_loaded()
        if not self._geotiff or not self._geotiff.loaded:
            return ThermalRegionResponse(
                bounds={"north": north, "south": south, "east": east, "west": west},
                stats={"available": False, "error": "SWIM thermal data unavailable"},
                tile_url="/api/swim-ice/thermal/tile/{z}/{x}/{y}.png",
            )

        region = self._geotiff.sample_region(north=north, south=south, east=east, west=west)
        bounds = region.get("bounds", {"north": north, "south": south, "east": east, "west": west})
        stats = region.get("stats", region)

        return ThermalRegionResponse(
            bounds=bounds,
            stats=stats,
            tile_url="/api/swim-ice/thermal/tile/{z}/{x}/{y}.png",
        )

    def get_tile(self, z: int, x: int, y: int) -> Optional[bytes]:
        self._ensure_loaded()
        if not self._geotiff:
            return None
        return render_consistency_tile(self._geotiff, z=z, x=x, y=y)
