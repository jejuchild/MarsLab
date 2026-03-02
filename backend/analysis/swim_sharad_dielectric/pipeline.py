from typing import Optional

from analysis.swim_common import load_swim_geotiff, render_consistency_tile, validate_swim_bounds

from .models import DielectricPointResponse, DielectricRegionResponse


class SwimSharadDielectricPipeline:
    _DEPTH_1_5M = "1-5m"
    _DEPTH_5M_PLUS = "5m-plus"

    def __init__(self) -> None:
        self._dielectric_1_5m = None
        self._dielectric_5m_plus = None

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
    def _epsilon_to_consistency(cls, epsilon: Optional[float]) -> Optional[float]:
        if epsilon is None:
            return None
        if epsilon < 4.5:
            return 1.0
        if epsilon <= 6.0:
            return 0.0
        return -1.0

    def _get_geotiff_for_depth(self, depth: str):
        self._ensure_loaded()
        if depth == self._DEPTH_1_5M:
            return self._dielectric_1_5m
        if depth == self._DEPTH_5M_PLUS:
            return self._dielectric_5m_plus
        raise ValueError("depth must be '1-5m' or '5m-plus'")

    def query_point(self, lat: float, lon: float) -> DielectricPointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        if not valid:
            return DielectricPointResponse(
                lat=lat,
                lon=lon,
                consistency_score_1_5m=None,
                consistency_score_5m_plus=None,
                estimated_epsilon=None,
                depth_ranges=["1-5m", ">5m"],
                nearest_track_id=None,
            )

        self._ensure_loaded()

        epsilon_1_5m = self._dielectric_1_5m.sample_point(lat, lon)
        epsilon_5m_plus = self._dielectric_5m_plus.sample_point(lat, lon)

        scores = [v for v in (epsilon_1_5m, epsilon_5m_plus) if v is not None]
        estimated_epsilon = round(sum(scores) / len(scores), 4) if scores else None

        return DielectricPointResponse(
            lat=lat,
            lon=lon,
            consistency_score_1_5m=self._epsilon_to_consistency(epsilon_1_5m),
            consistency_score_5m_plus=self._epsilon_to_consistency(epsilon_5m_plus),
            estimated_epsilon=estimated_epsilon,
            depth_ranges=["1-5m", ">5m"],
            nearest_track_id=None,
        )

    def query_region(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        depth: str,
    ) -> DielectricRegionResponse:
        geotiff = self._get_geotiff_for_depth(depth)
        region_data = geotiff.sample_region(north=north, south=south, east=east, west=west)
        stats = region_data.get("stats") if region_data.get("available") else {}
        bounds = region_data.get(
            "bounds",
            {"north": north, "south": south, "east": east, "west": west},
        )
        return DielectricRegionResponse(
            bounds=bounds,
            stats=stats,
            depth=depth,
            tile_url=f"/api/swim-ice/radar-dielectric/tile/{{z}}/{{x}}/{{y}}.png?depth={depth}",
        )

    def get_tile(self, z: int, x: int, y: int, depth: str) -> Optional[bytes]:
        geotiff = self._get_geotiff_for_depth(depth)
        return render_consistency_tile(geotiff, z=z, x=x, y=y)
