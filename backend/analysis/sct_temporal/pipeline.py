"""
SCT Temporal Change Detection Pipeline.

Two-stage automated pipeline:
  Stage 1 — MarsLandformNet identifies SCT regions (browse, 25m/px)
  Stage 2 — Phase correlation on full-res HiRISE pairs (RDR, 25cm/px)

First systematic measurement of scalloped terrain scarp retreat rates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .pair_finder import TemporalPair, find_temporal_pairs
from .hirise_download import download_hirise_rdr
from .coregistration import coregister_geotiffs
from .phase_correlation import sliding_window_correlation, DisplacementField
from .scarp_analysis import measure_retreat, RetreatAnalysis
from .visualize import plot_displacement_field, plot_retreat_analysis, plot_temporal_comparison

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    chip_size: int = 64
    step_size: int = 16
    upsample_factor: int = 100
    snr_threshold: float = 3.0
    max_emission_diff: float = 5.0
    min_time_gap_days: float = 300
    frost_free_only: bool = True
    gradient_threshold: float = 0.15
    search_radius_km: float = 30.0


@dataclass
class PipelineResult:
    pair: TemporalPair
    displacement: Optional[DisplacementField]
    retreat: Optional[RetreatAnalysis]
    output_dir: Path
    success: bool = True
    error: Optional[str] = None


class SCTTemporalPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    async def run(
        self,
        product_id: str,
        output_dir: Path,
        max_pairs: int = 5,
    ) -> List[PipelineResult]:
        """
        Full pipeline: find SCT temporal pairs and measure retreat.

        Parameters
        ----------
        product_id : str
            HiRISE observation ID of a known SCT region.
        output_dir : Path
            Directory for output files.
        max_pairs : int
            Maximum number of temporal pairs to process.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[PipelineResult] = []

        lat, lon = self._resolve_coordinates(product_id)
        logger.info(f"SCT Pipeline: {product_id} at ({lat:.2f}, {lon:.2f})")

        pairs = await find_temporal_pairs(
            lat=lat,
            lon=lon,
            radius_km=self.config.search_radius_km,
            max_emission_diff=self.config.max_emission_diff,
            min_time_gap_days=self.config.min_time_gap_days,
            frost_free_only=self.config.frost_free_only,
            max_pairs=max_pairs,
        )

        if not pairs:
            logger.warning("No suitable temporal pairs found")
            return results

        logger.info(f"Processing {len(pairs)} temporal pairs")

        for i, pair in enumerate(pairs):
            pair_dir = output_dir / f"pair_{i:02d}"
            pair_dir.mkdir(exist_ok=True)
            result = await self._process_pair(pair, pair_dir)
            results.append(result)

        self._save_summary(results, output_dir)
        return results

    async def run_pair(
        self,
        product_id_a: str,
        product_id_b: str,
        output_dir: Path,
        time_gap_days: Optional[float] = None,
    ) -> PipelineResult:
        """Process a specific pair of HiRISE products."""
        output_dir.mkdir(parents=True, exist_ok=True)

        from .pair_finder import HiRISEProduct, TemporalPair

        pair = TemporalPair(
            product_a=HiRISEProduct(
                product_id=product_id_a + "_RED",
                observation_id=product_id_a,
                center_lat=0, center_lon=0,
                emission_angle=0, incidence_angle=0,
                solar_longitude=0, observation_date="",
            ),
            product_b=HiRISEProduct(
                product_id=product_id_b + "_RED",
                observation_id=product_id_b,
                center_lat=0, center_lon=0,
                emission_angle=0, incidence_angle=0,
                solar_longitude=0, observation_date="",
            ),
            time_gap_days=time_gap_days or 687.0,
            time_gap_mars_years=(time_gap_days or 687.0) / 686.97,
            emission_angle_diff=0,
            incidence_angle_diff=0,
            score=1.0,
        )

        return await self._process_pair(pair, output_dir)

    async def _process_pair(
        self,
        pair: TemporalPair,
        output_dir: Path,
    ) -> PipelineResult:
        obs_a = pair.product_a.observation_id
        obs_b = pair.product_b.observation_id
        logger.info(
            f"Processing pair: {obs_a} → {obs_b} "
            f"(gap: {pair.time_gap_mars_years:.1f} Mars yr, "
            f"em_diff: {pair.emission_angle_diff:.1f}°)"
        )

        try:
            path_a = await download_hirise_rdr(obs_a)
            path_b = await download_hirise_rdr(obs_b)

            if path_a is None or path_b is None:
                return PipelineResult(
                    pair=pair, displacement=None, retreat=None,
                    output_dir=output_dir, success=False,
                    error=f"Download failed: {obs_a if path_a is None else obs_b}",
                )

            coreg = coregister_geotiffs(path_a, path_b)

            displacement = sliding_window_correlation(
                coreg.img1,
                coreg.img2,
                chip_size=self.config.chip_size,
                step_size=self.config.step_size,
                upsample_factor=self.config.upsample_factor,
                snr_threshold=self.config.snr_threshold,
                pixel_scale_m=coreg.pixel_scale_m,
                stable_mask=coreg.stable_mask,
            )

            retreat = measure_retreat(
                displacement,
                reference_image=coreg.img1,
                time_gap_mars_years=pair.time_gap_mars_years,
                gradient_threshold=self.config.gradient_threshold,
            )

            plot_displacement_field(
                displacement,
                output_dir / "displacement.png",
                title=f"{obs_a} → {obs_b}",
            )
            plot_retreat_analysis(retreat, output_dir / "retreat_analysis.png")
            plot_temporal_comparison(
                coreg.img1, coreg.img2, displacement,
                output_dir / "temporal_comparison.png",
                title=f"{obs_a} → {obs_b} ({pair.time_gap_mars_years:.1f} Mars yr)",
            )

            with open(output_dir / "result.json", "w") as f:
                json.dump(retreat.summary, f, indent=2)

            logger.info(
                f"Pair complete: mean retreat = "
                f"{retreat.mean_retreat_rate_m_per_yr:.2f} m/Mars yr"
            )

            return PipelineResult(
                pair=pair,
                displacement=displacement,
                retreat=retreat,
                output_dir=output_dir,
            )

        except Exception as e:
            logger.error(f"Pair processing failed: {e}", exc_info=True)
            return PipelineResult(
                pair=pair, displacement=None, retreat=None,
                output_dir=output_dir, success=False, error=str(e),
            )

    def _resolve_coordinates(self, product_id: str) -> tuple[float, float]:
        """Extract approximate lat/lon from HiRISE product ID encoding."""
        parts = product_id.replace("_RED", "").split("_")
        if len(parts) >= 3:
            try:
                lat_code = int(parts[2])
                lat = (lat_code - 1800) / 10.0
                return lat, 0.0  # lon not encoded in product ID
            except ValueError:
                pass
        return 45.0, 90.0  # default: Utopia Planitia SCT region

    def _save_summary(self, results: List[PipelineResult], output_dir: Path) -> None:
        result_list: list[dict[str, object]] = []
        summary: dict[str, object] = {
            "total_pairs": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": result_list,
        }

        for r in results:
            entry: dict[str, object] = {
                "product_a": r.pair.product_a.observation_id,
                "product_b": r.pair.product_b.observation_id,
                "time_gap_mars_years": r.pair.time_gap_mars_years,
                "emission_angle_diff": r.pair.emission_angle_diff,
                "success": r.success,
            }
            if r.success and r.retreat:
                entry.update(r.retreat.summary)
            if r.error:
                entry["error"] = r.error
            result_list.append(entry)

        with open(output_dir / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Pipeline summary saved to {output_dir / 'pipeline_summary.json'}")
