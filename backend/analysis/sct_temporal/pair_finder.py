"""
Find HiRISE temporal pairs suitable for scarp retreat measurement.

Queries ODE REST API for RDRV11 products covering the same area,
filters by emission angle difference (parallax minimization) and
seasonal constraints (CO2 frost avoidance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import aiohttp

logger = logging.getLogger(__name__)

ODE_REST_BASE = "https://oderest.rsl.wustl.edu/live2"

# Mars solar longitude ranges for frost-free northern mid-latitudes
# Ls 60-180 is roughly spring through mid-summer (frost-free at 45N)
FROST_FREE_LS_MIN = 60.0
FROST_FREE_LS_MAX = 200.0


@dataclass
class HiRISEProduct:
    """Metadata for a single HiRISE RDR product."""

    product_id: str
    observation_id: str
    center_lat: float
    center_lon: float  # -180 to 180
    emission_angle: float  # degrees
    incidence_angle: float
    solar_longitude: float  # Ls in degrees (0-360)
    observation_date: str  # ISO date string
    footprint: Optional[dict[str, float]] = None  # lat_min, lat_max, lon_min, lon_max


@dataclass
class TemporalPair:
    """A pair of HiRISE products suitable for temporal change measurement."""

    product_a: HiRISEProduct  # earlier observation
    product_b: HiRISEProduct  # later observation
    time_gap_days: float
    time_gap_mars_years: float
    emission_angle_diff: float
    incidence_angle_diff: float
    score: float  # composite quality score (higher = better)


async def find_temporal_pairs(
    lat: float,
    lon: float,
    radius_km: float = 30.0,
    max_emission_diff: float = 5.0,
    min_time_gap_days: float = 300,
    frost_free_only: bool = True,
    max_pairs: int = 20,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[TemporalPair]:
    """
    Find HiRISE temporal pairs for scarp retreat measurement.

    Parameters
    ----------
    lat, lon : float
        Center coordinates of the SCT region.
    radius_km : float
        Search radius in km. Default 30 km (typical HiRISE swath width ~6km).
    max_emission_diff : float
        Maximum emission angle difference between pair members (degrees).
        Smaller = less parallax error. Default 5°.
    min_time_gap_days : float
        Minimum time separation. Default 300 days (~0.5 Mars year).
    frost_free_only : bool
        If True, only include observations with Ls in frost-free range.
    max_pairs : int
        Maximum number of pairs to return.
    session : aiohttp.ClientSession, optional
        Reuse existing HTTP session.

    Returns
    -------
    List of TemporalPair, sorted by quality score (best first).
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        products = await _query_ode_hirise(lat, lon, radius_km, session)
    finally:
        if close_session:
            await session.close()

    if not products:
        logger.warning(f"No HiRISE products found at ({lat:.2f}, {lon:.2f})")
        return []

    logger.info(f"Found {len(products)} HiRISE RDR products at ({lat:.2f}, {lon:.2f})")

    # Filter by frost-free season
    if frost_free_only:
        before = len(products)
        products = [
            p for p in products
            if FROST_FREE_LS_MIN <= p.solar_longitude <= FROST_FREE_LS_MAX
        ]
        logger.info(f"Frost-free filter: {before} → {len(products)} products")

    if len(products) < 2:
        logger.warning("Fewer than 2 products after filtering")
        return []

    # Sort by date
    products.sort(key=lambda p: p.observation_date)

    # Generate pairs
    pairs: List[TemporalPair] = []
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            pa, pb = products[i], products[j]

            # Time gap
            try:
                da = datetime.fromisoformat(pa.observation_date.replace("Z", "+00:00"))
                db = datetime.fromisoformat(pb.observation_date.replace("Z", "+00:00"))
                gap_days = abs((db - da).total_seconds()) / 86400
            except (ValueError, TypeError):
                continue

            if gap_days < min_time_gap_days:
                continue

            # Emission angle difference
            em_diff = abs(pa.emission_angle - pb.emission_angle)
            if em_diff > max_emission_diff:
                continue

            # Incidence angle difference
            inc_diff = abs(pa.incidence_angle - pb.incidence_angle)

            # Quality score: prefer large time gap, small emission diff
            # Higher score = better pair
            time_score = min(gap_days / 1000, 1.0)  # saturates at ~1.5 Mars years
            emission_penalty = em_diff / max_emission_diff  # 0 = same angle, 1 = max diff
            score = time_score * (1.0 - 0.5 * emission_penalty)

            pairs.append(
                TemporalPair(
                    product_a=pa,
                    product_b=pb,
                    time_gap_days=gap_days,
                    time_gap_mars_years=gap_days / 686.97,
                    emission_angle_diff=em_diff,
                    incidence_angle_diff=inc_diff,
                    score=score,
                )
            )

    # Sort by score (best first) and limit
    pairs.sort(key=lambda p: p.score, reverse=True)
    pairs = pairs[:max_pairs]

    logger.info(f"Generated {len(pairs)} temporal pairs")
    return pairs


async def _query_ode_hirise(
    lat: float,
    lon: float,
    radius_km: float,
    session: aiohttp.ClientSession,
) -> List[HiRISEProduct]:
    """Query ODE REST for HiRISE RDRV11 products near a location."""
    ode_lon = lon % 360  # ODE expects 0-360

    url = (
        f"{ODE_REST_BASE}?"
        f"target=mars&query=product&results=p&output=JSON"
        f"&ihid=MRO&iid=HIRISE&pt=RDRV11"
        f"&lat={lat}&lon={ode_lon}&loc=o&r={radius_km}"
        f"&limit=100"
    )

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.error(f"ODE returned {resp.status}")
                return []
            data = await resp.json()
    except Exception as e:
        logger.error(f"ODE query failed: {e}")
        return []

    items = data.get("ODEResults", {}).get("Products", {})
    if items is None:
        return []
    item_list = items.get("Product", [])
    if isinstance(item_list, dict):
        item_list = [item_list]

    products: List[HiRISEProduct] = []
    for item in item_list:
        pid = item.get("pdsid", "")
        # Only RED channel RDR products
        if "_RED" not in pid.upper():
            continue

        obs_id = pid.rsplit("_RED", 1)[0] if "_RED" in pid.upper() else pid

        try:
            center_lat = float(item.get("Center_latitude", 0))
            center_lon = float(item.get("Center_longitude", 0))
            # Convert 0-360 to -180/180
            if center_lon > 180:
                center_lon -= 360

            emission = float(item.get("Emission_angle", 0))
            incidence = float(item.get("Incidence_angle", 0))
            solar_lon = float(item.get("Solar_longitude", 0))
            obs_date = item.get("UTC_start_time", "")

            footprint = None
            try:
                footprint = {
                    "lat_min": float(item.get("Minimum_latitude", 0)),
                    "lat_max": float(item.get("Maximum_latitude", 0)),
                    "lon_min": float(item.get("Westernmost_longitude", 0)),
                    "lon_max": float(item.get("Easternmost_longitude", 0)),
                }
            except (ValueError, TypeError):
                pass

            products.append(
                HiRISEProduct(
                    product_id=pid,
                    observation_id=obs_id,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    emission_angle=emission,
                    incidence_angle=incidence,
                    solar_longitude=solar_lon,
                    observation_date=obs_date,
                    footprint=footprint,
                )
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Skipping product {pid}: {e}")

    return products
