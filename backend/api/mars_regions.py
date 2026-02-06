"""
Mars named regions lookup table.

Provides coordinates for well-known Mars geographic features.
Coordinates are center point (lat, lon) with approximate radius in degrees.
"""

from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class MarsRegion:
    """Mars geographic region."""
    name: str
    lat: float  # Center latitude
    lon: float  # Center longitude (0-360 east)
    radius_deg: float  # Approximate radius in degrees
    description: str = ""


# Major Mars regions and features
# Coordinates use 0-360 east longitude system
MARS_REGIONS: Dict[str, MarsRegion] = {
    # Major Planitiae (Plains)
    "arcadia planitia": MarsRegion("Arcadia Planitia", 47.0, 184.0, 15.0, "Northern lowland plain"),
    "amazonis planitia": MarsRegion("Amazonis Planitia", 25.0, 196.0, 12.0, "Smooth lowland plain"),
    "utopia planitia": MarsRegion("Utopia Planitia", 46.0, 110.0, 20.0, "Large impact basin, Zhurong landing site"),
    "elysium planitia": MarsRegion("Elysium Planitia", 3.0, 154.0, 15.0, "InSight landing site"),
    "isidis planitia": MarsRegion("Isidis Planitia", 13.0, 87.0, 10.0, "Impact basin"),
    "hellas planitia": MarsRegion("Hellas Planitia", -42.0, 70.0, 12.0, "Deepest impact basin"),
    "argyre planitia": MarsRegion("Argyre Planitia", -50.0, 316.0, 8.0, "Second largest impact basin"),
    "chryse planitia": MarsRegion("Chryse Planitia", 22.0, 320.0, 10.0, "Viking 1 landing site"),
    "acidalia planitia": MarsRegion("Acidalia Planitia", 47.0, 338.0, 15.0, "Northern lowland"),

    # Major Terrae (Highlands)
    "terra sabaea": MarsRegion("Terra Sabaea", -5.0, 40.0, 15.0, "Southern highlands"),
    "terra cimmeria": MarsRegion("Terra Cimmeria", -35.0, 145.0, 15.0, "Ancient highlands"),
    "terra sirenum": MarsRegion("Terra Sirenum", -40.0, 210.0, 15.0, "Southern highlands"),
    "noachis terra": MarsRegion("Noachis Terra", -45.0, 350.0, 15.0, "Ancient cratered terrain"),
    "arabia terra": MarsRegion("Arabia Terra", 20.0, 5.0, 20.0, "Transitional terrain"),
    "xanthe terra": MarsRegion("Xanthe Terra", 2.0, 310.0, 10.0, "Chaotic terrain region"),

    # Major Volcanic Regions
    "tharsis": MarsRegion("Tharsis", 0.0, 260.0, 20.0, "Volcanic plateau with major volcanoes"),
    "olympus mons": MarsRegion("Olympus Mons", 18.65, 226.2, 3.0, "Largest volcano in solar system"),
    "alba mons": MarsRegion("Alba Mons", 40.5, 250.5, 4.0, "Large low-relief volcano"),
    "elysium mons": MarsRegion("Elysium Mons", 25.0, 147.0, 3.0, "Shield volcano"),
    "ascraeus mons": MarsRegion("Ascraeus Mons", 11.9, 255.9, 2.0, "Tharsis Montes volcano"),
    "pavonis mons": MarsRegion("Pavonis Mons", 1.0, 247.0, 2.0, "Tharsis Montes volcano"),
    "arsia mons": MarsRegion("Arsia Mons", -8.4, 239.0, 2.5, "Tharsis Montes volcano"),
    "syrtis major": MarsRegion("Syrtis Major", 8.0, 70.0, 8.0, "Low shield volcano, dark albedo feature"),

    # Major Canyons and Chasmata
    "valles marineris": MarsRegion("Valles Marineris", -14.0, 294.0, 20.0, "Vast canyon system"),
    "noctis labyrinthus": MarsRegion("Noctis Labyrinthus", -7.0, 262.0, 5.0, "Labyrinth of canyons"),
    "ius chasma": MarsRegion("Ius Chasma", -7.0, 275.0, 4.0, "Part of Valles Marineris"),
    "coprates chasma": MarsRegion("Coprates Chasma", -13.0, 295.0, 5.0, "Part of Valles Marineris"),
    "hebes chasma": MarsRegion("Hebes Chasma", -1.0, 284.0, 2.0, "Enclosed canyon"),
    "juventae chasma": MarsRegion("Juventae Chasma", -4.0, 298.0, 2.0, "Outflow channel source"),

    # Polar Regions
    "north pole": MarsRegion("North Polar Region", 85.0, 0.0, 10.0, "Northern polar ice cap"),
    "south pole": MarsRegion("South Polar Region", -85.0, 0.0, 10.0, "Southern polar ice cap"),
    "planum boreum": MarsRegion("Planum Boreum", 85.0, 180.0, 8.0, "North polar plateau"),
    "planum australe": MarsRegion("Planum Australe", -85.0, 180.0, 8.0, "South polar plateau"),
    "chasma boreale": MarsRegion("Chasma Boreale", 83.0, 315.0, 4.0, "North polar canyon"),

    # Landing Sites
    "jezero crater": MarsRegion("Jezero Crater", 18.38, 77.58, 0.5, "Perseverance landing site, ancient delta"),
    "gale crater": MarsRegion("Gale Crater", -5.4, 137.8, 1.5, "Curiosity landing site"),
    "gusev crater": MarsRegion("Gusev Crater", -14.5, 175.4, 1.5, "Spirit landing site"),
    "meridiani planum": MarsRegion("Meridiani Planum", -2.0, 354.0, 5.0, "Opportunity landing site"),
    "viking 1": MarsRegion("Viking 1 Site", 22.3, 312.0, 1.0, "First successful Mars lander"),
    "viking 2": MarsRegion("Viking 2 Site", 47.6, 134.0, 1.0, "Viking 2 landing site"),

    # Notable Craters
    "schiaparelli crater": MarsRegion("Schiaparelli Crater", -3.0, 14.0, 2.5, "Large impact crater"),
    "huygens crater": MarsRegion("Huygens Crater", -14.0, 55.0, 2.5, "Large impact crater"),
    "holden crater": MarsRegion("Holden Crater", -26.0, 326.0, 1.5, "Alluvial fan crater"),
    "eberswalde crater": MarsRegion("Eberswalde Crater", -24.0, 327.0, 0.5, "Ancient delta"),
    "mawrth vallis": MarsRegion("Mawrth Vallis", 22.0, 343.0, 3.0, "Phyllosilicate-rich region"),
    "nili fossae": MarsRegion("Nili Fossae", 21.0, 74.0, 4.0, "Graben system with diverse mineralogy"),

    # Outflow Channels
    "kasei valles": MarsRegion("Kasei Valles", 25.0, 290.0, 8.0, "Largest outflow channel"),
    "ares vallis": MarsRegion("Ares Vallis", 10.0, 335.0, 5.0, "Outflow channel, Pathfinder site"),
    "tiu vallis": MarsRegion("Tiu Vallis", 15.0, 325.0, 4.0, "Outflow channel"),
    "simud vallis": MarsRegion("Simud Vallis", 12.0, 322.0, 3.0, "Outflow channel"),
    "maja vallis": MarsRegion("Maja Vallis", 15.0, 305.0, 3.0, "Outflow channel"),
    "mangala vallis": MarsRegion("Mangala Vallis", -10.0, 210.0, 4.0, "Outflow channel"),

    # Other Notable Features
    "medusae fossae": MarsRegion("Medusae Fossae", 5.0, 195.0, 10.0, "Easily eroded deposit"),
    "cerberus fossae": MarsRegion("Cerberus Fossae", 10.0, 166.0, 5.0, "Young volcanic fissures"),
    "olympus mons aureole": MarsRegion("Olympus Aureole", 20.0, 215.0, 8.0, "Landslide deposits around Olympus"),
}


def find_region(query: str) -> Optional[MarsRegion]:
    """
    Find a Mars region by name (case-insensitive, partial match).

    Args:
        query: Region name to search for

    Returns:
        MarsRegion if found, None otherwise
    """
    query_lower = query.lower().strip()

    # Exact match first
    if query_lower in MARS_REGIONS:
        return MARS_REGIONS[query_lower]

    # Partial match
    for key, region in MARS_REGIONS.items():
        if query_lower in key or key in query_lower:
            return region
        # Also check the region name
        if query_lower in region.name.lower():
            return region

    return None


def find_all_matching_regions(query: str) -> list[MarsRegion]:
    """
    Find all Mars regions matching a query.

    Args:
        query: Region name to search for

    Returns:
        List of matching MarsRegion objects
    """
    query_lower = query.lower().strip()
    matches = []

    for key, region in MARS_REGIONS.items():
        if query_lower in key or query_lower in region.name.lower():
            matches.append(region)

    return matches


def list_all_regions() -> list[MarsRegion]:
    """Return all known Mars regions."""
    return list(MARS_REGIONS.values())


def find_nearest_region(lat: float, lon_360: float) -> Optional[MarsRegion]:
    """
    Find the nearest named region to a lat/lon point (0-360 east lon).

    Scans all MARS_REGIONS, computes angular distance, returns the closest
    region if the point falls within that region's radius_deg.
    """
    best: Optional[MarsRegion] = None
    best_dist = float("inf")

    for region in MARS_REGIONS.values():
        dlat = lat - region.lat
        # Handle longitude wrapping (both in 0-360)
        dlon = lon_360 - region.lon
        if dlon > 180:
            dlon -= 360
        elif dlon < -180:
            dlon += 360
        dist = math.sqrt(dlat * dlat + dlon * dlon)
        if dist < region.radius_deg and dist < best_dist:
            best = region
            best_dist = dist

    return best
