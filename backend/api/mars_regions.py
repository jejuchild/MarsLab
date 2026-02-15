"""
Mars named regions lookup table with bounding boxes.

Each region is defined by:
- region_id: machine-readable identifier
- display_name: human-readable name
- lat_min, lat_max, lon_min, lon_max: bounding box (lon in -180/180)
- description: scientific context
- tags: categorization for filtering

All coordinates use -180 to 180 longitude convention.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass, field
import math


@dataclass
class MarsRegion:
    """Mars geographic region defined by a bounding box."""
    region_id: str
    display_name: str
    lat_min: float
    lat_max: float
    lon_min: float  # -180 to 180
    lon_max: float  # -180 to 180
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Derived properties
    @property
    def name(self) -> str:
        return self.display_name

    @property
    def center_lat(self) -> float:
        return (self.lat_min + self.lat_max) / 2

    @property
    def center_lon(self) -> float:
        if self.lon_min <= self.lon_max:
            return (self.lon_min + self.lon_max) / 2
        # Wraps around antimeridian: shift lon_max up by 360, average, then normalize
        c = (self.lon_min + self.lon_max + 360) / 2
        return c if c <= 180 else c - 360

    @property
    def lat(self) -> float:
        """Legacy center latitude."""
        return self.center_lat

    @property
    def lon(self) -> float:
        """Legacy center longitude (0-360 for ODE)."""
        lon = self.center_lon
        if lon < 0:
            lon += 360
        return lon

    @property
    def radius_deg(self) -> float:
        """Approximate radius in degrees (half the larger span)."""
        lat_span = self.lat_max - self.lat_min
        if self.lon_min <= self.lon_max:
            lon_span = self.lon_max - self.lon_min
        else:
            lon_span = 360 + self.lon_max - self.lon_min
        return max(lat_span, lon_span) / 2

    def to_bbox_dict(self) -> Dict:
        """Return as bbox dict for API responses."""
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }

    def to_ode_bbox(self) -> Dict:
        """Return bbox in ODE's 0-360 longitude convention."""
        westernlon = self.lon_min
        easternlon = self.lon_max
        if westernlon < 0:
            westernlon += 360
        if easternlon < 0:
            easternlon += 360
        return {
            "minlat": self.lat_min,
            "maxlat": self.lat_max,
            "westernlon": westernlon,
            "easternlon": easternlon,
        }

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point falls within this region's bbox."""
        if lat < self.lat_min or lat > self.lat_max:
            return False
        # Handle longitude wrapping
        if self.lon_min <= self.lon_max:
            return self.lon_min <= lon <= self.lon_max
        else:
            # Wraps around antimeridian
            return lon >= self.lon_min or lon <= self.lon_max


# =============================================================================
# Region Lookup Table
# =============================================================================

MARS_REGIONS: Dict[str, MarsRegion] = {}

def _r(region_id: str, name: str, lat_min: float, lat_max: float,
       lon_min: float, lon_max: float, desc: str = "", tags: List[str] = None):
    """Helper to register a region."""
    region = MarsRegion(
        region_id=region_id,
        display_name=name,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        description=desc,
        tags=tags or [],
    )
    MARS_REGIONS[region_id] = region

# ---- Major Planitiae (Plains) ----
# All coordinates sourced from IAU/USGS Gazetteer of Planetary Nomenclature
# (planetarynames.wr.usgs.gov), using refined control-network geometry.
# Longitudes converted from 0-360 East to -180/180 convention.

_r("arcadia_planitia", "Arcadia Planitia",
   33.87, 64.17, 165.86, -149.57,
   "Northern lowland plain with subsurface ice deposits",
   ["planitia", "lowland", "ice", "northern"])

_r("amazonis_planitia", "Amazonis Planitia",
   2.14, 49.75, -179.66, -140.58,
   "Smooth lowland plain west of Tharsis",
   ["planitia", "lowland", "volcanic"])

_r("utopia_planitia", "Utopia Planitia",
   12.92, 73.17, 71.86, 164.89,
   "Large impact basin, Zhurong (Tianwen-1) landing site",
   ["planitia", "impact_basin", "landing_site", "ice"])

_r("elysium_planitia", "Elysium Planitia",
   -7.77, 11.40, 128.26, 179.11,
   "InSight landing site, young volcanic terrain",
   ["planitia", "volcanic", "landing_site"])

_r("isidis_planitia", "Isidis Planitia",
   3.20, 22.59, 77.62, 98.91,
   "Impact basin northeast of Syrtis Major",
   ["planitia", "impact_basin"])

_r("hellas_planitia", "Hellas Planitia",
   -55.42, -27.86, 45.58, 96.12,
   "Deepest and largest impact basin on Mars",
   ["planitia", "impact_basin", "deep"])

_r("argyre_planitia", "Argyre Planitia",
   -57.29, -42.44, -53.88, -33.38,
   "Second largest impact basin on Mars",
   ["planitia", "impact_basin"])

_r("chryse_planitia", "Chryse Planitia",
   14.85, 39.61, -54.77, -26.07,
   "Viking 1 landing site, terminus of outflow channels",
   ["planitia", "landing_site", "outflow"])

_r("acidalia_planitia", "Acidalia Planitia",
   14.75, 68.68, -54.88, 16.18,
   "Northern lowland, source region for recurring slope lineae",
   ["planitia", "lowland", "northern"])

# ---- Major Terrae (Highlands) ----
_r("terra_sabaea", "Terra Sabaea",
   -36.85, 42.16, 9.16, 82.00,
   "Southern highlands with ancient terrain",
   ["terra", "highlands", "ancient"])

_r("terra_cimmeria", "Terra Cimmeria",
   -73.54, 12.13, 98.79, 179.66,
   "Ancient cratered highlands",
   ["terra", "highlands", "ancient", "cratered"])

_r("terra_sirenum", "Terra Sirenum",
   -67.57, -11.40, -179.93, -110.32,
   "Southern highlands with chloride salt deposits",
   ["terra", "highlands", "minerals"])

_r("noachis_terra", "Noachis Terra",
   -83.60, -2.53, -59.96, 74.61,
   "Ancient heavily cratered terrain",
   ["terra", "highlands", "ancient", "cratered"])

_r("arabia_terra", "Arabia Terra",
   -18.07, 45.36, -29.69, 49.44,
   "Transitional terrain with buried craters",
   ["terra", "transitional"])

_r("xanthe_terra", "Xanthe Terra",
   -14.11, 17.27, -61.53, -35.06,
   "Chaotic terrain region, outflow channel source",
   ["terra", "chaotic", "outflow"])

# ---- Major Volcanic Regions ----
_r("tharsis", "Tharsis",
   -15.0, 15.0, -115.0, -85.0,
   "Volcanic plateau with the largest volcanoes in the solar system",
   ["volcanic", "plateau", "tharsis"])

_r("olympus_mons", "Olympus Mons",
   13.48, 23.68, -139.24, -127.80,
   "Largest volcano in the solar system, 21.9 km high",
   ["volcanic", "shield_volcano", "tharsis"])

_r("alba_mons", "Alba Mons",
   36.53, 45.63, -116.29, -104.79,
   "Large low-relief shield volcano",
   ["volcanic", "shield_volcano"])

_r("elysium_mons", "Elysium Mons",
   21.78, 28.26, 143.59, 150.82,
   "Shield volcano in Elysium region",
   ["volcanic", "shield_volcano"])

_r("ascraeus_mons", "Ascraeus Mons",
   8.11, 15.92, -107.60, -101.00,
   "Northernmost Tharsis Montes volcano",
   ["volcanic", "shield_volcano", "tharsis"])

_r("pavonis_mons", "Pavonis Mons",
   -1.98, 4.15, -116.20, -109.90,
   "Middle Tharsis Montes volcano on equator",
   ["volcanic", "shield_volcano", "tharsis"])

_r("arsia_mons", "Arsia Mons",
   -11.57, -4.94, -123.80, -116.50,
   "Southernmost Tharsis Montes volcano",
   ["volcanic", "shield_volcano", "tharsis"])

_r("syrtis_major", "Syrtis Major",
   -0.99, 18.79, 59.00, 77.00,
   "Low shield volcano, prominent dark albedo feature",
   ["volcanic", "shield_volcano", "albedo"])

# ---- Major Canyons and Chasmata ----
_r("valles_marineris", "Valles Marineris",
   -19.09, -2.96, -92.70, -28.90,
   "Vast canyon system, 4000 km long and up to 7 km deep",
   ["canyon", "chasma", "tectonic"])

_r("noctis_labyrinthus", "Noctis Labyrinthus",
   -12.26, -2.57, -111.70, -91.40,
   "Labyrinth of intersecting canyons west of Valles Marineris",
   ["canyon", "chasma", "labyrinth"])

_r("ius_chasma", "Ius Chasma",
   -9.97, -5.77, -91.50, -77.47,
   "Western part of Valles Marineris",
   ["canyon", "chasma", "valles_marineris"])

_r("coprates_chasma", "Coprates Chasma",
   -15.82, -10.38, -68.90, -52.70,
   "Central Valles Marineris with layered deposits",
   ["canyon", "chasma", "valles_marineris"])

_r("hebes_chasma", "Hebes Chasma",
   -2.17, 0.15, -78.61, -73.30,
   "Enclosed canyon north of Valles Marineris",
   ["canyon", "chasma", "enclosed"])

_r("juventae_chasma", "Juventae Chasma",
   -5.63, -0.59, -63.60, -60.10,
   "Outflow channel source with sulfate deposits",
   ["canyon", "chasma", "outflow", "minerals"])

# ---- Polar Regions ----
_r("north_pole", "North Polar Region",
   78.0, 90.0, -180.0, 180.0,
   "Northern polar ice cap (water ice + seasonal CO2)",
   ["polar", "ice", "northern"])

_r("south_pole", "South Polar Region",
   -90.0, -78.0, -180.0, 180.0,
   "Southern polar ice cap (CO2 ice + water ice)",
   ["polar", "ice", "southern"])

_r("planum_boreum", "Planum Boreum",
   80.0, 90.0, -180.0, 180.0,
   "North polar plateau with layered deposits",
   ["polar", "ice", "layered_deposits", "northern"])

_r("planum_australe", "Planum Australe",
   -90.0, -80.0, -180.0, 180.0,
   "South polar plateau with layered deposits",
   ["polar", "ice", "layered_deposits", "southern"])

_r("chasma_boreale", "Chasma Boreale",
   79.30, 85.07, -59.38, -16.20,
   "Major canyon cutting into north polar cap",
   ["polar", "canyon", "ice"])

# ---- Landing Sites ----
_r("jezero_crater", "Jezero Crater",
   18.01, 18.81, 77.27, 78.11,
   "Perseverance landing site, ancient river delta",
   ["crater", "landing_site", "delta", "astrobiology"])

_r("gale_crater", "Gale Crater",
   -6.62, -4.05, 136.40, 138.90,
   "Curiosity landing site, Mount Sharp central peak",
   ["crater", "landing_site", "layered_deposits"])

_r("gusev_crater", "Gusev Crater",
   -15.86, -13.20, 174.15, 176.91,
   "Spirit landing site, ancient lakebed",
   ["crater", "landing_site", "lakebed"])

_r("meridiani_planum", "Meridiani Planum",
   -4.00, 8.78, -10.90, 7.00,
   "Opportunity landing site, hematite deposits",
   ["planum", "landing_site", "minerals", "hematite"])

_r("viking_1", "Viking 1 Site",
   21.0, 24.0, -50.0, -46.0,
   "First successful Mars lander site",
   ["landing_site", "historic"])

_r("viking_2", "Viking 2 Site",
   46.0, 49.0, -134.0, -131.0,
   "Viking 2 landing site in Utopia Planitia",
   ["landing_site", "historic"])

# ---- Notable Craters ----
_r("schiaparelli_crater", "Schiaparelli Crater",
   -5.93, 1.09, 12.90, 20.80,
   "Large impact crater 461 km diameter",
   ["crater", "large"])

_r("huygens_crater", "Huygens Crater",
   -17.80, -9.88, 51.40, 59.60,
   "Large impact crater 467 km diameter",
   ["crater", "large"])

_r("holden_crater", "Holden Crater",
   -27.33, -24.75, -35.45, -32.58,
   "Crater with alluvial fans and layered sediments",
   ["crater", "alluvial", "sedimentary"])

_r("eberswalde_crater", "Eberswalde Crater",
   -24.44, -23.52, -33.81, -32.79,
   "Ancient inverted river delta",
   ["crater", "delta", "fluvial"])

_r("mawrth_vallis", "Mawrth Vallis",
   18.69, 26.23, -19.80, -13.40,
   "Phyllosilicate-rich outflow channel",
   ["vallis", "minerals", "phyllosilicate", "fluvial"])

_r("nili_fossae", "Nili Fossae",
   17.01, 27.12, 72.10, 81.70,
   "Graben system with olivine, carbonates, and diverse mineralogy",
   ["fossae", "graben", "minerals", "olivine", "carbonate"])

# ---- Outflow Channels ----
_r("kasei_valles", "Kasei Valles",
   16.71, 27.32, -74.40, -50.00,
   "Largest outflow channel system on Mars",
   ["vallis", "outflow", "fluvial"])

_r("ares_vallis", "Ares Vallis",
   0.33, 20.82, -36.00, -17.24,
   "Outflow channel, Pathfinder landing site nearby",
   ["vallis", "outflow", "fluvial", "landing_site"])

_r("tiu_vallis", "Tiu Vallis",
   2.77, 28.71, -36.70, -28.00,
   "Outflow channel draining into Chryse Planitia",
   ["vallis", "outflow", "fluvial"])

_r("simud_vallis", "Simud Vallis",
   10.97, 27.20, -43.42, -33.93,
   "Outflow channel parallel to Tiu Vallis",
   ["vallis", "outflow", "fluvial"])

_r("maja_vallis", "Maja Vallis",
   0.33, 20.54, -61.62, -49.74,
   "Outflow channel from Juventae Chasma",
   ["vallis", "outflow", "fluvial"])

_r("mangala_vallis", "Mangala Vallis",
   -18.20, -4.30, -151.85, -149.20,
   "Long outflow channel system",
   ["vallis", "outflow", "fluvial"])

# ---- Other Notable Features ----
_r("medusae_fossae", "Medusae Fossae",
   -2.0, 12.0, -175.0, -155.0,
   "Easily eroded volcanic ash or dust deposits",
   ["fossae", "volcanic", "aeolian"])

_r("cerberus_fossae", "Cerberus Fossae",
   6.23, 16.16, 154.43, 174.72,
   "Young volcanic fissures with recent lava flows",
   ["fossae", "volcanic", "young"])

_r("olympus_aureole", "Olympus Mons Aureole",
   12.38, 34.48, -150.75, -130.83,
   "Massive landslide deposits surrounding Olympus Mons",
   ["volcanic", "landslide", "aureole"])


# =============================================================================
# Lookup Functions
# =============================================================================

# Build name-based index for fast lookup
_NAME_INDEX: Dict[str, MarsRegion] = {}

def _build_name_index():
    """Build case-insensitive name lookup index."""
    for region in MARS_REGIONS.values():
        # Index by region_id
        _NAME_INDEX[region.region_id.lower()] = region
        # Index by display_name
        _NAME_INDEX[region.display_name.lower()] = region
        # Index by display_name without common suffixes for partial matching
        name_lower = region.display_name.lower()
        for suffix in [" planitia", " terra", " mons", " chasma", " vallis",
                       " valles", " fossae", " planum", " crater", " site",
                       " region", " labyrinthus", " aureole"]:
            if name_lower.endswith(suffix):
                base = name_lower[:-len(suffix)].strip()
                if base and base not in _NAME_INDEX:
                    _NAME_INDEX[base] = region

_build_name_index()


def find_region(query: str) -> Optional[MarsRegion]:
    """
    Find a Mars region by name (case-insensitive, partial match).

    Args:
        query: Region name to search for

    Returns:
        MarsRegion if found, None otherwise
    """
    query_lower = query.lower().strip()

    # Exact match on region_id or display_name
    if query_lower in _NAME_INDEX:
        return _NAME_INDEX[query_lower]

    # Try matching with underscores replaced by spaces
    query_spaced = query_lower.replace("_", " ")
    if query_spaced in _NAME_INDEX:
        return _NAME_INDEX[query_spaced]

    # Partial match: check if query is substring of any key
    for key, region in _NAME_INDEX.items():
        if query_lower in key or key in query_lower:
            return region

    return None


def find_all_matching_regions(query: str) -> List[MarsRegion]:
    """
    Find all Mars regions matching a query.

    Args:
        query: Region name to search for

    Returns:
        List of matching MarsRegion objects
    """
    query_lower = query.lower().strip()
    seen_ids = set()
    matches = []

    for region in MARS_REGIONS.values():
        if region.region_id in seen_ids:
            continue
        name_lower = region.display_name.lower()
        id_lower = region.region_id.lower()
        if query_lower in name_lower or query_lower in id_lower:
            matches.append(region)
            seen_ids.add(region.region_id)

    return matches


def find_regions_by_tag(tag: str) -> List[MarsRegion]:
    """Find all regions with a given tag."""
    tag_lower = tag.lower()
    return [r for r in MARS_REGIONS.values() if tag_lower in [t.lower() for t in r.tags]]


def list_all_regions() -> List[MarsRegion]:
    """Return all known Mars regions."""
    return list(MARS_REGIONS.values())


def get_region_by_id(region_id: str) -> Optional[MarsRegion]:
    """Get a region by its exact region_id."""
    return MARS_REGIONS.get(region_id)


def find_nearest_region(lat: float, lon_180: float) -> Optional[MarsRegion]:
    """
    Find the nearest named region to a lat/lon point (-180/180 lon).

    Checks if the point falls within any region's bounding box.
    Falls back to closest region by center distance.
    """
    # First: check direct containment
    for region in MARS_REGIONS.values():
        if region.contains_point(lat, lon_180):
            return region

    # Fallback: closest center
    best: Optional[MarsRegion] = None
    best_dist = float("inf")

    for region in MARS_REGIONS.values():
        dlat = lat - region.center_lat
        dlon = lon_180 - region.center_lon
        if dlon > 180:
            dlon -= 360
        elif dlon < -180:
            dlon += 360
        dist = math.sqrt(dlat * dlat + dlon * dlon)
        if dist < best_dist:
            best = region
            best_dist = dist

    return best
