"""Spatial index of classified HiRISE landform observations.

Maintains an in-memory cache of landform classification results keyed by
product-id, with lat/lon bounds for spatial lookup.  Persisted to JSON on
disk so data survives server restarts.

Thread-safe: all mutations go through a lock.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "Data" / "HiRISE" / "landform_cache.json"
)

# Mars pixel scale (HiRISE browse ~25 m/px, images ~6000x10000 px)
_HIRISE_BROWSE_PIXEL_SCALE_M = 25.0
_MARS_CIRCUMFERENCE_M = 2 * np.pi * 3_389_500


@dataclass
class LandformEntry:
    """A single classified HiRISE observation."""

    product_id: str
    lat: float
    lon: float

    # Bounding box (approximate, from image centre + size estimate)
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    dominant_class: str  # LDA, LVF, CCF, OTHER
    confidence: float  # 0-1

    # Per-tile predictions (optional, for sub-image resolution)
    tile_classes: Dict[str, str] = field(default_factory=dict)  # "row_col" -> class
    tile_confidences: Dict[str, float] = field(default_factory=dict)

    classified_at: str = ""  # ISO timestamp
    model_version: str = ""

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)


class LandformCache:
    """Spatial index of classified HiRISE landform results."""

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self._path = cache_path or _DEFAULT_CACHE_PATH
        self._entries: Dict[str, LandformEntry] = {}
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        product_id: str,
        lat: float,
        lon: float,
        dominant_class: str,
        confidence: float,
        tile_predictions: Optional[List[dict]] = None,
        model_version: str = "",
        img_width: int = 6000,
        img_height: int = 10000,
    ) -> None:
        """Register a landform classification result."""
        bounds = self._estimate_bounds(lat, lon, img_width, img_height)

        tile_classes: Dict[str, str] = {}
        tile_confidences: Dict[str, float] = {}
        if tile_predictions:
            for tp in tile_predictions:
                key = f"{tp.get('y', 0)}_{tp.get('x', 0)}"
                tile_classes[key] = tp.get("predicted_class", "OTHER")
                tile_confidences[key] = tp.get("confidence", 0.0)

        entry = LandformEntry(
            product_id=product_id,
            lat=lat,
            lon=lon,
            lat_min=bounds[0],
            lat_max=bounds[1],
            lon_min=bounds[2],
            lon_max=bounds[3],
            dominant_class=dominant_class,
            confidence=confidence,
            tile_classes=tile_classes,
            tile_confidences=tile_confidences,
            classified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_version=model_version,
        )

        with self._lock:
            self._entries[product_id] = entry
            self._save()

        logger.info(
            "Landform cache: registered %s → %s (%.2f) at (%.2f, %.2f)",
            product_id, dominant_class, confidence, lat, lon,
        )

    def lookup(self, lat: float, lon: float) -> Optional[str]:
        """Find the dominant landform class at (lat, lon), or None."""
        best = self.lookup_entry(lat, lon)
        if best is None:
            return None
        return best.dominant_class

    def lookup_entry(self, lat: float, lon: float) -> Optional[LandformEntry]:
        """Find the best matching LandformEntry at (lat, lon), or None."""
        with self._lock:
            candidates: List[Tuple[float, LandformEntry]] = []
            for entry in self._entries.values():
                if (entry.lat_min <= lat <= entry.lat_max and
                        entry.lon_min <= lon <= entry.lon_max):
                    # Distance to image centre (for tie-breaking)
                    d = np.sqrt((entry.lat - lat) ** 2 + (entry.lon - lon) ** 2)
                    candidates.append((d, entry))

        if not candidates:
            return None

        # Prefer highest confidence, then closest to centre
        candidates.sort(key=lambda x: (-x[1].confidence, x[0]))
        return candidates[0][1]

    def lookup_tile_class(
        self, lat: float, lon: float
    ) -> Optional[Tuple[str, float]]:
        """Lookup at tile resolution: returns (class, confidence) or None."""
        entry = self.lookup_entry(lat, lon)
        if entry is None:
            return None

        if not entry.tile_classes:
            return (entry.dominant_class, entry.confidence)

        # Find which tile this point falls in
        tile_key = self._point_to_tile_key(
            lat, lon, entry.lat, entry.lon,
            entry.lat_min, entry.lat_max, entry.lon_min, entry.lon_max,
            n_tile_rows=max(1, len(set(k.split("_")[0] for k in entry.tile_classes))),
            n_tile_cols=max(1, len(set(k.split("_")[1] for k in entry.tile_classes))),
        )

        if tile_key in entry.tile_classes:
            return (
                entry.tile_classes[tile_key],
                entry.tile_confidences.get(tile_key, entry.confidence),
            )

        return (entry.dominant_class, entry.confidence)

    def all_entries(self) -> List[LandformEntry]:
        """Return all cached entries."""
        with self._lock:
            return list(self._entries.values())

    def get_entries_in_bounds(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
    ) -> List[LandformEntry]:
        """Return entries whose bounding boxes overlap the query region."""
        with self._lock:
            results: List[LandformEntry] = []
            for entry in self._entries.values():
                if (entry.lat_max >= lat_min and entry.lat_min <= lat_max and
                        entry.lon_max >= lon_min and entry.lon_min <= lon_max):
                    results.append(entry)
            return results

    @property
    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_bounds(
        lat: float, lon: float,
        img_w: int = 6000, img_h: int = 10000,
    ) -> Tuple[float, float, float, float]:
        """Estimate lat/lon bounding box from image centre + pixel dimensions."""
        deg_per_m_lat = 360.0 / _MARS_CIRCUMFERENCE_M
        deg_per_m_lon = deg_per_m_lat / max(np.cos(np.radians(lat)), 0.01)

        half_h_m = img_h * _HIRISE_BROWSE_PIXEL_SCALE_M / 2.0
        half_w_m = img_w * _HIRISE_BROWSE_PIXEL_SCALE_M / 2.0

        lat_min = lat - half_h_m * deg_per_m_lat
        lat_max = lat + half_h_m * deg_per_m_lat
        lon_min = lon - half_w_m * deg_per_m_lon
        lon_max = lon + half_w_m * deg_per_m_lon

        return (lat_min, lat_max, lon_min, lon_max)

    @staticmethod
    def _point_to_tile_key(
        lat: float, lon: float,
        center_lat: float, center_lon: float,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        n_tile_rows: int, n_tile_cols: int,
    ) -> str:
        """Map a (lat, lon) point to a tile grid key 'row_col'."""
        # Row: top (lat_max) = 0, bottom (lat_min) = n_rows-1
        frac_row = (lat_max - lat) / max(lat_max - lat_min, 1e-6)
        row = int(np.clip(frac_row * n_tile_rows, 0, n_tile_rows - 1))

        frac_col = (lon - lon_min) / max(lon_max - lon_min, 1e-6)
        col = int(np.clip(frac_col * n_tile_cols, 0, n_tile_cols - 1))

        return f"{row}_{col}"

    def _load(self) -> None:
        """Load cache from disk."""
        if not self._path.exists():
            logger.info("Landform cache: no file at %s, starting empty", self._path)
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for pid, d in data.items():
                self._entries[pid] = LandformEntry(**d)
            logger.info("Landform cache: loaded %d entries from %s", len(self._entries), self._path)
        except Exception as exc:
            logger.warning("Landform cache: failed to load %s: %s", self._path, exc)

    def _save(self) -> None:
        """Persist cache to disk. Caller must hold _lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {pid: asdict(e) for pid, e in self._entries.items()}
            self._path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Landform cache: failed to save: %s", exc)
