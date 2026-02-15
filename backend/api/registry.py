"""
Instrument Registry - Single source of truth for instrument configuration.
Replaces all hardcoded instrument-specific logic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import json
import re


@dataclass
class InstrumentConfig:
    """Configuration for a single instrument."""
    id: str
    name: str
    display_name: str
    geometry_type: str  # Polygon, LineString, Point
    color: str  # Hex color for UI
    data_directory: str
    index_file: str
    product_id_pattern: str
    quickview_path: str
    supports_spectrum: bool
    supports_rgb: bool
    file_types: Dict[str, List[str]]
    browse_path: Optional[str] = None
    highres_path: Optional[str] = None

    _compiled_pattern: Optional[re.Pattern] = None

    def __post_init__(self):
        """Compile the regex pattern for efficient matching."""
        self._compiled_pattern = re.compile(self.product_id_pattern, re.IGNORECASE)

    def matches_product_id(self, product_id: str) -> bool:
        """Check if a product ID matches this instrument's pattern."""
        if self._compiled_pattern is None:
            self._compiled_pattern = re.compile(self.product_id_pattern, re.IGNORECASE)
        return bool(self._compiled_pattern.match(product_id))

    def get_index_path(self, base_dir: Path) -> Path:
        """Get the full path to the index.geojson file."""
        return base_dir / self.data_directory / self.index_file

    def get_data_dir(self, base_dir: Path) -> Path:
        """Get the full path to the data directory."""
        return base_dir / self.data_directory

    def get_quickview_dir(self, base_dir: Path) -> Path:
        """Get the full path to the quickview directory."""
        return base_dir / self.quickview_path

    def get_browse_dir(self, base_dir: Path) -> Optional[Path]:
        """Get the full path to the browse directory (if available)."""
        if self.browse_path:
            return base_dir / self.browse_path
        return None

    def get_highres_dir(self, base_dir: Path) -> Optional[Path]:
        """Get the full path to the high-res directory (if available)."""
        if self.highres_path:
            return base_dir / self.highres_path
        return None


class InstrumentRegistry:
    """Registry of all supported instruments (singleton)."""

    _instance: Optional['InstrumentRegistry'] = None
    _initialized: bool = False

    def __new__(cls, config_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None):
        if self._initialized:
            return

        if config_path is None:
            config_path = Path(__file__).parent.parent / "data" / "registry.json"

        self._config_path = config_path
        self._base_dir = config_path.parent.parent  # backend/
        self._instruments: Dict[str, InstrumentConfig] = {}
        self._load_config()
        self._initialized = True

    def _load_config(self):
        """Load instrument configuration from registry.json."""
        with open(self._config_path) as f:
            data = json.load(f)

        for inst_id, inst_data in data.get("instruments", {}).items():
            self._instruments[inst_id] = InstrumentConfig(
                id=inst_data["id"],
                name=inst_data["name"],
                display_name=inst_data["display_name"],
                geometry_type=inst_data["geometry_type"],
                color=inst_data["color"],
                data_directory=inst_data["data_directory"],
                index_file=inst_data["index_file"],
                product_id_pattern=inst_data["product_id_pattern"],
                quickview_path=inst_data.get("quickview_path", ""),
                supports_spectrum=inst_data["supports_spectrum"],
                supports_rgb=inst_data["supports_rgb"],
                file_types=inst_data.get("file_types", {}),
                browse_path=inst_data.get("browse_path"),
                highres_path=inst_data.get("highres_path"),
            )

    @property
    def base_dir(self) -> Path:
        """Get the base directory (backend/)."""
        return self._base_dir

    def get(self, instrument_id: str) -> Optional[InstrumentConfig]:
        """Get instrument configuration by ID (case-insensitive)."""
        return self._instruments.get(instrument_id.lower())

    def all(self) -> List[InstrumentConfig]:
        """Get all registered instruments."""
        return list(self._instruments.values())

    def ids(self) -> List[str]:
        """Get all registered instrument IDs."""
        return list(self._instruments.keys())

    def detect_instrument(self, product_id: str) -> Optional[InstrumentConfig]:
        """Detect instrument from product ID using pattern matching."""
        for config in self._instruments.values():
            if config.matches_product_id(product_id):
                return config
        return None

    @lru_cache(maxsize=8)
    def load_index(self, instrument_id: str) -> Dict[str, Any]:
        """Load and cache GeoJSON index for an instrument."""
        config = self.get(instrument_id)
        if not config:
            raise ValueError(f"Unknown instrument: {instrument_id}")

        index_path = config.get_index_path(self._base_dir)
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")

        with open(index_path) as f:
            return json.load(f)

    def clear_index_cache(self):
        """Clear the cached indexes."""
        self.load_index.cache_clear()


# Singleton accessor
_registry: Optional[InstrumentRegistry] = None

def get_registry() -> InstrumentRegistry:
    """Get the global instrument registry instance."""
    global _registry
    if _registry is None:
        _registry = InstrumentRegistry()
    return _registry


def get_instrument(instrument_id: str) -> Optional[InstrumentConfig]:
    """Convenience function to get an instrument config."""
    return get_registry().get(instrument_id)


def detect_instrument(product_id: str) -> Optional[InstrumentConfig]:
    """Convenience function to detect instrument from product ID."""
    return get_registry().detect_instrument(product_id)
