"""Utilities for locating and managing SHARAD cluttergram data."""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Filesystem paths
SHARAD_HR_DIR = Path("/disk1/cspark/MarsLab/backend/sharad_highres")


def extract_obs_id(product_id: str) -> str:
    """Extract observation ID from SHARAD product ID and zero-pad to 8 digits.

    Example: "R_0277201_001_SS19_700_A" → "00277201"

    Args:
        product_id: SHARAD_HIGHRES product ID

    Returns:
        8-digit zero-padded observation ID
    """
    parts = product_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid product_id format: {product_id}")
    obs_id = parts[1]
    return obs_id.zfill(8)


def find_clutter_pair(product_id: str) -> Optional[Tuple[str, str]]:
    """Resolve cluttergram (simulated clutter) file pair for a SHARAD product.

    Searches for cluttergram files (s_XXXXXXXX_sim.img and .xml) in the
    SHARAD_HR_DIR. If not found, returns None (graceful degradation).

    Args:
        product_id: SHARAD_HIGHRES product ID (e.g., "R_0277201_001_SS19_700_A")

    Returns:
        Tuple (img_path, xml_path) as strings if found, else None
    """
    try:
        sim_id = extract_obs_id(product_id)
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to extract obs_id from {product_id}: {e}")
        return None

    # Construct expected cluttergram filenames
    img_name = f"s_{sim_id}_sim.img"
    xml_name = f"s_{sim_id}_sim.xml"

    img_path = SHARAD_HR_DIR / img_name
    xml_path = SHARAD_HR_DIR / xml_name

    if not img_path.exists() or not xml_path.exists():
        logger.debug(
            f"Cluttergram not found for {product_id} (sim_id={sim_id}): "
            f"img_exists={img_path.exists()}, xml_exists={xml_path.exists()}"
        )
        return None

    logger.info(
        f"Found cluttergram pair for {product_id}: "
        f"{img_path.name}, {xml_path.name}"
    )
    return (str(img_path), str(xml_path))
