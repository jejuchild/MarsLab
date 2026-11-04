"""
Which standard SHARAD (THM) products have a quickview image.

132 of the 580 indexed standard tracks have no browse image; clicking one
only ever showed "Image not available", so they are left out of the map and
the Inspector. The set is re-read whenever the quickview directory changes
(the background fetcher may add images).
"""

import os
import threading

QUICKVIEW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sharad_quickview")

_lock = threading.Lock()
_state: dict = {"mtime": None, "ids": frozenset()}


def _ids() -> frozenset:
    try:
        mtime = os.stat(QUICKVIEW_DIR).st_mtime
    except OSError:
        return frozenset()
    with _lock:
        if _state["mtime"] != mtime:
            _state["ids"] = frozenset(
                name[:-4].lower() for name in os.listdir(QUICKVIEW_DIR) if name.lower().endswith(".jpg")
            )
            _state["mtime"] = mtime
        return _state["ids"]


def has_sharad_quickview(product_id: str) -> bool:
    return product_id.lower() in _ids()


def keep_viewable_sharad(features: list) -> list:
    """Drop standard SHARAD features that have no quickview image."""
    ids = _ids()
    return [f for f in features if str(f.get("properties", {}).get("product_id", "")).lower() in ids]
