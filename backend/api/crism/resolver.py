import os
import re
import logging

from api.validation import validate_product_id

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CRISM_DATA_ROOT = os.path.join(BASE_DIR, "crism_data")

def resolve_crism_paths(product_id: str):
    product_id = validate_product_id(product_id)

    hdr = os.path.join(CRISM_DATA_ROOT, f"{product_id}.hdr")
    img = os.path.join(CRISM_DATA_ROOT, f"{product_id}.img")

    # if → wv substitution
    wv_product_id = re.sub(
        r"_if([0-9a-z]+)_",
        r"_wv\1_",
        product_id,
        flags=re.I,
    )
    wv = os.path.join(CRISM_DATA_ROOT, f"{wv_product_id}.tab")

    for p in [hdr, img, wv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    return hdr, img, wv
