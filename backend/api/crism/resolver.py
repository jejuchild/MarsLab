import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CRISM_DATA_ROOT = os.path.join(BASE_DIR, "crism_data")
print("===== CRISM DEBUG =====")
print("CRISM_DATA_ROOT =", CRISM_DATA_ROOT)
print("exists =", os.path.exists(CRISM_DATA_ROOT))
print("=======================")

def resolve_crism_paths(product_id: str):
    # product_id 예:
    # frt00009c0a_07_if164j_mtr3

    hdr = os.path.join(CRISM_DATA_ROOT, f"{product_id}.hdr")
    img = os.path.join(CRISM_DATA_ROOT, f"{product_id}.img")

    # if → wv 치환
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
