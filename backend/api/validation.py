import os
import re


_PRODUCT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_product_id(product_id: str) -> str:
    if not product_id:
        raise ValueError("product_id is required")
    if not _PRODUCT_ID_RE.match(product_id):
        raise ValueError(f"Invalid product_id format: {product_id}")
    if ".." in product_id or "/" in product_id or "\\" in product_id:
        raise ValueError(f"Invalid product_id path traversal: {product_id}")
    if os.path.basename(product_id) != product_id:
        raise ValueError(f"Invalid product_id path segment: {product_id}")
    return product_id
