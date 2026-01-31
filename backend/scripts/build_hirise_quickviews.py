# scripts/build_hirise_quickviews.py
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "hirise_data"
OUT_DIR = BASE_DIR / "hirise_quickview"

OUT_DIR.mkdir(exist_ok=True)

DOWNSAMPLE = 64   # 🔥 핵심

def stretch_to_uint8(arr, pmin=2, pmax=98):
    lo, hi = np.percentile(arr, (pmin, pmax))
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)

tifs = sorted(DATA_DIR.glob("ESP_*_RED.tif"))
print(f"[INFO] total tifs: {len(tifs)}")

for tif in tqdm(tifs):
    out_path = OUT_DIR / (tif.stem.replace("_RED", "") + ".jpg")
    if out_path.exists():
        continue

    with rasterio.open(tif) as ds:
        out_h = ds.height // DOWNSAMPLE
        out_w = ds.width // DOWNSAMPLE

        data = ds.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
        )

    img = stretch_to_uint8(data)

    from PIL import Image
    Image.fromarray(img).save(out_path, quality=90)

print("[DONE] quickviews generated")
