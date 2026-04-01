#!/usr/bin/env python3
"""
Phase 2: Download full-res Mastcam-Z cube tiles and stitch into equirectangular panoramas.
Memory-optimized: stitches faces to disk one at a time, then streams equirectangular conversion
loading only the needed face pixels per chunk.
"""

import os
import sys
import math
import time
import gc
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
BASE_URL = "https://maps.planet.fu-berlin.de/vtour"
TOUR_XML = f"{BASE_URL}/tour1.xml"
DOWNLOAD_DIR = Path("/disk1/cspark/mastcam/downloads")
FULL_DIR = DOWNLOAD_DIR / "full"
TILES_DIR = FULL_DIR / "tiles"
FACES_DIR = FULL_DIR / "faces"  # intermediate stitched faces
TILE_SIZE = 512
FACES = ['f', 'r', 'b', 'l', 'u', 'd']
STEREO_LABEL = "1"
MAX_WORKERS = 8
MAX_RETRIES = 3
CHUNK_ROWS = 128  # rows per equirect chunk

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (research-crawler; mastcam-z-panorama)'
})


def fetch_url(url, dest_path, retries=MAX_RETRIES):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                dest_path.write_bytes(r.content)
                return True
            elif r.status_code == 404:
                return False
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"  FAIL ({e}): {url}")
            time.sleep(1)
    return False


def parse_tour_xml():
    print("Fetching tour1.xml...", flush=True)
    r = session.get(TOUR_XML, timeout=30)
    r.raise_for_status()
    content = r.content.decode('utf-8-sig')
    root = ET.fromstring(content)
    scenes = []
    for scene in root.findall('.//scene'):
        name = scene.get('name', '')
        title = scene.get('title', '')
        if 'mastcam' not in name.lower() and 'mastcam' not in title.lower():
            continue
        cube_el = scene.find('.//cube')
        if cube_el is None:
            continue
        multires_str = cube_el.get('multires', '')
        multires = [int(x) for x in multires_str.split(',')] if multires_str else []
        scenes.append({
            'name': name,
            'title': title,
            'cube_url_template': cube_el.get('url', ''),
            'multires': multires,
        })
    print(f"Found {len(scenes)} Mastcam-Z scenes", flush=True)
    return scenes


def resolve_tile_url(template, face, level, row, col):
    url = template
    url = url.replace('%t', STEREO_LABEL)
    url = url.replace('%s', face)
    url = url.replace('%l', str(level))
    url = url.replace('%0v', f"{row:02d}")
    url = url.replace('%0h', f"{col:02d}")
    url = url.replace('%v', str(row))
    url = url.replace('%h', str(col))
    return url


def download_scene_tiles(scene):
    """Download tiles at highest resolution level."""
    multires = scene['multires']
    level = len(multires) - 1  # always highest
    tile_size = multires[0]
    face_size = multires[level]
    ntiles = math.ceil(face_size / tile_size)

    safe_name = scene['title'].replace('/', '_').replace(' ', '_')
    scene_tile_dir = TILES_DIR / safe_name / f"l{level}"

    total = len(FACES) * ntiles * ntiles
    print(f"  Level {level} (max): {face_size}px/face, {ntiles}x{ntiles} tiles/face, {total} total", flush=True)

    downloaded = 0
    failed = 0

    def dl(face, row, col):
        rel = resolve_tile_url(scene['cube_url_template'], face, level, row, col)
        url = f"{BASE_URL}/{rel}"
        dest = scene_tile_dir / face / f"{face}_{row:02d}_{col:02d}.jpg"
        return fetch_url(url, dest)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for face in FACES:
            for row in range(1, ntiles + 1):
                for col in range(1, ntiles + 1):
                    futs.append(ex.submit(dl, face, row, col))
        for f in as_completed(futs):
            if f.result():
                downloaded += 1
            else:
                failed += 1

    print(f"  Tiles: {downloaded} ok, {failed} failed", flush=True)
    return {
        'safe_name': safe_name,
        'tile_dir': scene_tile_dir,
        'level': level,
        'tile_size': tile_size,
        'face_size': face_size,
        'ntiles': ntiles,
    }


def stitch_face_to_disk(tile_info, face, out_path):
    """Stitch one cube face from tiles and save to disk as raw numpy memmap.
    Returns the face_size. Only one face in memory at a time.
    """
    ts = tile_info['tile_size']
    fs = tile_info['face_size']
    n = tile_info['ntiles']
    td = tile_info['tile_dir'] / face

    # Use memmap to avoid holding entire face in RAM
    fp = np.memmap(str(out_path), dtype=np.uint8, mode='w+', shape=(fs, fs, 3))

    for row in range(1, n + 1):
        for col in range(1, n + 1):
            p = td / f"{face}_{row:02d}_{col:02d}.jpg"
            if p.exists():
                tile = np.array(Image.open(p))
                y0 = (row - 1) * ts
                x0 = (col - 1) * ts
                h, w = tile.shape[:2]
                h = min(h, fs - y0)
                w = min(w, fs - x0)
                fp[y0:y0+h, x0:x0+w] = tile[:h, :w, :3]
                del tile

    fp.flush()
    del fp
    return fs


def cube_to_equirect_streaming(tile_info, output_path):
    """Convert cube faces to equirectangular by streaming from disk-backed memmaps.
    Each face is a memmap file — only pages accessed are loaded into RAM.
    """
    fs = tile_info['face_size']
    safe_name = tile_info['safe_name']
    out_w = fs * 4
    out_h = out_w // 2

    face_dir = FACES_DIR / safe_name
    face_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output: {out_w}x{out_h} (face: {fs}px)", flush=True)

    # Step 1: stitch each face to a memmap file on disk (one at a time)
    face_paths = {}
    for fk in FACES:
        fp = face_dir / f"{fk}.raw"
        if fp.exists() and fp.stat().st_size == fs * fs * 3:
            print(f"  Face '{fk}' already stitched, reusing", flush=True)
        else:
            print(f"  Stitching face '{fk}' to disk...", flush=True)
            stitch_face_to_disk(tile_info, fk, fp)
        face_paths[fk] = fp
    gc.collect()

    # Step 2: open all faces as read-only memmaps
    faces = {}
    for fk in FACES:
        faces[fk] = np.memmap(str(face_paths[fk]), dtype=np.uint8, mode='r', shape=(fs, fs, 3))

    # Step 3: create output memmap
    tmp_output = face_dir / "equirect.raw"
    output = np.memmap(str(tmp_output), dtype=np.uint8, mode='w+', shape=(out_h, out_w, 3))

    x_coords = np.arange(out_w, dtype=np.float32)
    lon_base = (x_coords / out_w - 0.5) * 2 * np.pi

    for chunk_start in range(0, out_h, CHUNK_ROWS):
        chunk_end = min(chunk_start + CHUNK_ROWS, out_h)
        chunk_h = chunk_end - chunk_start

        y_coords = np.arange(chunk_start, chunk_end, dtype=np.float32)
        lat = (0.5 - y_coords / out_h) * np.pi

        lat_2d = lat[:, np.newaxis]
        lon_2d = lon_base[np.newaxis, :]

        dx = np.cos(lat_2d) * np.sin(lon_2d)
        dy = np.broadcast_to(np.sin(lat_2d), (chunk_h, out_w)).copy()
        dz = np.cos(lat_2d) * np.cos(lon_2d)

        abs_x = np.abs(dx)
        abs_y = np.abs(dy)
        abs_z = np.abs(dz)

        face_masks = {
            'f': (abs_z >= abs_x) & (abs_z >= abs_y) & (dz > 0),
            'b': (abs_z >= abs_x) & (abs_z >= abs_y) & (dz <= 0),
            'r': (abs_x > abs_z) & (abs_x >= abs_y) & (dx > 0),
            'l': (abs_x > abs_z) & (abs_x >= abs_y) & (dx <= 0),
            'u': (abs_y > abs_x) & (abs_y > abs_z) & (dy > 0),
            'd': (abs_y > abs_x) & (abs_y > abs_z) & (dy <= 0),
        }

        uv_funcs = {
            'f': lambda: (dx / abs_z, -dy / abs_z),
            'b': lambda: (-dx / abs_z, -dy / abs_z),
            'r': lambda: (-dz / abs_x, -dy / abs_x),
            'l': lambda: (dz / abs_x, -dy / abs_x),
            'u': lambda: (dx / abs_y, dz / abs_y),
            'd': lambda: (dx / abs_y, -dz / abs_y),
        }

        chunk_out = np.zeros((chunk_h, out_w, 3), dtype=np.uint8)

        for fk in FACES:
            mask = face_masks[fk]
            if not np.any(mask):
                continue
            u, v = uv_funcs[fk]()
            px = ((u[mask] + 1) / 2 * (fs - 1)).astype(np.int32)
            py = ((v[mask] + 1) / 2 * (fs - 1)).astype(np.int32)
            np.clip(px, 0, fs - 1, out=px)
            np.clip(py, 0, fs - 1, out=py)
            chunk_out[mask] = faces[fk][py, px]

        output[chunk_start:chunk_end] = chunk_out
        pct = chunk_end / out_h * 100
        if int(pct) % 10 == 0:
            print(f"  Equirect: {pct:.0f}%", flush=True)

    print(f"  Equirect: 100%", flush=True)

    # Close memmaps
    for fk in FACES:
        del faces[fk]
    del faces

    # Step 4: save as JPEG
    # For very large images (>20000px wide), PIL can't handle the full array.
    # Solution: save at reduced resolution for huge images, full res for normal ones.
    MAX_OUTPUT_WIDTH = 40000  # 9472*4=37888 fits, 29952*4=119808 doesn't
    if out_w > MAX_OUTPUT_WIDTH:
        # Downsample by reading stripes from memmap
        scale = MAX_OUTPUT_WIDTH / out_w
        save_w = MAX_OUTPUT_WIDTH
        save_h = int(out_h * scale)
        print(f"  Downsampling {out_w}x{out_h} -> {save_w}x{save_h} for JPEG...", flush=True)
        # Build downsampled image in chunks
        stripe_h = 256
        result = np.zeros((save_h, save_w, 3), dtype=np.uint8)
        for sy in range(0, save_h, stripe_h):
            ey = min(sy + stripe_h, save_h)
            # Source rows
            src_sy = int(sy / scale)
            src_ey = int(ey / scale)
            if src_ey > out_h:
                src_ey = out_h
            stripe = np.array(output[src_sy:src_ey])
            pil_stripe = Image.fromarray(stripe)
            pil_stripe = pil_stripe.resize((save_w, ey - sy), Image.LANCZOS)
            result[sy:ey] = np.array(pil_stripe)
            del stripe, pil_stripe

        img = Image.fromarray(result)
        del result, output
        gc.collect()
    else:
        print(f"  Saving JPEG ({out_w}x{out_h})...", flush=True)
        # Read from memmap directly — PIL can handle fromarray on a memmap
        img = Image.fromarray(output)
        del output
        gc.collect()

    img.save(str(output_path), 'JPEG', quality=92)
    print(f"  Saved: {output_path} ({img.size[0]}x{img.size[1]})", flush=True)
    del img
    gc.collect()

    # Cleanup raw files
    tmp_output.unlink(missing_ok=True)
    for fk in FACES:
        (face_dir / f"{fk}.raw").unlink(missing_ok=True)
    face_dir.rmdir()


def main():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)

    scenes = parse_tour_xml()
    if not scenes:
        print("No scenes found!", flush=True)
        sys.exit(1)

    for i, s in enumerate(scenes, 1):
        mr = s['multires']
        max_px = mr[-1] if mr else '?'
        print(f"  {i:2d}. {s['title']} ({max_px}px)", flush=True)

    print(flush=True)

    done = 0
    skipped = 0
    for i, scene in enumerate(scenes, 1):
        title = scene['title']
        safe_name = title.replace('/', '_').replace(' ', '_')
        output_path = FULL_DIR / f"{safe_name}_equirectangular.jpg"

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[{i}/{len(scenes)}] {title} -- already done", flush=True)
            skipped += 1
            continue

        multires = scene['multires']
        level = len(multires) - 1
        print(f"\n[{i}/{len(scenes)}] {title} ({multires[level]}px, max res)", flush=True)

        tile_info = download_scene_tiles(scene)

        try:
            cube_to_equirect_streaming(tile_info, output_path)
            done += 1
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

        gc.collect()

    print(f"\n{'=' * 60}", flush=True)
    print(f"PHASE 2 COMPLETE! {done} new + {skipped} existing", flush=True)
    print(f"Panoramas: {FULL_DIR}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == '__main__':
    main()
