#!/usr/bin/env python3
"""
Mastcam-Z 360° Panorama Crawler
Crawls all Mastcam-Z panorama images from FU Berlin Jezero Crater site.
Phase 1: Download previews/thumbnails
Phase 2: Download full-res cube tiles and stitch into equirectangular panoramas
"""

import os
import sys
import math
import time
import re
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
    import numpy as np
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False

# ── Config ──────────────────────────────────────────────────────────────
BASE_URL = "https://maps.planet.fu-berlin.de/vtour"
TOUR_XML = f"{BASE_URL}/tour1.xml"
DOWNLOAD_DIR = Path("/disk1/cspark/mastcam/downloads")
PREVIEW_DIR = DOWNLOAD_DIR / "previews"
FULL_DIR = DOWNLOAD_DIR / "full"
TILES_DIR = FULL_DIR / "tiles"
MAX_RETRIES = 3
DELAY = 0.1  # seconds between requests
TILE_SIZE = 512
FACES = ['f', 'r', 'b', 'l', 'u', 'd']  # front, right, back, left, up, down
STEREO_LABEL = "1"  # left eye
MAX_WORKERS = 8

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (research-crawler; mastcam-z-panorama)'
})


def fetch_url(url, dest_path, retries=MAX_RETRIES):
    """Download a URL to a file with retry logic. Returns True on success."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True  # already downloaded
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
    """Parse tour1.xml and extract all Mastcam scene definitions."""
    print("Fetching tour1.xml...")
    r = session.get(TOUR_XML, timeout=30)
    r.raise_for_status()

    # Fix XML: remove BOM if present and handle encoding
    content = r.content.decode('utf-8-sig')
    root = ET.fromstring(content)

    scenes = []
    for scene in root.findall('.//scene'):
        name = scene.get('name', '')
        title = scene.get('title', '')
        thumburl = scene.get('thumburl', '')

        # Only Mastcam scenes
        if 'mastcam' not in name.lower() and 'mastcam' not in title.lower():
            continue

        preview_el = scene.find('.//preview')
        cube_el = scene.find('.//cube')
        image_el = scene.find('.//image')

        preview_url = preview_el.get('url', '') if preview_el is not None else ''
        cube_url_template = cube_el.get('url', '') if cube_el is not None else ''
        multires_str = cube_el.get('multires', '') if cube_el is not None else ''
        stereo = image_el.get('stereo', 'false') if image_el is not None else 'false'
        stereolabels = image_el.get('stereolabels', '1|2') if image_el is not None else '1|2'

        # Parse multires: first value is tilesize, rest are level sizes
        multires = [int(x) for x in multires_str.split(',')] if multires_str else []

        scenes.append({
            'name': name,
            'title': title,
            'thumburl': thumburl,
            'preview_url': preview_url,
            'cube_url_template': cube_url_template,
            'multires': multires,
            'stereo': stereo == 'true',
            'stereolabels': stereolabels.split('|'),
        })

    print(f"Found {len(scenes)} Mastcam-Z scenes")
    return scenes


# ── Phase 1: Previews & Thumbnails ──────────────────────────────────────

def phase1_download_previews(scenes):
    """Download preview.jpg and thumb.jpg for each scene."""
    print("\n" + "=" * 60)
    print("PHASE 1: Downloading previews and thumbnails")
    print("=" * 60)

    success = 0
    for i, scene in enumerate(scenes, 1):
        safe_name = scene['title'].replace('/', '_').replace(' ', '_')
        print(f"[{i}/{len(scenes)}] {scene['title']}")

        # Thumbnail
        if scene['thumburl']:
            thumb_url = f"{BASE_URL}/{scene['thumburl']}"
            thumb_path = PREVIEW_DIR / f"{safe_name}_thumb.jpg"
            if fetch_url(thumb_url, thumb_path):
                pass  # ok
            else:
                print(f"  ✗ thumb failed")

        # Preview
        if scene['preview_url']:
            preview_url = f"{BASE_URL}/{scene['preview_url']}"
            preview_path = PREVIEW_DIR / f"{safe_name}_preview.jpg"
            if fetch_url(preview_url, preview_path):
                success += 1
            else:
                print(f"  ✗ preview failed")

        time.sleep(DELAY)

    print(f"\nPhase 1 complete: {success}/{len(scenes)} previews downloaded")
    print(f"Saved to: {PREVIEW_DIR}")


# ── Phase 2: Full-resolution tiles ──────────────────────────────────────

def resolve_tile_url(template, stereo_label, face, level, row, col):
    """Resolve krpano cube tile URL template."""
    url = template
    url = url.replace('%t', stereo_label)
    url = url.replace('%s', face)
    url = url.replace('%l', str(level))
    url = url.replace('%0v', f"{row:02d}")
    url = url.replace('%0h', f"{col:02d}")
    url = url.replace('%v', str(row))
    url = url.replace('%h', str(col))
    return url


def get_tiles_for_level(multires, level):
    """Calculate number of tiles (rows, cols) for a given level.
    multires[0] = tile_size, multires[1..] = face sizes per level.
    Level is 1-based.
    """
    tile_size = multires[0]
    face_size = multires[level]  # level 1 -> index 1
    ntiles = math.ceil(face_size / tile_size)
    return ntiles, ntiles  # square cube faces


def download_scene_tiles(scene, level=None):
    """Download all tiles for a scene at the highest resolution level."""
    if not scene['cube_url_template'] or not scene['multires']:
        print(f"  Skipping (no cube data): {scene['title']}")
        return None

    multires = scene['multires']
    if level is None:
        level = len(multires) - 1  # highest resolution (last entry after tilesize)

    tile_size = multires[0]
    face_size = multires[level]
    nrows, ncols = get_tiles_for_level(multires, level)
    stereo_label = STEREO_LABEL

    safe_name = scene['title'].replace('/', '_').replace(' ', '_')
    scene_tile_dir = TILES_DIR / safe_name / f"l{level}"

    total_tiles = len(FACES) * nrows * ncols
    print(f"  Level {level}: {face_size}px per face, {nrows}x{ncols} tiles/face, {total_tiles} total tiles")

    downloaded = 0
    failed = 0

    def download_tile(face, row, col):
        rel_url = resolve_tile_url(
            scene['cube_url_template'], stereo_label, face, level, row, col
        )
        url = f"{BASE_URL}/{rel_url}"
        dest = scene_tile_dir / face / f"{face}_{row:02d}_{col:02d}.jpg"
        return fetch_url(url, dest), face, row, col

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for face in FACES:
            for row in range(1, nrows + 1):
                for col in range(1, ncols + 1):
                    futures.append(executor.submit(download_tile, face, row, col))

        for future in as_completed(futures):
            ok, face, row, col = future.result()
            if ok:
                downloaded += 1
            else:
                failed += 1

    print(f"  Downloaded: {downloaded}, Failed: {failed}")

    return {
        'safe_name': safe_name,
        'tile_dir': scene_tile_dir,
        'level': level,
        'tile_size': tile_size,
        'face_size': face_size,
        'nrows': nrows,
        'ncols': ncols,
    }


def stitch_cube_face(tile_info, face):
    """Stitch tiles for one cube face into a single image."""
    tile_dir = tile_info['tile_dir'] / face
    tile_size = tile_info['tile_size']
    face_size = tile_info['face_size']
    nrows = tile_info['nrows']
    ncols = tile_info['ncols']

    face_img = Image.new('RGB', (face_size, face_size))

    for row in range(1, nrows + 1):
        for col in range(1, ncols + 1):
            tile_path = tile_dir / f"{face}_{row:02d}_{col:02d}.jpg"
            if tile_path.exists():
                tile = Image.open(tile_path)
                x = (col - 1) * tile_size
                y = (row - 1) * tile_size
                face_img.paste(tile, (x, y))

    # Crop to exact face size (last tiles may be padded)
    face_img = face_img.crop((0, 0, face_size, face_size))
    return face_img


def cube_to_equirectangular_fast(face_images, output_width=None):
    """Vectorized cube-to-equirectangular conversion using numpy."""
    face_size = face_images['f'].size[0]
    if output_width is None:
        output_width = face_size * 4
    output_height = output_width // 2

    faces = {}
    for key in FACES:
        faces[key] = np.array(face_images[key], dtype=np.uint8)

    # Create coordinate grids
    y_coords = np.arange(output_height)
    x_coords = np.arange(output_width)
    xx, yy = np.meshgrid(x_coords, y_coords)

    lat = (0.5 - yy / output_height) * np.pi
    lon = (xx / output_width - 0.5) * 2 * np.pi

    # Spherical to cartesian
    dx = np.cos(lat) * np.sin(lon)
    dy = np.sin(lat)
    dz = np.cos(lat) * np.cos(lon)

    abs_x = np.abs(dx)
    abs_y = np.abs(dy)
    abs_z = np.abs(dz)

    output = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Process each face
    face_map = {
        'f': (abs_z >= abs_x) & (abs_z >= abs_y) & (dz > 0),
        'b': (abs_z >= abs_x) & (abs_z >= abs_y) & (dz <= 0),
        'r': (abs_x > abs_z) & (abs_x >= abs_y) & (dx > 0),
        'l': (abs_x > abs_z) & (abs_x >= abs_y) & (dx <= 0),
        'u': (abs_y > abs_x) & (abs_y > abs_z) & (dy > 0),
        'd': (abs_y > abs_x) & (abs_y > abs_z) & (dy <= 0),
    }

    uv_map = {
        'f': lambda: (dx / abs_z, -dy / abs_z),
        'b': lambda: (-dx / abs_z, -dy / abs_z),
        'r': lambda: (-dz / abs_x, -dy / abs_x),
        'l': lambda: (dz / abs_x, -dy / abs_x),
        'u': lambda: (dx / abs_y, dz / abs_y),
        'd': lambda: (dx / abs_y, -dz / abs_y),
    }

    for face_key in FACES:
        mask = face_map[face_key]
        if not np.any(mask):
            continue

        u, v = uv_map[face_key]()
        px = ((u[mask] + 1) / 2 * (face_size - 1)).astype(np.int32)
        py = ((v[mask] + 1) / 2 * (face_size - 1)).astype(np.int32)
        px = np.clip(px, 0, face_size - 1)
        py = np.clip(py, 0, face_size - 1)

        output[mask] = faces[face_key][py, px]

    return Image.fromarray(output)


def phase2_download_full(scenes):
    """Download full-resolution tiles and stitch panoramas."""
    print("\n" + "=" * 60)
    print("PHASE 2: Downloading full-resolution tiles & stitching")
    print("=" * 60)

    if not HAS_IMAGING:
        print("WARNING: Pillow/numpy not installed. Downloading tiles only (no stitching).")
        print("Install with: pip install Pillow numpy")

    for i, scene in enumerate(scenes, 1):
        title = scene['title']
        print(f"\n[{i}/{len(scenes)}] {title}")

        # Download tiles at highest resolution
        tile_info = download_scene_tiles(scene)
        if tile_info is None:
            continue

        if not HAS_IMAGING:
            continue

        # Stitch cube faces
        safe_name = tile_info['safe_name']
        print(f"  Stitching cube faces...")
        face_images = {}
        all_ok = True
        for face in FACES:
            face_img = stitch_cube_face(tile_info, face)
            if face_img:
                face_images[face] = face_img
                # Save individual face
                face_path = FULL_DIR / f"{safe_name}_face_{face}.jpg"
                face_img.save(face_path, 'JPEG', quality=95)
            else:
                print(f"  ✗ Failed to stitch face {face}")
                all_ok = False

        if all_ok and len(face_images) == 6:
            # Convert to equirectangular
            print(f"  Converting to equirectangular panorama...")
            equirect = cube_to_equirectangular_fast(face_images)
            output_path = FULL_DIR / f"{safe_name}_equirectangular.jpg"
            equirect.save(output_path, 'JPEG', quality=95)
            print(f"  Saved: {output_path}")
            print(f"  Size: {equirect.size[0]}x{equirect.size[1]}")
        else:
            print(f"  ✗ Skipping equirectangular conversion (missing faces)")

    print(f"\nPhase 2 complete!")
    print(f"Saved to: {FULL_DIR}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    scenes = parse_tour_xml()

    if not scenes:
        print("No Mastcam scenes found!")
        sys.exit(1)

    # Print scene list
    print("\nMastcam-Z Scenes:")
    for i, s in enumerate(scenes, 1):
        mr = s['multires']
        max_res = mr[-1] if mr else '?'
        print(f"  {i:2d}. {s['title']} (max face: {max_res}px)")

    # Phase 1
    phase1_download_previews(scenes)

    # Phase 2
    phase2_download_full(scenes)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print(f"Previews: {PREVIEW_DIR}")
    print(f"Full panoramas: {FULL_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
