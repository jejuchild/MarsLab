#!/usr/bin/env python3
"""
Generate midlat_metadata.json from hirise_ice.db with title→class regex mapping.

Reads all rows from the database and classifies each image based on its title
into one of: LDA, CCF, LVF, GLF, or UNLABELED.

Usage:
    python scripts/landform_pipeline/generate_metadata.py
    python scripts/landform_pipeline/generate_metadata.py --output Data/HiRISE/midlat_metadata.json
"""

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "backend" / "data" / "hirise_ice.db"
OUTPUT_PATH = PROJECT_ROOT / "Data" / "HiRISE" / "midlat_metadata.json"

# Classification rules: order matters — first match wins
# Each rule: (class_label, compiled_regex)
CLASSIFICATION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("LDA", re.compile(r"lobate\s+debris\s+apron", re.IGNORECASE)),
    ("CCF", re.compile(r"concentric\s+crater\s+fill", re.IGNORECASE)),
    ("LVF", re.compile(r"lineated\s+valley\s+(fill|flow)", re.IGNORECASE)),
    # GLF broad patterns — glacier, glacial, viscous flow, flow+lobate
    ("GLF", re.compile(
        r"glacier[\-\s]like\s+form"
        r"|glacial\s+feature"
        r"|glacial\s+flow"
        r"|glacial\s+landform"
        r"|glacier"
        r"|viscous\s+flow\s+feature"
        r"|viscous\s+flow"
        r"|(?:flow\s+feature.*lobate)"
        r"|(?:lobate.*flow\s+feature)",
        re.IGNORECASE,
    )),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metadata JSON from hirise_ice.db with class labels."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Path to hirise_ice.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output path for metadata JSON",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        default=True,
        help="Print class distribution stats",
    )
    return parser.parse_args()


def classify_title(title: str) -> str:
    """Classify an image title into LDA/CCF/LVF/GLF/UNLABELED."""
    if not title:
        return "UNLABELED"
    for class_label, pattern in CLASSIFICATION_RULES:
        if pattern.search(title):
            return class_label
    return "UNLABELED"


def main() -> None:
    args = parse_args()
    t0 = time.time()

    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT image_id, title, lat, lon, quickview_url FROM hirise_ice "
        "ORDER BY image_id"
    ).fetchall()
    conn.close()

    records: list[dict] = []
    class_counts: dict[str, int] = {}

    for row in rows:
        image_id = row["image_id"]
        title = row["title"] or ""
        lat = row["lat"]
        lon = row["lon"]

        class_label = classify_title(title)
        class_counts[class_label] = class_counts.get(class_label, 0) + 1

        records.append({
            "image_id": image_id,
            "title": title,
            "class": class_label,
            "lat": lat,
            "lon": lon,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    elapsed = time.time() - t0

    print(f"Generated metadata for {len(records)} images")
    print(f"Output: {args.output}")
    print(f"Time: {elapsed:.2f}s")
    print("\nClass distribution:")
    for cls in ["LDA", "CCF", "LVF", "GLF", "UNLABELED"]:
        count = class_counts.get(cls, 0)
        pct = 100.0 * count / len(records) if records else 0
        print(f"  {cls}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
