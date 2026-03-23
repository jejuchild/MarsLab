#!/usr/bin/env python3
"""
Cleanup residual aria2 temp files and 0-byte failed downloads.

Usage:
    python cleanup_aria2.py          # Dry run (show what would be deleted)
    python cleanup_aria2.py --delete  # Actually delete files
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIRS = [
    ("crism", BASE_DIR / "crism_data"),
    ("hirise", BASE_DIR / "hirise_data"),
    ("sharad", BASE_DIR / "sharad_data"),
    ("sharad_highres", BASE_DIR / "sharad_highres"),
]


def scan_residuals():
    residuals = []
    for label, data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for f in data_dir.rglob("*"):
            if not f.is_file():
                continue
            size = f.stat().st_size
            is_aria2 = f.suffix == ".aria2"
            is_empty = size == 0
            if is_aria2 or is_empty:
                residuals.append({
                    "path": f,
                    "size": size,
                    "reason": "aria2_temp" if is_aria2 else "empty_file",
                    "directory": label,
                })
    return residuals


def format_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Cleanup aria2 residual files")
    parser.add_argument("--delete", action="store_true", help="Actually delete files (default: dry run)")
    args = parser.parse_args()

    residuals = scan_residuals()

    if not residuals:
        print("No residual files found. All clean!")
        return

    total_size = sum(r["size"] for r in residuals)
    aria2_count = sum(1 for r in residuals if r["reason"] == "aria2_temp")
    empty_count = sum(1 for r in residuals if r["reason"] == "empty_file")

    print(f"Found {len(residuals)} residual files ({format_size(total_size)}):")
    print(f"  .aria2 temp files: {aria2_count}")
    print(f"  Empty (0-byte) files: {empty_count}")
    print()

    for r in residuals:
        tag = "[ARIA2]" if r["reason"] == "aria2_temp" else "[EMPTY]"
        print(f"  {tag} {r['path']} ({format_size(r['size'])})")

    print()

    if args.delete:
        deleted = 0
        errors = 0
        for r in residuals:
            try:
                r["path"].unlink()
                deleted += 1
            except Exception as e:
                print(f"  ERROR: {r['path']}: {e}", file=sys.stderr)
                errors += 1
        print(f"Deleted {deleted} files, freed {format_size(total_size)}")
        if errors:
            print(f"  {errors} errors occurred", file=sys.stderr)
    else:
        print("Dry run — use --delete to remove these files")


if __name__ == "__main__":
    main()
