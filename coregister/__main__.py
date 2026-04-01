"""CLI entry point for Mastcam-Z / HiRISE coregistration pipeline.

Usage:
    python -m coregister --sol 100
    python -m coregister --sol 100 --hirise DTEEC_045994_1985
    python -m coregister --setup-spice --sol 100
    python -m coregister --list-products --sol 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import OUTPUT_DIR, PDS_CACHE


def parse_args():
    p = argparse.ArgumentParser(
        prog="coregister",
        description="Mastcam-Z / HiRISE coregistration using PDS XYZ + SPICE",
    )
    p.add_argument("--sol", type=int, help="Mars 2020 sol number")
    p.add_argument(
        "--product-id", type=str, default=None,
        help="Specific Mastcam-Z product ID (overrides --sol search)",
    )
    p.add_argument(
        "--hirise", type=str, default=None,
        help="HiRISE DTM product ID or local file path",
    )
    p.add_argument(
        "--setup-spice", action="store_true",
        help="Download SPICE kernels and exit",
    )
    p.add_argument(
        "--list-products", action="store_true",
        help="List available Mastcam-Z products for a sol and exit",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output GeoTIFF path (default: auto-named in output dir)",
    )
    p.add_argument(
        "--no-dtm", action="store_true",
        help="Skip DTM draping, produce georeferenced overlay only",
    )
    return p.parse_args()


def cmd_list_products(sol: int):
    """List available Mastcam-Z products for a given sol."""
    from .pds_fetch import search_mastcamz_products, list_available_sols

    print(f"\nSearching Mastcam-Z products for sol {sol}...")

    for ptype in ["XYZ", "RAS"]:
        products = search_mastcamz_products(sol, product_type=ptype)
        if products:
            print(f"\n  {ptype} products ({len(products)}):")
            for p in products[:10]:
                print(f"    {p['product_id']}")
        else:
            print(f"\n  {ptype}: none found for sol {sol}")

    # Show nearby available sols if nothing found
    xyz = search_mastcamz_products(sol, "XYZ")
    if not xyz:
        print("\n  Available sols with stereo data:")
        sols = list_available_sols()
        nearby = [s for s in sols if abs(s - sol) <= 50]
        if nearby:
            print(f"    Near sol {sol}: {nearby}")
        else:
            print(f"    First 20: {sols[:20]}")


def cmd_setup_spice(sol: int):
    """Download SPICE kernels only."""
    from .spice_setup import init_spice, cleanup_spice

    print(f"\nSetting up SPICE kernels for sol {sol}...")
    mk_path = init_spice(sol)
    cleanup_spice()
    print(f"\nDone. Meta-kernel: {mk_path}")


def cmd_coregister(args):
    """Full coregistration pipeline."""
    from .pds_fetch import (
        search_mastcamz_products,
        download_mastcamz_product,
        search_hirise_dtm,
        download_hirise_dtm,
    )
    from .spice_setup import init_spice, cleanup_spice
    from .mastcam_xyz import process_mastcamz_xyz
    from .hirise_dtm import HiRISEDTM
    from .drape import drape_on_dtm, drape_simple_overlay, load_mastcamz_texture

    sol = args.sol

    # ── Step 1: Find and download Mastcam-Z XYZ product ──────────────
    print(f"\n{'='*60}")
    print(f"Step 1: Mastcam-Z XYZ product (sol {sol})")
    print(f"{'='*60}")

    xyz_products = search_mastcamz_products(sol, product_type="XYZ")
    if not xyz_products:
        print(f"No XYZ products found for sol {sol}.")
        print("Trying nearby sols...")
        for delta in range(1, 11):
            for s in [sol - delta, sol + delta]:
                xyz_products = search_mastcamz_products(s, product_type="XYZ")
                if xyz_products:
                    print(f"Found XYZ products at sol {s}")
                    sol = s
                    break
            if xyz_products:
                break

    if not xyz_products:
        print("ERROR: No Mastcam-Z XYZ products found. Use --list-products to check availability.")
        sys.exit(1)

    # Use specified product or first available
    if args.product_id:
        xyz_product = next(
            (p for p in xyz_products if args.product_id in p["product_id"]),
            None,
        )
        if not xyz_product:
            print(f"Product {args.product_id} not found. Available:")
            for p in xyz_products:
                print(f"  {p['product_id']}")
            sys.exit(1)
    else:
        xyz_product = xyz_products[0]

    print(f"Selected: {xyz_product['product_id']}")

    # Download
    xyz_files = download_mastcamz_product(xyz_product)
    if "data_path" not in xyz_files:
        print("ERROR: Failed to download XYZ data file")
        sys.exit(1)

    # Also get the corresponding RAS (texture) product
    print("\nSearching for matching RAS (texture) product...")
    ras_products = search_mastcamz_products(sol, product_type="RAS")
    ras_path = None
    if ras_products:
        # Try to find matching RAS for the same observation
        xyz_pid = xyz_product["product_id"]
        # Extract observation ID pattern (e.g., ZL0_0100_...)
        base_pattern = xyz_pid.rsplit("_", 2)[0] if "_" in xyz_pid else xyz_pid
        matching_ras = [p for p in ras_products if base_pattern in p["product_id"]]
        ras_to_download = matching_ras[0] if matching_ras else ras_products[0]

        ras_files = download_mastcamz_product(ras_to_download)
        ras_path = ras_files.get("data_path")
        if ras_path:
            print(f"Texture: {ras_to_download['product_id']}")

    # ── Step 2: Initialize SPICE ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 2: SPICE initialization")
    print(f"{'='*60}")

    init_spice(sol)

    # ── Step 3: Process XYZ → lon/lat ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 3: XYZ coordinate transformation")
    print(f"{'='*60}")

    result = process_mastcamz_xyz(
        data_path=xyz_files["data_path"],
        label_path=xyz_files.get("label_path"),
    )

    lon = result["lon"]
    lat = result["lat"]

    # Load texture
    if ras_path and ras_path.exists():
        texture = load_mastcamz_texture(ras_path)
        # Ensure texture matches XYZ dimensions
        if texture.shape[:2] != lon.shape:
            from PIL import Image as PILImage
            texture = np.array(
                PILImage.fromarray(texture).resize((lon.shape[1], lon.shape[0]))
            )
    else:
        # Use XYZ magnitude as grayscale fallback
        print("  No texture found, using XYZ magnitude as grayscale")
        mag = np.sqrt(np.nansum(result["xyz_site"] ** 2, axis=-1))
        denom = np.nanmax(mag) - np.nanmin(mag)
        if denom == 0:
            denom = 1.0
        mag_norm = np.clip((mag - np.nanmin(mag)) / denom * 255, 0, 255).astype(np.uint8)
        texture = np.stack([mag_norm] * 3, axis=-1)

    # ── Step 4: HiRISE DTM ──────────────────────────────────────────
    if not args.no_dtm:
        print(f"\n{'='*60}")
        print(f"Step 4: HiRISE DTM")
        print(f"{'='*60}")

        dtm = None
        if args.hirise:
            hirise_path = Path(args.hirise)
            if hirise_path.exists():
                # Local file
                dtm = HiRISEDTM(hirise_path)
            else:
                # Search PDS by product ID
                print(f"Searching for HiRISE DTM: {args.hirise}")
                hirise_products = search_hirise_dtm()
                matching = [p for p in hirise_products if args.hirise in p["product_id"]]
                if matching:
                    files = download_hirise_dtm(matching[0])
                    if "data_path" in files:
                        dtm = HiRISEDTM(files["data_path"], files.get("label_path"))
                else:
                    print(f"HiRISE DTM {args.hirise} not found in PDS")
        else:
            # Auto-search for Jezero DTMs
            print("Auto-searching HiRISE DTMs for Jezero crater...")
            hirise_products = search_hirise_dtm()
            if hirise_products:
                print(f"Found {len(hirise_products)} DTMs. Using first match.")
                files = download_hirise_dtm(hirise_products[0])
                if "data_path" in files:
                    dtm = HiRISEDTM(files["data_path"], files.get("label_path"))
            else:
                print("No HiRISE DTMs found for Jezero area")

        if dtm is None:
            print("WARNING: No DTM available, falling back to simple overlay")
            args.no_dtm = True

    # ── Step 5: Drape and output ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 5: Draping and output")
    print(f"{'='*60}")

    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"sol{sol:04d}_coregistered.tif"

    if args.no_dtm:
        result_path = drape_simple_overlay(lon, lat, texture, output_path)
    else:
        result_path = drape_on_dtm(lon, lat, texture, dtm, output_path)
        dtm.close()

    # Cleanup
    cleanup_spice()

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"Output: {result_path}")
    print(f"{'='*60}")

    return result_path


def main():
    args = parse_args()

    if args.sol is None and not args.product_id:
        print("ERROR: --sol or --product-id required")
        sys.exit(1)

    if args.sol is None:
        args.sol = 100  # default for SPICE setup

    if args.list_products:
        cmd_list_products(args.sol)
    elif args.setup_spice:
        cmd_setup_spice(args.sol)
    else:
        cmd_coregister(args)


if __name__ == "__main__":
    main()
