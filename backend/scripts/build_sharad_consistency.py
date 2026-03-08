#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np


if __package__ is None or __package__ == "":
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

_rdr_loader = importlib.import_module("analysis.sharad_rdr_pipeline.rdr_loader")
_surface_power = importlib.import_module("analysis.sharad_rdr_pipeline.surface_power")
_dielectric = importlib.import_module("analysis.sharad_rdr_pipeline.dielectric")
_gridder = importlib.import_module("analysis.sharad_rdr_pipeline.gridder")

list_available_products = _rdr_loader.list_available_products
load_track_data = _rdr_loader.load_track_data
SurfacePowerResult = _surface_power.SurfacePowerResult
build_global_power_stats = _surface_power.build_global_power_stats
compute_surface_power = _surface_power.compute_surface_power
score_surface_consistency = _surface_power.score_surface_consistency
DielectricResult = _dielectric.DielectricResult
compute_dielectric = _dielectric.compute_dielectric
grid_measurements = _gridder.grid_measurements
write_geotiff = _gridder.write_geotiff


def _concat(parts: list[np.ndarray], dtype=np.float64) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SHARAD RDR radar consistency GeoTIFFs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (backend_dir / "data" / "swim")
    output_dir.mkdir(parents=True, exist_ok=True)

    products = list_available_products()
    print(f"Found {len(products)} SHARAD RDR products")
    if not products:
        return 0

    surface_results: list[SurfacePowerResult] = []
    dielectric_results: list[DielectricResult] = []
    n_errors = 0

    for i, pid in enumerate(products, start=1):
        print(f"[{i}/{len(products)}] {pid}")
        try:
            track = load_track_data(pid)
            if track is None:
                if args.verbose:
                    print(f"  skip: track load returned None for {pid}")
                continue

            surface = compute_surface_power(track)
            if surface is not None:
                surface_results.append(surface)

            dielectric = compute_dielectric(track)
            if dielectric is not None:
                dielectric_results.append(dielectric)

        except Exception as exc:
            n_errors += 1
            print(f"  error: {pid}: {exc}")

    print(
        "Track summary: "
        f"surface={len(surface_results)}, dielectric={len(dielectric_results)}, errors={n_errors}"
    )

    if surface_results:
        mu, sigma = build_global_power_stats(surface_results)
        print(f"Global power stats: mu={mu:.2f} dB, sigma={sigma:.2f} dB")
        s_lats: list[np.ndarray] = []
        s_lons: list[np.ndarray] = []
        s_vals: list[np.ndarray] = []
        s_w: list[np.ndarray] = []

        for sr in surface_results:
            lat, lon, consistency, snr = score_surface_consistency(sr, mu, sigma)
            if consistency.size == 0:
                continue
            s_lats.append(lat)
            s_lons.append(lon)
            s_vals.append(consistency)
            s_w.append(np.asarray(snr, dtype=np.float64))

        surf_grid = grid_measurements(_concat(s_lats), _concat(s_lons), _concat(s_vals), _concat(s_w))
        surf_out = output_dir / "radar_surface_rdr.tif"
        write_geotiff(surf_grid, str(surf_out), description="SHARAD RDR radar surface consistency")
        print(f"Wrote {surf_out}")
    else:
        print("No surface measurements available; skipping radar_surface_rdr.tif")

    if dielectric_results:
        d_lats_15: list[np.ndarray] = []
        d_lons_15: list[np.ndarray] = []
        d_vals_15: list[np.ndarray] = []
        d_w_15: list[np.ndarray] = []

        d_lats_5p: list[np.ndarray] = []
        d_lons_5p: list[np.ndarray] = []
        d_vals_5p: list[np.ndarray] = []
        d_w_5p: list[np.ndarray] = []

        for dr in dielectric_results:
            depth = np.asarray(dr.depth_bin)
            shallow = depth == "1-5m"
            deep = depth == "5m-plus"

            if np.any(shallow):
                d_lats_15.append(dr.lat[shallow])
                d_lons_15.append(dr.lon[shallow])
                d_vals_15.append(dr.consistency[shallow])
                d_w_15.append(dr.snr[shallow])

            if np.any(deep):
                d_lats_5p.append(dr.lat[deep])
                d_lons_5p.append(dr.lon[deep])
                d_vals_5p.append(dr.consistency[deep])
                d_w_5p.append(dr.snr[deep])

        diel15_grid = grid_measurements(_concat(d_lats_15), _concat(d_lons_15), _concat(d_vals_15), _concat(d_w_15))
        diel15_out = output_dir / "radar_dielectric_rdr_1_5m.tif"
        write_geotiff(diel15_grid, str(diel15_out), description="SHARAD RDR radar dielectric consistency (1-5m)")
        print(f"Wrote {diel15_out}")

        diel5p_grid = grid_measurements(_concat(d_lats_5p), _concat(d_lons_5p), _concat(d_vals_5p), _concat(d_w_5p))
        diel5p_out = output_dir / "radar_dielectric_rdr_5m_plus.tif"
        write_geotiff(diel5p_grid, str(diel5p_out), description="SHARAD RDR radar dielectric consistency (5m-plus)")
        print(f"Wrote {diel5p_out}")
    else:
        print("No dielectric measurements available; skipping dielectric outputs")

    n_surface_traces = int(sum(sr.lat.size for sr in surface_results))
    n_dielectric_points = int(sum(dr.lat.size for dr in dielectric_results))
    print(
        "Measurement summary: "
        f"surface_traces={n_surface_traces}, dielectric_points={n_dielectric_points}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
