"""HiRISE DTM and orthoimage loading with coordinate utilities.

Handles PDS IMG and GeoTIFF formats, extracts CRS metadata,
and provides coordinate conversion between lon/lat and DTM pixel space.
"""

import struct
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.transform import rowcol, xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _parse_pds3_label(label_path: Path) -> dict:
    """Parse a PDS3 .lbl file for HiRISE DTM metadata."""
    meta = {}
    with open(label_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip().upper()
            val = val.strip().strip('"').strip("'")

            # Remove units in angle brackets
            if "<" in val:
                val = val[:val.index("<")].strip()

            try:
                if key == "LINES":
                    meta["lines"] = int(val)
                elif key == "LINE_SAMPLES":
                    meta["samples"] = int(val)
                elif key == "MAP_SCALE":
                    meta["map_scale_m"] = float(val)
                elif key == "CENTER_LATITUDE":
                    meta["center_lat"] = float(val)
                elif key == "CENTER_LONGITUDE":
                    meta["center_lon"] = float(val)
                elif key == "LINE_PROJECTION_OFFSET":
                    meta["line_offset"] = float(val)
                elif key == "SAMPLE_PROJECTION_OFFSET":
                    meta["sample_offset"] = float(val)
                elif key == "MAP_PROJECTION_TYPE":
                    meta["projection"] = val
                elif key == "RECORD_BYTES":
                    meta["record_bytes"] = int(val)
                elif key == "LABEL_RECORDS":
                    meta["label_records"] = int(val)
                elif key in ("MAXIMUM_LATITUDE", "NORTHERNMOST_LATITUDE"):
                    meta["max_lat"] = float(val)
                elif key in ("MINIMUM_LATITUDE", "SOUTHERNMOST_LATITUDE"):
                    meta["min_lat"] = float(val)
                elif key == "WESTERNMOST_LONGITUDE":
                    meta["min_lon"] = float(val)
                elif key == "EASTERNMOST_LONGITUDE":
                    meta["max_lon"] = float(val)
                elif key == "SAMPLE_TYPE":
                    meta["sample_type"] = val
                elif key == "SAMPLE_BITS":
                    meta["sample_bits"] = int(val)
                elif key == "A_AXIS_RADIUS":
                    meta["a_radius"] = float(val)
                elif key == "C_AXIS_RADIUS":
                    meta["c_radius"] = float(val)
                elif key == "^IMAGE":
                    meta["image_start_record"] = int(val)
                elif key == "VALID_MINIMUM":
                    meta["valid_min"] = float(val)
                elif key == "VALID_MAXIMUM":
                    meta["valid_max"] = float(val)
            except (ValueError, IndexError):
                pass

            if line == "END":
                break

    return meta


class HiRISEDTM:
    """HiRISE DTM reader supporting GeoTIFF and PDS IMG formats."""

    def __init__(self, data_path: Path, label_path: Path = None):
        self.data_path = Path(data_path)
        self.label_path = Path(label_path) if label_path else None
        self.meta = {}
        self._dataset = None
        self._elevation = None

        suffix = self.data_path.suffix.lower()

        if suffix in (".tif", ".tiff") and HAS_RASTERIO:
            self._load_geotiff()
        elif suffix == ".img":
            self._load_pds_img()
        else:
            # Try rasterio for any format it supports
            if HAS_RASTERIO:
                self._load_geotiff()
            else:
                raise ValueError(
                    f"Unsupported format: {suffix}. Install rasterio for GeoTIFF support."
                )

    def _load_geotiff(self):
        """Load a GeoTIFF DTM using rasterio."""
        self._dataset = rasterio.open(self.data_path)
        self.meta = {
            "lines": self._dataset.height,
            "samples": self._dataset.width,
            "crs": str(self._dataset.crs),
            "transform": self._dataset.transform,
            "bounds": self._dataset.bounds,
            "nodata": self._dataset.nodata,
        }
        print(f"  DTM loaded (GeoTIFF): {self._dataset.width}x{self._dataset.height}")
        print(f"  CRS: {self._dataset.crs}")
        print(f"  Bounds: {self._dataset.bounds}")

    def _load_pds_img(self):
        """Load a PDS3 IMG DTM."""
        # Find label
        if self.label_path and self.label_path.exists():
            self.meta = _parse_pds3_label(self.label_path)
        else:
            # Try detached label
            lbl = self.data_path.with_suffix(".lbl")
            if lbl.exists():
                self.meta = _parse_pds3_label(lbl)
            else:
                # Try embedded label (read from IMG file itself)
                self.meta = _parse_pds3_label(self.data_path)

        lines = self.meta.get("lines", 0)
        samples = self.meta.get("samples", 0)
        if lines == 0 or samples == 0:
            raise ValueError("Cannot determine DTM dimensions from label")

        # Calculate data offset
        record_bytes = self.meta.get("record_bytes", 0)
        # ^IMAGE = N means data starts at record N (1-based)
        image_start = self.meta.get("image_start_record", 2)
        offset = record_bytes * (image_start - 1)

        # Read elevation data
        bits = self.meta.get("sample_bits", 32)
        dtype = np.float32 if bits == 32 else np.float64
        sample_type = self.meta.get("sample_type", "IEEE_REAL")

        if "LSB" in sample_type or "PC" in sample_type:
            dtype = np.dtype(dtype).newbyteorder("<")
        else:
            dtype = np.dtype(dtype).newbyteorder(">")

        self._elevation = np.fromfile(
            str(self.data_path), dtype=dtype, offset=offset, count=lines * samples
        ).reshape(lines, samples)

        # Mark nodata using valid range from label
        valid_min = self.meta.get("valid_min")
        valid_max = self.meta.get("valid_max")
        if valid_min is not None and valid_max is not None:
            self._elevation[(self._elevation < valid_min - 100) |
                            (self._elevation > valid_max + 100)] = np.nan
        else:
            nodata = -32768.0
            self._elevation[self._elevation <= nodata] = np.nan

        print(f"  DTM loaded (PDS IMG): {samples}x{lines}")
        print(f"  Elevation range: [{np.nanmin(self._elevation):.1f}, "
              f"{np.nanmax(self._elevation):.1f}] m")

    @property
    def bounds(self) -> dict:
        """Return bounding box as dict with min_lon, max_lon, min_lat, max_lat."""
        if self._dataset:
            b = self._dataset.bounds
            return {
                "min_lon": b.left,
                "max_lon": b.right,
                "min_lat": b.bottom,
                "max_lat": b.top,
            }
        return {
            "min_lon": self.meta.get("min_lon", 0),
            "max_lon": self.meta.get("max_lon", 0),
            "min_lat": self.meta.get("min_lat", 0),
            "max_lat": self.meta.get("max_lat", 0),
        }

    @property
    def resolution_m(self) -> float:
        """Ground resolution in meters/pixel."""
        if self._dataset:
            res = abs(self._dataset.transform.a)
            # If CRS is geographic (degrees), convert to meters
            if self._dataset.crs and self._dataset.crs.is_geographic:
                mars_r_m = 3396190.0
                res = res * np.radians(1) * mars_r_m  # degrees -> meters
            return res
        return self.meta.get("map_scale_m", 1.0)

    def lonlat_to_pixel(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert lon/lat arrays to DTM pixel coordinates (row, col).

        Args:
            lon: longitude in degrees (same convention as DTM)
            lat: latitude in degrees

        Returns:
            rows, cols: integer pixel coordinates (may be out of bounds)
        """
        if self._dataset:
            # Use rasterio's transform
            rows, cols = rowcol(self._dataset.transform, lon, lat)
            return np.asarray(rows), np.asarray(cols)

        # PDS3 equirectangular projection
        scale = self.meta.get("map_scale_m", 1.0)
        center_lon = self.meta.get("center_lon", 0)
        center_lat = self.meta.get("center_lat", 0)
        sample_offset = self.meta.get("sample_offset", 0)
        line_offset = self.meta.get("line_offset", 0)

        # Mars radius
        mars_r = self.meta.get("a_radius", 3396190.0)
        if mars_r > 10000:
            mars_r_m = mars_r
        else:
            mars_r_m = mars_r * 1000  # km -> m

        deg_per_pixel_lon = np.degrees(scale / (mars_r_m * np.cos(np.radians(center_lat))))
        deg_per_pixel_lat = np.degrees(scale / mars_r_m)

        cols = sample_offset + (lon - center_lon) / deg_per_pixel_lon
        rows = line_offset - (lat - center_lat) / deg_per_pixel_lat

        return rows.astype(np.int32), cols.astype(np.int32)

    def get_elevation(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Get elevation values at given pixel coordinates.

        Out-of-bounds pixels return NaN.
        """
        if self._dataset:
            # Use windowed reading for memory efficiency
            h = self._dataset.height
            w = self._dataset.width

            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            result = np.full(rows.shape, np.nan)

            if np.any(valid):
                vr = rows[valid]
                vc = cols[valid]

                # Read a window covering all valid pixels
                r_min, r_max = int(vr.min()), int(vr.max())
                c_min, c_max = int(vc.min()), int(vc.max())

                window = rasterio.windows.Window(
                    c_min, r_min, c_max - c_min + 1, r_max - r_min + 1
                )
                data = self._dataset.read(1, window=window)

                # Apply nodata
                nodata = self._dataset.nodata
                if nodata is not None:
                    data = data.astype(np.float64)
                    data[data == nodata] = np.nan

                local_r = vr - r_min
                local_c = vc - c_min
                result[valid] = data[local_r, local_c]

            return result

        # PDS3 in-memory array
        h, w = self._elevation.shape
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        result = np.full(rows.shape, np.nan)
        if np.any(valid):
            result[valid] = self._elevation[rows[valid], cols[valid]]
        return result

    def read_full(self) -> np.ndarray:
        """Read the full elevation array. Use with caution for large DTMs."""
        if self._elevation is not None:
            return self._elevation
        if self._dataset:
            data = self._dataset.read(1).astype(np.float64)
            nodata = self._dataset.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            return data
        raise RuntimeError("No elevation data loaded")

    def close(self):
        """Release resources."""
        if self._dataset:
            self._dataset.close()
            self._dataset = None
