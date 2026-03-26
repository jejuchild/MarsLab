#!/usr/bin/env python3
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnnecessaryIsInstance=false, reportUnreachable=false
import json
import copy
from pathlib import Path
from typing import Any


BASE = Path("/disk1/cspark/MarsLab")
INDEX_PATH = BASE / "backend/hirise_data/index.geojson"
INDEX_BACKUP_PATH = BASE / "backend/hirise_data/index.geojson.backup"
QUICKVIEW_DIR = BASE / "backend/hirise_quickview"
METADATA_PATH = BASE / "Data/HiRISE/midlat_metadata.json"
BROWSE_DIR = BASE / "Data/HiRISE/midlat_browse"
RDR_INDEX_PATH = BASE / "Data/HiRISE/RDRCUMINDEX.TAB"


def parse_float(raw: str):
    value = raw.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def lon_360_to_180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def lon_any_to_360(lon: float) -> float:
    return lon % 360.0


def normalize_geometry_longitudes(geometry: dict[str, object]):
    if not isinstance(geometry, dict):
        return geometry

    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
        return {"type": "Point", "coordinates": [lon_360_to_180(coords[0]), coords[1]]}

    if gtype == "Polygon" and isinstance(coords, list):
        rings = []
        for ring in coords:
            rings.append([[lon_360_to_180(point[0]), point[1]] for point in ring])
        return {"type": "Polygon", "coordinates": rings}

    if gtype == "MultiPolygon" and isinstance(coords, list):
        out = []
        for poly in coords:
            rings = []
            for ring in poly:
                rings.append([[lon_360_to_180(point[0]), point[1]] for point in ring])
            out.append(rings)
        return {"type": "MultiPolygon", "coordinates": out}

    return geometry


def rectangle_footprint(center_lat: float, center_lon_360: float):
    # HiRISE swaths are long & narrow strips (~6km wide, ~10-20km long)
    half_lat = 0.1   # ~11 km along-track (was 0.025)
    half_lon = 0.03   # ~3.5 km cross-track at equator (was 0.05)
    west = lon_360_to_180(center_lon_360 - half_lon)
    east = lon_360_to_180(center_lon_360 + half_lon)
    south = max(-90.0, center_lat - half_lat)
    north = min(90.0, center_lat + half_lat)
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]


def parse_rdr_rows(observation_ids: set[str]):
    import csv

    records = {}
    with RDR_INDEX_PATH.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            # Quick check before expensive CSV parsing
            obs_id_raw = line[99:114].strip()
            if obs_id_raw not in observation_ids or obs_id_raw in records:
                continue

            fields = list(csv.reader([line]))[0]
            obs_id = fields[4].strip()
            if obs_id not in observation_ids or obs_id in records:
                continue

            # Corner coordinates (fields 46-53)
            c1_lat = parse_float(fields[46])
            c1_lon = parse_float(fields[47])
            c2_lat = parse_float(fields[48])
            c2_lon = parse_float(fields[49])
            c3_lat = parse_float(fields[50])
            c3_lon = parse_float(fields[51])
            c4_lat = parse_float(fields[52])
            c4_lon = parse_float(fields[53])

            center_lat = parse_float(fields[29])
            center_lon = parse_float(fields[30])

            # Full product dimensions (for quickview aspect-ratio correction)
            rdr_lines = parse_float(fields[17])
            rdr_samples = parse_float(fields[18])
            proj_center_lat = parse_float(fields[42])

            corners = [
                (c1_lon, c1_lat),
                (c2_lon, c2_lat),
                (c3_lon, c3_lat),
                (c4_lon, c4_lat),
            ]
            have_polygon = all(lon is not None and lat is not None for lon, lat in corners)

            if have_polygon:
                # Convert to -180..180
                converted = []
                for lon, lat in corners:
                    if lon is None or lat is None:
                        continue
                    converted.append([lon_360_to_180(lon), lat])

                # Detect antimeridian crossing: if any pair of corners
                # spans > 180° in longitude, shift negative lons by +360
                lons_180 = [c[0] for c in converted]
                if max(lons_180) - min(lons_180) > 180:
                    # Straddles antimeridian — shift to keep all corners together
                    converted = [
                        [lon + 360 if lon < 0 else lon, lat]
                        for lon, lat in converted
                    ]

                polygon = converted
                polygon.append(polygon[0])
            else:
                polygon = None

            records[obs_id] = {
                "center_lat": center_lat,
                "center_lon_360": center_lon,
                "polygon": polygon,
                "rdr_lines": int(rdr_lines) if rdr_lines else None,
                "rdr_samples": int(rdr_samples) if rdr_samples else None,
                "proj_center_lat": proj_center_lat,
            }

            if len(records) == len(observation_ids):
                break

    return records


def build_feature(
    image_id: str,
    title: str | None,
    metadata_lat: float | None,
    metadata_lon: float | None,
    rdr_record: dict[str, object] | None,
):
    center_lat = metadata_lat
    center_lon_360 = lon_any_to_360(metadata_lon) if metadata_lon is not None else None
    polygon = None

    if rdr_record is not None:
        center_lat_value = to_float(rdr_record.get("center_lat"))
        if center_lat_value is not None:
            center_lat = center_lat_value
        center_lon_value = to_float(rdr_record.get("center_lon_360"))
        if center_lon_value is not None:
            center_lon_360 = lon_any_to_360(center_lon_value)
        polygon = rdr_record.get("polygon")

    if center_lat is None:
        center_lat = 0.0
    if center_lon_360 is None:
        center_lon_360 = 0.0

    if polygon is None:
        polygon = rectangle_footprint(center_lat, center_lon_360)

    # Full product aspect ratio for quickview alignment correction
    rdr_lines = rdr_record.get("rdr_lines") if rdr_record else None
    rdr_samples = rdr_record.get("rdr_samples") if rdr_record else None
    proj_center_lat = rdr_record.get("proj_center_lat") if rdr_record else None

    properties: dict[str, object] = {
        "instrument": "HIRISE",
        "product_id": image_id,
        "quicklook": f"/hirise/quickview/{image_id}.jpg",
        "red_tif": "",
        "center_latitude": center_lat,
        "center_longitude": center_lon_360,
    }
    if title:
        properties["title"] = title
    if rdr_lines and rdr_samples:
        properties["rdr_lines"] = rdr_lines
        properties["rdr_samples"] = rdr_samples
    if proj_center_lat is not None:
        properties["proj_center_lat"] = proj_center_lat

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon],
        },
    }


def create_symlink(image_id: str):
    dst = QUICKVIEW_DIR / f"{image_id}.jpg"
    if dst.exists() or dst.is_symlink():
        return False

    src = BROWSE_DIR / f"{image_id}_RED.abrowse.jpg"
    if not src.exists():
        return False

    dst.symlink_to(src)
    return True


def main():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata_ids = {item["image_id"] for item in metadata}

    source_index_path = INDEX_BACKUP_PATH if INDEX_BACKUP_PATH.exists() else INDEX_PATH
    existing_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    existing_red_tif = {}
    for feature in existing_index.get("features", []):
        props = feature.get("properties", {})
        image_id = props.get("product_id")
        red_tif = props.get("red_tif")
        if image_id and isinstance(red_tif, str) and red_tif.strip():
            kept = copy.deepcopy(feature)
            kept["geometry"] = normalize_geometry_longitudes(kept.get("geometry", {}))
            existing_red_tif[image_id] = kept

    rdr_records = parse_rdr_rows(metadata_ids)

    features = list(existing_red_tif.values())
    feature_ids = set(existing_red_tif.keys())
    symlinks_created = 0
    fallback_count = 0
    from_existing_count = len(existing_red_tif)

    for item in metadata:
        image_id = item["image_id"]

        if image_id not in feature_ids:
            record = rdr_records.get(image_id)
            feature = build_feature(
                image_id=image_id,
                title=item.get("title"),
                metadata_lat=(float(item["lat"]) if item.get("lat") is not None else None),
                metadata_lon=(float(item["lon"]) if item.get("lon") is not None else None),
                rdr_record=record,
            )
            if record is None or record.get("polygon") is None:
                fallback_count += 1
            features.append(feature)
            feature_ids.add(image_id)

        if create_symlink(image_id):
            symlinks_created += 1

    out = {
        "type": "FeatureCollection",
        "features": features,
    }
    INDEX_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"metadata_images={len(metadata)}")
    print(f"features_written={len(features)}")
    print(f"kept_existing_red_tif={from_existing_count}")
    print(f"rdr_records_found={len(rdr_records)}")
    print(f"fallback_rectangles={fallback_count}")
    print(f"symlinks_created={symlinks_created}")


if __name__ == "__main__":
    main()
