from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import sys

_ = sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.marslandform_v2.config import CLASS_NAMES, GEOJSON_DIR, METADATA_JSON, PREDICTIONS_DIR

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | dict[str, "JSONValue"] | list["JSONValue"]
JSONObject = dict[str, JSONValue]

MARS_GEO_CRS: JSONObject = {
    "type": "name",
    "properties": {"name": "EPSG:4326 (Mars IAU2000 geographic)"},
}


def _read_json(path: Path) -> JSONValue:
    with path.open("r", encoding="utf-8") as f:
        return cast(JSONValue, json.load(f))


def _as_obj(value: JSONValue) -> JSONObject | None:
    if isinstance(value, dict):
        return cast(JSONObject, value)
    return None


def _as_list(value: JSONValue) -> list[JSONValue] | None:
    if isinstance(value, list):
        return value
    return None


def _to_float(value: JSONValue) -> float | None:
    if isinstance(value, (str, int, float, bool)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _extract_lat_lon(record: JSONObject) -> tuple[float | None, float | None]:
    lat = record.get("lat") if record.get("lat") is not None else record.get("center_lat")
    lon = record.get("lon") if record.get("lon") is not None else record.get("center_lon")
    return _to_float(cast(JSONValue, lat)), _to_float(cast(JSONValue, lon))


def _load_metadata_by_image_id(metadata_path: Path) -> dict[str, JSONObject]:
    raw = _read_json(metadata_path)
    raw_obj = _as_obj(raw)
    if raw_obj is not None:
        out: dict[str, JSONObject] = {}
        for k, v in raw_obj.items():
            item = _as_obj(v)
            if item is not None:
                out[str(k)] = item
        return out

    raw_list = _as_list(raw)
    if raw_list is not None:
        out = {}
        for entry in raw_list:
            item = _as_obj(entry)
            if item is None:
                continue
            image_id = item.get("image_id") or item.get("id")
            if image_id is not None:
                out[str(image_id)] = item
        return out

    raise ValueError(f"Unsupported metadata structure in {metadata_path}")


def _normalize_predictions(raw: JSONValue) -> dict[str, JSONObject]:
    raw_obj = _as_obj(raw)
    if raw_obj is not None:
        out: dict[str, JSONObject] = {}
        for key, value in raw_obj.items():
            item = _as_obj(value)
            if item is None:
                continue
            image_id = item.get("image_id") or key
            out[str(image_id)] = item
        return out

    raw_list = _as_list(raw)
    if raw_list is not None:
        out = {}
        for value in raw_list:
            item = _as_obj(value)
            if item is None:
                continue
            image_id = item.get("image_id")
            if image_id is not None:
                out[str(image_id)] = item
        return out

    raise ValueError("Predictions JSON must be an object or an array")


def _normalize_probabilities(pred: JSONObject) -> JSONObject:
    probs = pred.get("probabilities")
    if probs is None:
        return {}

    probs_obj = _as_obj(cast(JSONValue, probs))
    if probs_obj is not None:
        out: JSONObject = {}
        for cls_name, prob in probs_obj.items():
            prob_float = _to_float(prob)
            if prob_float is not None:
                out[str(cls_name)] = prob_float
        return out

    probs_list = _as_list(cast(JSONValue, probs))
    if probs_list is not None:
        out = {}
        for idx, prob in enumerate(probs_list):
            if idx >= len(CLASS_NAMES):
                break
            prob_float = _to_float(prob)
            if prob_float is not None:
                out[CLASS_NAMES[idx]] = prob_float
        return out

    return {}


def _prediction_class(pred: JSONObject) -> str | None:
    value = pred.get("predicted_class") or pred.get("pred_label_name") or pred.get("class")
    return None if value is None else str(value)


def _ground_truth_class(label_record: JSONValue) -> str | None:
    if isinstance(label_record, str):
        return label_record
    item = _as_obj(label_record)
    if item is None:
        return None
    for key in ("ground_truth_class", "final_class", "class", "label", "existing_class"):
        value = item.get(key)
        if value is not None and str(value).upper() != "UNLABELED":
            return str(value)
    return None


def _build_point_feature(image_id: str, lat: float, lon: float, properties: JSONObject) -> JSONObject:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"image_id": image_id, **properties},
    }


def export_geojson(predictions_path: Path, metadata_path: Path, output_path: Path, include_tiles: bool = False) -> Path:
    predictions = _normalize_predictions(_read_json(predictions_path))
    metadata = _load_metadata_by_image_id(metadata_path)

    features: list[JSONValue] = []
    skipped = 0
    for image_id, pred in predictions.items():
        meta = metadata.get(image_id)
        if meta is None:
            skipped += 1
            continue

        lat, lon = _extract_lat_lon(meta)
        if lat is None or lon is None:
            skipped += 1
            continue

        source = str(pred.get("mode") or pred.get("source") or "fast")
        confidence = _to_float(cast(JSONValue, pred.get("confidence")))
        reasoning = pred.get("reasoning") or pred.get("reasoning_chain")
        class_probs = _normalize_probabilities(pred)
        pred_class = _prediction_class(pred)

        features.append(
            _build_point_feature(
                image_id,
                lat,
                lon,
                {
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "class_probabilities": class_probs,
                    "source": source,
                    "reasoning": reasoning if source == "agent" else None,
                    "feature_type": "image",
                },
            )
        )

        if include_tiles:
            attention = _as_list(cast(JSONValue, pred.get("attention_weights") or pred.get("tile_attention") or []))
            if attention is None:
                continue
            for idx, raw_weight in enumerate(attention):
                tile_weight = _to_float(raw_weight)
                if tile_weight is None:
                    continue
                features.append(
                    _build_point_feature(
                        image_id,
                        lat,
                        lon,
                        {
                            "predicted_class": pred_class,
                            "confidence": confidence,
                            "class_probabilities": class_probs,
                            "source": source,
                            "reasoning": reasoning if source == "agent" else None,
                            "feature_type": "tile",
                            "tile_index": idx,
                            "attention_weight": tile_weight,
                        },
                    )
                )

    payload: JSONObject = {
        "type": "FeatureCollection",
        "crs": MARS_GEO_CRS,
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "model_version": "MarsLandformNet V2",
            "num_images": len(predictions),
            "num_features": len(features),
            "missing_or_invalid_metadata": skipped,
        },
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def export_catalog_geojson(metadata_path: Path, labels_path: Path, predictions_path: Path, output_path: Path) -> Path:
    metadata = _load_metadata_by_image_id(metadata_path)
    labels_raw: JSONValue = _read_json(labels_path) if labels_path.exists() else {}
    predictions_raw: JSONValue = _read_json(predictions_path) if predictions_path.exists() else {}

    labels: dict[str, JSONValue] = {}
    labels_obj = _as_obj(labels_raw)
    if labels_obj is not None:
        labels = {str(k): v for k, v in labels_obj.items()}
    else:
        labels_arr = _as_list(labels_raw)
        if labels_arr is not None:
            for entry in labels_arr:
                item = _as_obj(entry)
                if item is None:
                    continue
                image_id = item.get("image_id")
                if image_id is not None:
                    labels[str(image_id)] = item

    predictions = _normalize_predictions(predictions_raw)

    features: list[JSONValue] = []
    for image_id, meta in metadata.items():
        lat, lon = _extract_lat_lon(meta)
        if lat is None or lon is None:
            continue
        pred = predictions.get(image_id)
        pred_class = _prediction_class(pred) if pred is not None else None
        truth = _ground_truth_class(labels.get(image_id))
        features.append(
            _build_point_feature(
                image_id,
                lat,
                lon,
                {
                    "lat": lat,
                    "lon": lon,
                    "predicted_class": pred_class,
                    "ground_truth_class": truth,
                    "is_labeled": bool(truth),
                    "is_predicted": bool(pred_class),
                },
            )
        )

    num_labeled = sum(
        1
        for feature in features
        if bool(_as_obj(cast(JSONValue, cast(JSONObject, feature).get("properties"))).get("is_labeled") if _as_obj(cast(JSONValue, cast(JSONObject, feature).get("properties"))) else False)
    )
    num_predicted = sum(
        1
        for feature in features
        if bool(_as_obj(cast(JSONValue, cast(JSONObject, feature).get("properties"))).get("is_predicted") if _as_obj(cast(JSONValue, cast(JSONObject, feature).get("properties"))) else False)
    )

    payload: JSONObject = {
        "type": "FeatureCollection",
        "crs": MARS_GEO_CRS,
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "model_version": "MarsLandformNet V2",
            "num_images": len(features),
            "num_labeled": num_labeled,
            "num_predicted": num_predicted,
        },
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Export MarsLandformNet V2 predictions as GeoJSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pred_parser = subparsers.add_parser("predictions", help="Export prediction results as GeoJSON")
    _ = pred_parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_DIR / "test_predictions_with_attention.json")
    _ = pred_parser.add_argument("--metadata-path", type=Path, default=METADATA_JSON)
    _ = pred_parser.add_argument("--output-path", type=Path, default=GEOJSON_DIR / "predictions.geojson")
    _ = pred_parser.add_argument("--include-tiles", action="store_true")

    catalog_parser = subparsers.add_parser("catalog", help="Export full image catalog as GeoJSON")
    _ = catalog_parser.add_argument("--metadata-path", type=Path, default=METADATA_JSON)
    _ = catalog_parser.add_argument("--labels-path", type=Path, required=True)
    _ = catalog_parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_DIR / "test_predictions_with_attention.json")
    _ = catalog_parser.add_argument("--output-path", type=Path, default=GEOJSON_DIR / "catalog.geojson")
    return cast(dict[str, object], vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    command = str(args.get("command", ""))
    if command == "predictions":
        out_path = export_geojson(
            predictions_path=cast(Path, args["predictions_path"]),
            metadata_path=cast(Path, args["metadata_path"]),
            output_path=cast(Path, args["output_path"]),
            include_tiles=bool(cast(bool, args.get("include_tiles", False))),
        )
    else:
        out_path = export_catalog_geojson(
            metadata_path=cast(Path, args["metadata_path"]),
            labels_path=cast(Path, args["labels_path"]),
            predictions_path=cast(Path, args["predictions_path"]),
            output_path=cast(Path, args["output_path"]),
        )
    print(f"GeoJSON written to: {out_path}")


if __name__ == "__main__":
    main()
