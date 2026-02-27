from __future__ import annotations

from typing import Any, Dict, List

from ...config import CLASS_ORDER


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class ClassifyTool:
    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier

    def run(self, image_id: str) -> Dict[str, Any]:
        if not self.classifier:
            return {
                "class": "BACKGROUND",
                "confidence": 0.0,
                "per_class_probs": {},
                "top_3_tiles": [],
                "attention_summary": "Classifier unavailable.",
                "error": "Classifier is not initialized.",
            }

        trained_flag = getattr(self.classifier, "is_trained", None)
        if trained_flag is False:
            return {
                "class": "BACKGROUND",
                "confidence": 0.0,
                "per_class_probs": {},
                "top_3_tiles": [],
                "attention_summary": "Classifier has not been trained yet.",
                "error": "Classifier not trained.",
            }

        try:
            raw = self._invoke_model(image_id)
        except Exception as exc:
            return {
                "class": "BACKGROUND",
                "confidence": 0.0,
                "per_class_probs": {},
                "top_3_tiles": [],
                "attention_summary": "Inference failed.",
                "error": str(exc),
            }

        output = raw if isinstance(raw, dict) else {"raw": raw}

        pred_class = str(
            output.get("class")
            or output.get("predicted_class")
            or output.get("label")
            or "BACKGROUND"
        ).upper()
        if pred_class not in CLASS_ORDER:
            pred_class = "BACKGROUND"

        confidence = float(output.get("confidence", output.get("score", 0.0)) or 0.0)
        confidence = max(0.0, min(1.0, confidence))

        per_class_probs = self._format_probs(output)
        attention_weights = self._extract_attention(output)
        top_3_tiles = self._top_tiles(attention_weights, top_k=3)
        attention_summary = self._summarize_attention(attention_weights, pred_class)

        return {
            "class": pred_class,
            "confidence": confidence,
            "per_class_probs": per_class_probs,
            "top_3_tiles": top_3_tiles,
            "attention_summary": attention_summary,
        }

    def _invoke_model(self, image_id: str) -> Any:
        if hasattr(self.classifier, "predict_image"):
            return self.classifier.predict_image(image_id)
        if hasattr(self.classifier, "predict"):
            return self.classifier.predict(image_id)
        if callable(self.classifier):
            return self.classifier(image_id)
        raise RuntimeError("Classifier has no callable inference interface.")

    def _format_probs(self, output: Dict[str, Any]) -> Dict[str, float]:
        probs = output.get("per_class_probs")
        if isinstance(probs, dict):
            return {str(k).upper(): float(v) for k, v in probs.items()}

        probs = output.get("probs") or output.get("probabilities")
        probs = _to_python(probs)
        if isinstance(probs, list):
            result: Dict[str, float] = {}
            for idx, cls_name in enumerate(CLASS_ORDER):
                if idx < len(probs):
                    result[cls_name] = float(probs[idx])
            return result
        return {}

    def _extract_attention(self, output: Dict[str, Any]) -> List[float]:
        attention = (
            output.get("attention_weights")
            or output.get("tile_attention")
            or output.get("attention")
            or []
        )
        attention = _to_python(attention)
        if not isinstance(attention, list):
            return []
        return [float(v) for v in attention]

    @staticmethod
    def _top_tiles(attention_weights: List[float], top_k: int) -> List[Dict[str, float]]:
        ranked = sorted(enumerate(attention_weights), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [{"tile_index": idx, "attention_weight": float(weight)} for idx, weight in ranked]

    @staticmethod
    def _summarize_attention(attention_weights: List[float], pred_class: str) -> str:
        if not attention_weights:
            return f"No attention map available for predicted class {pred_class}."
        max_w = max(attention_weights)
        min_w = min(attention_weights)
        mean_w = sum(attention_weights) / len(attention_weights)
        return (
            f"Attention concentrated around a few tiles (max={max_w:.3f}, "
            f"mean={mean_w:.3f}, min={min_w:.3f}) for class {pred_class}."
        )
