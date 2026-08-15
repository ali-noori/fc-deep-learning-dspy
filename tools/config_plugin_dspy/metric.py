"""DSPy teleprompt metric: score predicted intent JSON against gold examples."""

from __future__ import annotations

import json

from .json_sanitizer import extract_json_object

INTENT_KEYS = (
    "model_name",
    "trainer_name",
    "aggregator_name",
    "loss_name",
    "data_loader_name",
)


def intent_metric(example, predicted, trace=None) -> float:
    try:
        gold_raw = getattr(example, "intent_json", None)
        if gold_raw is None and isinstance(example, dict):
            gold_raw = example.get("intent_json")
        pred_raw = getattr(predicted, "intent_json", None)
        if pred_raw is None and isinstance(predicted, dict):
            pred_raw = predicted.get("intent_json")

        if not pred_raw or not str(pred_raw).strip():
            return 0.0

        gold_intent = json.loads(str(gold_raw))
        pred_intent = extract_json_object(str(pred_raw))

        matches = sum(
            1 for key in INTENT_KEYS if gold_intent.get(key) == pred_intent.get(key)
        )
        return matches / len(INTENT_KEYS)
    except Exception:
        return 0.0
