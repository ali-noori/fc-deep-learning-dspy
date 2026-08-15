"""DSPy signature and module for config intent extraction."""

from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .dspy_support import dspy
from .json_sanitizer import extract_json_object
from .types import IntentExtractionResult, UserIntent


class IntentExtractionSignature(dspy.Signature):
    """Extract structured FeatureCloud config intent from a user request."""

    user_request: str = dspy.InputField(desc="Natural language configuration request")
    known_models: str = dspy.InputField(desc="Available model plugin filenames")
    known_trainers: str = dspy.InputField(desc="Available trainer plugin filenames")
    known_dataloaders: str = dspy.InputField(desc="Available data loader plugin filenames")
    known_aggregators: str = dspy.InputField(desc="Available aggregator plugin filenames")
    known_losses: str = dspy.InputField(desc="Available loss plugin filenames")
    known_optimizers: str = dspy.InputField(desc="Available optimizer names")
    known_devices: str = dspy.InputField(desc="Available device options (cpu, gpu)")

    intent_json: str = dspy.OutputField(
        desc=(
            "JSON object with optional keys: task, model_name, trainer_name, "
            "aggregator_name, loss_name, data_loader_name. Use exact names from known lists."
        )
    )


class ConfigIntentExtractorModule(dspy.Module):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._predictor = dspy.ChainOfThought(IntentExtractionSignature)

    def forward(
        self,
        user_request: str,
        known_models: str,
        known_trainers: str,
        known_dataloaders: str,
        known_aggregators: str,
        known_losses: str,
        known_optimizers: str,
        known_devices: str,
    ) -> IntentExtractionResult:
        last_error: Exception | None = None
        for _ in range(self._settings.intent_extract_max_retries):
            try:
                prediction = self._predictor(
                    user_request=user_request,
                    known_models=known_models,
                    known_trainers=known_trainers,
                    known_dataloaders=known_dataloaders,
                    known_aggregators=known_aggregators,
                    known_losses=known_losses,
                    known_optimizers=known_optimizers,
                    known_devices=known_devices,
                )
                raw = getattr(prediction, "intent_json", None)
                if raw is None and isinstance(prediction, dict):
                    raw = prediction.get("intent_json")
                intent = prediction_to_user_intent(user_request, prediction)

                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                return IntentExtractionResult(
                    intent=intent,
                    intent_json=str(raw or ""),
                    backend=backend,
                    model=model,
                    base_url=base_url,
                )
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Intent extraction failed after {self._settings.intent_extract_max_retries} attempts."
        ) from last_error


def _intent_str(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    if value is None:
        return default
    return str(value).strip()


def prediction_to_user_intent(user_request: str, prediction: Any) -> UserIntent:
    raw = getattr(prediction, "intent_json", None)
    if raw is None and isinstance(prediction, dict):
        raw = prediction.get("intent_json")
    payload = extract_json_object(str(raw or ""))
    return UserIntent(
        task=str(payload.get("task") or "federated_training"),
        model_name=_intent_str(payload, "model_name"),
        trainer_name=_intent_str(payload, "trainer_name"),
        aggregator_name=_intent_str(payload, "aggregator_name", "federated_model"),
        loss_name=_intent_str(payload, "loss_name", "loss"),
        data_loader_name=_intent_str(payload, "data_loader_name", "data_loader"),
        raw_request=user_request,
    )
