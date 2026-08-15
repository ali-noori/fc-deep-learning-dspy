"""DSPy teleprompt metric: score predicted model fields against gold examples."""

from __future__ import annotations

from .model_config_selector import _resolve_model_name


def _norm_int(value: object) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return ""


def _norm_model_name(
    value: object,
    model_artifacts: dict[str, str],
) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        return _resolve_model_name(text, model_artifacts)
    except ValueError:
        return ""


def _get_field(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def model_config_metric(
    example,
    predicted,
    trace=None,
    model_artifacts: dict[str, str] | None = None,
) -> float:
    try:
        artifacts = model_artifacts or {}

        actual_name = _norm_model_name(
            _get_field(example, "name"),
            artifacts,
        )
        pred_name = _norm_model_name(
            _get_field(predicted, "name"),
            artifacts,
        )
        if not actual_name or actual_name != pred_name:
            return 0.0

        actual_n_classes = _norm_int(_get_field(example, "n_classes"))
        pred_n_classes = _norm_int(_get_field(predicted, "n_classes"))
        if not actual_n_classes or actual_n_classes != pred_n_classes:
            return 0.0

        actual_in_features = _norm_int(_get_field(example, "in_features"))
        pred_in_features = _norm_int(_get_field(predicted, "in_features"))
        if not actual_in_features or actual_in_features != pred_in_features:
            return 0.0

        return 1.0
    except Exception as exc:
        raise RuntimeError("model_config_metric evaluation failed.") from exc
