"""DSPy teleprompt metric: score predicted hyperparameter fields against gold examples."""

from __future__ import annotations

from .hyper_params_selector import _resolve_federated_model


def _norm_int(value: object) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return ""


def _norm_federated_model(
    value: object,
    aggregator_artifacts: dict[str, str],
) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        return _resolve_federated_model(text, aggregator_artifacts)
    except ValueError:
        return ""


def _get_field(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def hyper_params_config_metric(
    example,
    predicted,
    trace=None,
    aggregator_artifacts: dict[str, str] | None = None,
) -> float:
    try:
        artifacts = aggregator_artifacts or {}

        actual_max_iter = _norm_int(_get_field(example, "max_iter"))
        pred_max_iter = _norm_int(_get_field(predicted, "max_iter"))
        if not actual_max_iter or actual_max_iter != pred_max_iter:
            return 0.0

        actual_n_classes = _norm_int(_get_field(example, "n_classes"))
        pred_n_classes = _norm_int(_get_field(predicted, "n_classes"))
        if not actual_n_classes or actual_n_classes != pred_n_classes:
            return 0.0

        actual_model = _norm_federated_model(
            _get_field(example, "federated_model"),
            artifacts,
        )
        pred_model = _norm_federated_model(
            _get_field(predicted, "federated_model"),
            artifacts,
        )
        if not actual_model or actual_model != pred_model:
            return 0.0

        return 1.0
    except Exception as exc:
        raise RuntimeError("hyper_params_config_metric evaluation failed.") from exc
