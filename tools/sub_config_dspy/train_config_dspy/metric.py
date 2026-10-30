"""DSPy teleprompt metric: score predicted train_config fields against gold examples."""

from __future__ import annotations

from .train_config_selector import _parse_bool, _parse_positive_float, _parse_positive_int


def _get_field(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _norm_bool(value: object) -> str:
    try:
        return "true" if _parse_bool(str(value or "")) else "false"
    except ValueError:
        return ""


def _norm_int(value: object, field_name: str) -> str:
    try:
        return str(_parse_positive_int(str(value or ""), field_name))
    except ValueError:
        return ""


def _norm_float(value: object, field_name: str) -> str:
    try:
        return str(_parse_positive_float(str(value or ""), field_name))
    except ValueError:
        return ""


def train_config_metric(example, predicted, trace=None) -> float:
    try:
        if _norm_bool(_get_field(example, "verbose")) != _norm_bool(
            _get_field(predicted, "verbose")
        ):
            return 0.0
        if _norm_int(_get_field(example, "batch_size"), "batch_size") != _norm_int(
            _get_field(predicted, "batch_size"), "batch_size"
        ):
            return 0.0
        if _norm_int(
            _get_field(example, "test_batch_size"), "test_batch_size"
        ) != _norm_int(_get_field(predicted, "test_batch_size"), "test_batch_size"):
            return 0.0
        if _norm_int(_get_field(example, "epochs"), "epochs") != _norm_int(
            _get_field(predicted, "epochs"), "epochs"
        ):
            return 0.0
        if _norm_float(_get_field(example, "lr"), "lr") != _norm_float(
            _get_field(predicted, "lr"), "lr"
        ):
            return 0.0
        if _norm_int(_get_field(example, "batch_count"), "batch_count") != _norm_int(
            _get_field(predicted, "batch_count"), "batch_count"
        ):
            return 0.0
        if not _norm_bool(_get_field(example, "verbose")):
            return 0.0
        return 1.0
    except Exception as exc:
        raise RuntimeError("train_config_metric evaluation failed.") from exc
