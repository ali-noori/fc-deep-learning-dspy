"""DSPy teleprompt metric: score predicted trainer fields against gold examples."""

from __future__ import annotations

from .trainer_selector import (
    _resolve_dataloader_name,
    _resolve_loss_name,
    _resolve_trainer_name,
)


def _norm_int(value: object) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return ""


def _norm_trainer_name(
    value: object,
    trainer_artifacts: dict[str, str],
) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        return _resolve_trainer_name(text, trainer_artifacts)
    except ValueError:
        return ""


def _norm_dataloader_name(
    value: object,
    dataloader_artifacts: dict[str, str],
) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        return _resolve_dataloader_name(text, dataloader_artifacts)
    except ValueError:
        return ""


def _norm_loss_name(
    value: object,
    loss_artifacts: dict[str, str],
) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        return _resolve_loss_name(text, loss_artifacts)
    except ValueError:
        return ""


def _get_field(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def trainer_config_metric(
    example,
    predicted,
    trace=None,
    trainer_artifacts: dict[str, str] | None = None,
    dataloader_artifacts: dict[str, str] | None = None,
    loss_artifacts: dict[str, str] | None = None,
) -> float:
    try:
        trainers = trainer_artifacts or {}
        dataloaders = dataloader_artifacts or {}
        losses = loss_artifacts or {}

        actual_name = _norm_trainer_name(_get_field(example, "name"), trainers)
        pred_name = _norm_trainer_name(_get_field(predicted, "name"), trainers)
        if not actual_name or actual_name != pred_name:
            return 0.0

        actual_loader = _norm_dataloader_name(
            _get_field(example, "data_loader"),
            dataloaders,
        )
        pred_loader = _norm_dataloader_name(
            _get_field(predicted, "data_loader"),
            dataloaders,
        )
        if not actual_loader or actual_loader != pred_loader:
            return 0.0

        actual_loss = _norm_loss_name(_get_field(example, "loss_name"), losses)
        pred_loss = _norm_loss_name(_get_field(predicted, "loss_name"), losses)
        if not actual_loss or actual_loss != pred_loss:
            return 0.0

        actual_classes = _norm_int(_get_field(example, "num_classes"))
        pred_classes = _norm_int(_get_field(predicted, "num_classes"))
        if not actual_classes or actual_classes != pred_classes:
            return 0.0

        return 1.0
    except Exception as exc:
        raise RuntimeError("trainer_config_metric evaluation failed.") from exc
