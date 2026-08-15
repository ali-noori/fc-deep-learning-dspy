"""DSPy teleprompt metric: score predicted execution_mode against gold examples."""

from __future__ import annotations

from .config import EXECUTION_MODES


def execution_mode_metric(example, predicted, trace=None) -> float:
    try:
        
        # Get the actual execution mode
        actual = getattr(example, "execution_mode", None)
        if actual is None and isinstance(example, dict):
            actual = example.get("execution_mode")
        
        # Get the predicted execution mode
        pred = getattr(predicted, "execution_mode", None)
        if pred is None and isinstance(predicted, dict):
            pred = predicted.get("execution_mode")

        # prediction is not empty
        if not pred or not str(pred).strip():
            return 0.0

        actual_mode = str(actual).strip().lower()
        pred_mode = str(pred).strip().lower()
        
        # predictio be exactly just liek the actual
        if pred_mode not in EXECUTION_MODES:
            return 0.0
        return 1.0 if actual_mode == pred_mode else 0.0
    except Exception as exc:
        raise RuntimeError("execution_mode_metric evaluation failed.") from exc
