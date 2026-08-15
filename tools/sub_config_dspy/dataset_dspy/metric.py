"""DSPy teleprompt metric: score predicted dataset fields against gold examples."""

from __future__ import annotations

from pathlib import Path


def _norm_filename(value: object) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    return Path(text).name.lower() if text else ""


def _norm_logic_dir(value: object) -> str:
    return str(value or "").strip().strip('"').strip("'").strip("/\\").lower()


def _norm_dirs(value: object) -> tuple[str, ...]:
    raw = str(value or "").strip()
    if not raw:
        return ()
    parts = [p.strip().strip('"').strip("'") for p in raw.replace(";", ",").split(",")]
    normalized = []
    for part in parts:
        if not part:
            continue
        path = part.replace("\\", "/").rstrip("/").lower()
        normalized.append(path)
    return tuple(sorted(normalized))

def _norm_dir(value: object) -> str:
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    
    normalized = raw.replace("\\", "/").rstrip("/").lower()
    
    return normalized
    


def _get_field(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None

def dataset_federated_config_metric(example, predicted, trace=None) -> float:
    try:
        actual_dirs = _norm_dirs(_get_field(example, "data_dirs"))
        pred_dirs = _norm_dirs(_get_field(predicted, "data_dirs"))
        if not actual_dirs or actual_dirs != pred_dirs:
            return 0.0

        fields = (
            "train_dataset_file_name",
            "test_dataset_file_name",
            "logic_dir",
        )
        for field in fields:
            actual_raw = _get_field(example, field)
            pred_raw = _get_field(predicted, field)
            if field == "logic_dir":
                actual = _norm_logic_dir(actual_raw)
                pred = _norm_logic_dir(pred_raw)
            else:
                actual = _norm_filename(actual_raw)
                pred = _norm_filename(pred_raw)
            if not actual or actual != pred:
                return 0.0

        return 1.0
    except Exception as exc:
        raise RuntimeError("dataset_config_metric evaluation failed.") from exc
    
def dataset_centralized_config_metric(example, predicted, trace=None) -> float:
    try:
        actual_dirs = _norm_dir(_get_field(example, "data_dir"))
        pred_dirs = _norm_dir(_get_field(predicted, "data_dir"))
        if not actual_dirs or actual_dirs != pred_dirs:
            return 0.0

        fields = (
            "train_dataset_file_name",
            "test_dataset_file_name",
            "logic_dir",
        )
        for field in fields:
            actual_raw = _get_field(example, field)
            pred_raw = _get_field(predicted, field)
            if field == "logic_dir":
                actual = _norm_logic_dir(actual_raw)
                pred = _norm_logic_dir(pred_raw)
            else:
                actual = _norm_filename(actual_raw)
                pred = _norm_filename(pred_raw)
            if not actual or actual != pred:
                return 0.0

        return 1.0
    except Exception as exc:
        raise RuntimeError("dataset_config_metric evaluation failed.") from exc
    
def dataset_simulation_config_metric(example, predicted, trace=None) -> float:
    try:
        actual_client_dirs = _norm_dirs(_get_field(example, "client_dirs"))
        pred_client_dirs = _norm_dirs(_get_field(predicted, "client_dirs"))
        if not actual_client_dirs or actual_client_dirs != pred_client_dirs:
            return 0.0

        actual_data_dir = _norm_dir(_get_field(example, "data_dir"))
        pred_data_dir = _norm_dir(_get_field(predicted, "data_dir"))
        if not actual_data_dir or actual_data_dir != pred_data_dir:
            return 0.0

        fields = (
            "train_dataset_file_name",
            "test_dataset_file_name",
            "logic_dir",
        )
        for field in fields:
            actual_raw = _get_field(example, field)
            pred_raw = _get_field(predicted, field)
            if field == "logic_dir":
                actual = _norm_logic_dir(actual_raw)
                pred = _norm_logic_dir(pred_raw)
            else:
                actual = _norm_filename(actual_raw)
                pred = _norm_filename(pred_raw)
            if not actual or actual != pred:
                return 0.0
            
        return 1.0
    except Exception as exc:
        raise RuntimeError("dataset_config_metric evaluation failed.") from exc
                