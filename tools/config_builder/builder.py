"""Build final {mode}_config.yml from template + generated DSPy blocks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

MODES = ("federated", "centralized", "simulation")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    data = yaml.safe_load(text)
    if isinstance(data, dict):
        return data
    return {}


def _deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        out = deepcopy(base)
        for key, value in patch.items():
            if key in out:
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = deepcopy(value)
        return out
    return deepcopy(patch)


def _mode_template_paths(mode: str) -> tuple[Path, Path]:
    root = _repo_root()
    required = root / "tools" / "execution_mode_dspy" / "valid_config_file" / f"{mode}.template.yml"
    partial = (
        root
        / "tools"
        / "execution_mode_dspy"
        / "valid_config_file"
        / "valid_config_file_partially_filled"
        / mode
        / f"{mode}.template.partially.yml"
    )
    return required, partial


def _generated_block_paths() -> dict[str, tuple[Path, ...]]:
    root = _repo_root()
    return {
        "dataset": (
            root / "tools" / "sub_config_dspy" / "dataset_dspy" / "output" / "output_dataset_dspy.yml",
        ),
        "hyper_params": (
            root / "tools" / "sub_config_dspy" / "hyper_params_dspy" / "output" / "output_fed_hyper_param.yml",
        ),
        "model": (
            root / "tools" / "sub_config_dspy" / "model_dspy" / "output" / "output_model_dspy.yml",
        ),
        "trainer": (
            # current writer location
            root / "tools" / "sub_config_dspy" / "trainer_dspy" / "output" / "output_trainer_dspy.yml",
            # compatibility with typo in request
            root / "tools" / "sub_config_dspy" / "trainer_dspy" / "outcome" / "output_trainer_dspy.yml",
        ),
    }


def _load_generated_blocks() -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for key, candidates in _generated_block_paths().items():
        loaded: dict[str, Any] = {}
        for path in candidates:
            loaded = _load_yaml(path)
            if loaded:
                break
        blocks[key] = loaded
    return blocks


def dataset_output_path() -> Path:
    return (
        _repo_root()
        / "tools"
        / "sub_config_dspy"
        / "dataset_dspy"
        / "output"
        / "output_dataset_dspy.yml"
    )


def read_execution_mode_from_dataset() -> str:
    """Read execution_mode from the dataset DSPy output file."""
    path = dataset_output_path()
    if not path.exists():
        raise ValueError(f"Dataset output file not found: {path}")

    data = _load_yaml(path)
    if "execution_mode" not in data:
        raise ValueError(
            f"Dataset output file does not contain 'execution_mode', need to run the pipeline: {path}"
        )

    raw_mode = data["execution_mode"]
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        raise ValueError(
            f"Dataset output file has invalid 'execution_mode' "
            f"(must be a non-empty string): {path}"
        )

    mode = raw_mode.strip().lower()
    if mode not in MODES:
        raise ValueError(
            f"Incompatible execution_mode {raw_mode!r} in dataset file. "
            f"Expected one of: {', '.join(MODES)}."
        )
    return mode


def resolve_execution_mode(cli_mode: str | None = None) -> str:
    """
    Determine execution mode from the dataset output file.

    If cli_mode is given, it must match the dataset file's execution_mode.
    """
    dataset_mode = read_execution_mode_from_dataset()
    if cli_mode is None:
        return dataset_mode
    if cli_mode != dataset_mode:
        raise ValueError(
            f"Incompatible execution mode: --execution-mode requested {cli_mode!r} "
            f"but dataset file contains execution_mode: {dataset_mode!r}."
        )
    return dataset_mode


def _extract_fc_deep_patch(block: dict[str, Any]) -> dict[str, Any]:
    if not block:
        return {}
    if isinstance(block.get("fc_deep"), dict):
        return deepcopy(block["fc_deep"])
    # sub-agents usually emit top-level snippets directly
    return deepcopy(block)


def _pick_value(
    key: str,
    required: dict[str, Any],
    partial: dict[str, Any],
    generated_patch: dict[str, Any],
) -> Any:
    if key in generated_patch:
        return deepcopy(generated_patch[key])
    if key in partial:
        return deepcopy(partial[key])
    return deepcopy(required.get(key))


def _ordered_union_keys(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in first.keys():
        if key not in keys:
            keys.append(key)
    for key in second.keys():
        if key not in keys:
            keys.append(key)
    return keys


def build_config(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of: {', '.join(MODES)}.")

    required_path, partial_path = _mode_template_paths(mode)
    required_template = _load_yaml(required_path)
    partial_template = _load_yaml(partial_path)

    required_fc = required_template.get("fc_deep", {})
    partial_fc = partial_template.get("fc_deep", {})

    # Merge all generated snippets into one patch.
    merged_patch: dict[str, Any] = {}
    for block in _load_generated_blocks().values():
        merged_patch = _deep_merge(merged_patch, _extract_fc_deep_patch(block))

    out_fc: dict[str, Any] = {}
    for key in _ordered_union_keys(required_fc, partial_fc):
        out_fc[key] = _pick_value(key, required_fc, partial_fc, merged_patch)

    return {"fc_deep": out_fc}


def output_filename_for_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of: {', '.join(MODES)}.")
    return f"{mode}_config.yml"


def output_path_for_mode(mode: str) -> Path:
    return (
        _repo_root()
        / "tools"
        / "config_builder"
        / "output"
        / mode
        / output_filename_for_mode(mode)
    )


def write_generated_config(config: dict[str, Any], mode: str) -> Path:
    out_path = output_path_for_mode(mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# execution mode: {mode}\n"
    body = yaml.safe_dump(config, sort_keys=False)
    out_path.write_text(header + body, encoding="utf-8")
    return out_path

