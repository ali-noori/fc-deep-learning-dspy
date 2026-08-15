"""Discover plugin and built-in artifact names (aligned with tools.dspy.optimization)."""

from __future__ import annotations

from pathlib import Path


def _py_files_in_folder(path: Path) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []
    out: list[str] = []
    for entry in path.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix != ".py":
            continue
        if entry.name == "__init__.py" or entry.name.startswith("."):
            continue
        out.append(entry.name)
    out.sort(key=str.casefold)
    return out


def discover_repo_artifacts(repo_root: Path) -> dict[str, list[str]]:
    root = Path(repo_root)
    aggregator_builtins = ["FedAvg"]
    loss_builtins = ["CrossEntropyLoss"]
    dataloader_builtins = ["ImageLoader", "ImageLoader.py"]

    artifacts: dict[str, list[str]] = {
        "models": _py_files_in_folder(root / "plugins" / "models")
        + _py_files_in_folder(root / "models" / "pytorch"),
        "trainers": _py_files_in_folder(root / "plugins" / "trainers"),
        "dataloaders": dataloader_builtins
        + _py_files_in_folder(root / "plugins" / "dataloaders"),
        "aggregators": aggregator_builtins
        + _py_files_in_folder(root / "plugins" / "aggregators"),
        "losses": loss_builtins + _py_files_in_folder(root / "plugins" / "loss"),
        "optimizers": ["SGD", "Adam", "AdamW", "RMSprop"],
        "devices": ["cpu", "gpu"],
    }

    for key, items in artifacts.items():
        if isinstance(items, list):
            artifacts[key] = list(dict.fromkeys(items))
    return artifacts
