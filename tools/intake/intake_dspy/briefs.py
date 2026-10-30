"""Shared brief file written by intake_dspy and read by later agents."""

from __future__ import annotations

import json
from pathlib import Path

BRIEF_KEYS: tuple[str, ...] = (
    "execution_mode_dspy",
    "dataset_dspy",
    "hyper_params_dspy",
    "trainer_dspy",
    "train_config_dspy",
)


def briefs_path() -> Path:
    return Path(__file__).resolve().parent / "output" / "intake_briefs.json"


def save_briefs(briefs: dict[str, str]) -> Path:
    path = briefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: str(briefs.get(key, "")).strip() for key in BRIEF_KEYS}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_briefs() -> dict[str, str]:
    path = briefs_path()
    if not path.is_file():
        raise FileNotFoundError(f"Intake briefs not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Intake briefs file is not an object: {path}")
    return {key: str(data.get(key, "")).strip() for key in BRIEF_KEYS}


def brief_for(key: str) -> str:
    briefs = load_briefs()
    if key not in briefs:
        raise KeyError(f"Unknown intake brief key: {key}")
    text = briefs[key].strip()
    if not text:
        raise ValueError(f"Intake brief {key!r} is empty.")
    return text
