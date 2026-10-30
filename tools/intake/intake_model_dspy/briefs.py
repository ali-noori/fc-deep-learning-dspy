"""Brief file written by intake_model_dspy."""

from __future__ import annotations

import json
from pathlib import Path

BRIEF_KEYS: tuple[str, ...] = (
    "architecture_description",
    "model_selection",
    "in_feature",
    "n_classes",
)

OPTIONAL_KEYS = {"architecture_description", "model_selection"}


def briefs_path() -> Path:
    return Path(__file__).resolve().parent / "output" / "intake_model_briefs.json"


def save_briefs(briefs: dict[str, str]) -> Path:
    path = briefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: str(briefs.get(key, "")).strip() for key in BRIEF_KEYS}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_briefs() -> dict[str, str]:
    path = briefs_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model intake briefs not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Model intake briefs file is not an object: {path}")
    return {key: str(data.get(key, "")).strip() for key in BRIEF_KEYS}


def brief_for(key: str) -> str:
    briefs = load_briefs()
    if key not in briefs:
        raise KeyError(f"Unknown model intake brief key: {key}")
    text = briefs[key].strip()
    if not text and key not in OPTIONAL_KEYS:
        raise ValueError(f"Model intake brief {key!r} is empty.")
    return text
