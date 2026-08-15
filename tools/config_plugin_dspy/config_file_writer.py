"""Low-level UTF-8 file write."""

from __future__ import annotations

from pathlib import Path


def write_config_file(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file at {path}") from exc
