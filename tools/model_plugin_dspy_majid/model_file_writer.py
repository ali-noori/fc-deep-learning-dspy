"""Write generated Python to disk (one job: filesystem I/O)."""

from __future__ import annotations

from pathlib import Path


def write_model_file(path: Path, content: str) -> None:
    """Create parent folders if needed and write UTF-8 text."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write model file at {path}") from exc
