"""Combine header + body and persist under the configured repo path."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .generation_header import build_generation_header
from .model_file_writer import write_model_file


class GeneratedModelWriter:
    """Writes `customized_model_majid.py` (or configured filename) under repo_root."""

    def __init__(self, settings: Settings, repo_root: Path) -> None:
        self._settings = settings
        self._repo_root = repo_root

    def write(self, code: str, backend: str, model: str, base_url: str | None) -> Path:
        """Return the path that was written."""
        try:
            output_path = self._repo_root.joinpath(
                *self._settings.output_path_parts,
                self._settings.output_filename,
            )
            header = build_generation_header(self._settings, backend, model, base_url)
            body = code.rstrip() + "\n"
            write_model_file(output_path, header + body)
            return output_path
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Failed to write generated model artifact.") from exc
