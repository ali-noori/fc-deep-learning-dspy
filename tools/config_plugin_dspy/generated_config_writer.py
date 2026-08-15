"""Write header + merged YAML under tools/output/config/generated/."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings
from .config_file_writer import write_config_file
from .generation_header import build_generation_header


class GeneratedConfigWriter:
    def __init__(self, settings: Settings, repo_root: Path) -> None:
        self._settings = settings
        self._repo_root = repo_root

    def write(
        self,
        config_data: dict,
        backend: str,
        model: str,
        base_url: str | None,
        output_path: Path | None = None,
    ) -> Path:
        if output_path is None:
            output_path = self._repo_root.joinpath(
                *self._settings.output_path_parts,
                self._settings.output_filename,
            )
        header = build_generation_header(self._settings, backend, model, base_url)
        body = yaml.safe_dump(config_data, sort_keys=False, default_flow_style=False)
        write_config_file(output_path, header + body)
        return output_path
