"""Write the five intake briefs to YAML and to the JSON handoff file."""

from __future__ import annotations

from pathlib import Path

import yaml

from .briefs import save_briefs
from .config import Settings
from .types import IntakeBriefs


class IntakeOutputWriter:
    def __init__(self, settings: Settings, package_dir: Path) -> None:
        self._settings = settings
        self._package_dir = package_dir

    def output_path(self) -> Path:
        return (
            self._package_dir
            / self._settings.output_dir_name
            / self._settings.output_filename
        )

    def write(self, briefs: IntakeBriefs) -> Path:
        payload = briefs.as_dict()
        save_briefs(payload)
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        path.write_text(text, encoding="utf-8")
        return path
