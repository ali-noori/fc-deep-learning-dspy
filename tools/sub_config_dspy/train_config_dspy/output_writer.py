"""Write train_config snippet to output_train_config_dspy.yml."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings
from .types import TrainConfig


class TrainConfigOutputWriter:
    def __init__(self, settings: Settings, package_dir: Path) -> None:
        self._settings = settings
        self._package_dir = package_dir

    def output_path(self) -> Path:
        return (
            self._package_dir
            / self._settings.output_dir_name
            / self._settings.output_filename
        )

    def build_yaml_dict(self, config: TrainConfig) -> dict:
        return {
            "train_config": {
                "verbose": config.verbose,
                "batch_size": config.batch_size,
                "test_batch_size": config.test_batch_size,
                "epochs": config.epochs,
                "lr": config.lr,
                "batch_count": config.batch_count,
                "device": config.device,
            },
        }

    def write(self, config: TrainConfig) -> Path:
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_yaml_dict(config)
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        path.write_text(text, encoding="utf-8")
        return path
