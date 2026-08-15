"""Write trainer snippet to output_trainer_dspy.yml."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings
from .types import TrainerConfig


class TrainerOutputWriter:
    def __init__(self, settings: Settings, package_dir: Path) -> None:
        self._settings = settings
        self._package_dir = package_dir

    def output_path(self) -> Path:
        return (
            self._package_dir
            / self._settings.output_dir_name
            / self._settings.output_filename
        )

    def build_yaml_dict(self, config: TrainerConfig) -> dict:
        return {
            "trainer": {
                "name": config.name,
                "param": {},
                "local_updates": config.local_updates,
                "data_loader": config.data_loader,
                "optimizer": {
                    "name": config.optimizer_name,
                    "param": {"lr": config.optimizer_lr},
                },
                "loss": {
                    "name": config.loss_name,
                    "param": {},
                },
                "metrics": [
                    {
                        "name": "Accuracy",
                        "package": "torchmetrics.classification",
                        "param": {
                            "task": "multiclass",
                            "num_classes": config.num_classes,
                        },
                    }
                ],
            },
        }

    def write(self, config: TrainerConfig) -> Path:
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_yaml_dict(config)
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        path.write_text(text, encoding="utf-8")
        return path
