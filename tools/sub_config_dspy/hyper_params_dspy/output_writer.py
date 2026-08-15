"""Write fed_hyper_params snippet to output_fed_hyper_param.yml."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings
from .types import FedHyperParamsConfig


class HyperParamsOutputWriter:
    def __init__(self, settings: Settings, package_dir: Path) -> None:
        self._settings = settings
        self._package_dir = package_dir

    def output_path(self) -> Path:
        return (
            self._package_dir
            / self._settings.output_dir_name
            / self._settings.output_filename
        )

    def build_yaml_dict(self, config: FedHyperParamsConfig) -> dict:
        return {
            "fed_hyper_params": {
                "max_iter": config.max_iter,
                "n_classes": config.n_classes,
                "federated_model": config.federated_model,
                "global_updates": config.global_updates,
                "param": {},
            },
        }

    def write(self, config: FedHyperParamsConfig) -> Path:
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_yaml_dict(config)
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        path.write_text(text, encoding="utf-8")
        return path
