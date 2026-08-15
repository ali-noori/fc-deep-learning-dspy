"""Write local_dataset + logic snippet to output_dataset_dspy.yml."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings
from .types import DatasetFederatedConfig
from .types import DatasetCentralizedConfig
from .types import DatasetSimulationConfig


class DatasetOutputWriter:
    def __init__(self, settings: Settings, package_dir: Path) -> None:
        self._settings = settings
        self._package_dir = package_dir

    def output_path(self) -> Path:
        return (
            self._package_dir
            / self._settings.output_dir_name
            / self._settings.output_filename
        )

    def build_yaml_dict_federated(self, config: DatasetFederatedConfig, execution_mode: str) -> dict:
        return {
            "execution_mode": execution_mode,
            "data_dirs": list(config.data_dirs),
            "local_dataset": {
                "train": config.train_dataset_file_name,
                "test": config.test_dataset_file_name,
                "central_test": None,
                "detail": {},
            },
            "logic": {
                "mode": "directory",
                "dir": config.logic_dir,
            },
        }
    def build_yaml_dict_centralized(self, config: DatasetCentralizedConfig, execution_mode: str) -> dict:
        return {
            "execution_mode": execution_mode,
            "data_dir": config.data_dir,
            "centralized": True,  # pyright: ignore[reportUndefinedVariable]
            "local_dataset": {
                "train": config.train_dataset_file_name,
                "test": config.test_dataset_file_name,
                "central_test": None,
                "detail": {},
            },
            "logic": {
                "mode": "directory",
                "dir": config.logic_dir,
            },
        }
    def build_yaml_dict_simulation(self, config: DatasetSimulationConfig, execution_mode: str) -> dict:
        return {
            "execution_mode": execution_mode,
            "data_dir": config.data_dir,
            "simulation": {
                "clients_dir": ", ".join(config.client_dirs),
            },
            "local_dataset": {
                "train": config.train_dataset_file_name,
                "test": config.test_dataset_file_name,
                "central_test": None,
                "detail": {},
            },
            "logic": {
                "mode": "directory",
                "dir": config.logic_dir,
            },
        }

    def write_federated(self, config: DatasetFederatedConfig, execution_mode: str) -> Path:
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_yaml_dict_federated(config, execution_mode)
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        path.write_text(text, encoding="utf-8")
        return path
    
    def write_centralized(self, config: DatasetCentralizedConfig, execution_mode: str) -> Path:
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_yaml_dict_centralized(config, execution_mode)
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        path.write_text(text, encoding="utf-8")
        return path
    
    def write_simulation(self, config: DatasetSimulationConfig, execution_mode: str) -> Path:
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_yaml_dict_simulation(config, execution_mode)
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        path.write_text(text, encoding="utf-8")
        return path