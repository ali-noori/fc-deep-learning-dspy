"""Prepare and run FeatureCloud tests from DSPy pipeline artifacts."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MODES = ("federated", "centralized", "simulation")

_DEFAULT_APP_IMAGE = "featurecloud.ai/fc_deep_networks1"
_DEFAULT_CONTROLLER_HOST = "http://localhost:8000"

# Staged under repo data/; FeatureCloud controller mounts host data/ as /data.
_STAGING_REL = Path("data") / "tools" / "config"
_FEATURECLOUD_GENERIC_DIR = "tools/config"

# Nested keys under fc_deep that may hold plugins/.../*.py references.
_PLUGIN_FIELD_PATHS: tuple[tuple[str, ...], ...] = (
    ("fed_hyper_params", "federated_model"),
    ("trainer", "name"),
    ("trainer", "data_loader"),
    ("trainer", "loss", "name"),
    ("model", "name"),
)


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved paths and flags for one featurecloud test start."""

    mode: str
    client_dirs: str
    generated_config_path: Path
    generated_config_rel: str
    featurecloud_generic_dir: str


class ConfigExecutor:
    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        app_image: str = _DEFAULT_APP_IMAGE,
        controller_host: str = _DEFAULT_CONTROLLER_HOST,
        keep_staging: bool = False,
    ) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.app_image = app_image
        self.controller_host = controller_host
        self.keep_staging = keep_staging

    @property
    def dataset_output_path(self) -> Path:
        return (
            self.repo_root
            / "tools"
            / "sub_config_dspy"
            / "dataset_dspy"
            / "output"
            / "output_dataset_dspy.yml"
        )

    @property
    def staging_dir(self) -> Path:
        return self.repo_root / _STAGING_REL

    @property
    def staged_config_path(self) -> Path:
        return self.staging_dir / "config.yml"

    def source_config_path(self, mode: str) -> Path:
        if mode not in MODES:
            raise ValueError(f"Unsupported execution mode: {mode!r}")
        path = (
            self.repo_root
            / "tools"
            / "config_builder"
            / "output"
            / mode
            / f"{mode}_config.yml"
        )
        if not path.exists():
            raise ValueError(
                f"Generated config not found: {path}. "
                "Run the mode config agent and config_builder first."
            )
        return path

    def load_dataset(self) -> dict[str, Any]:
        path = self.dataset_output_path
        if not path.exists():
            raise ValueError(f"Dataset output file not found: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Dataset output file must be a YAML mapping: {path}")
        return data

    def _assert_execution_mode(self, cli_mode: str, dataset: dict[str, Any]) -> str:
        if "execution_mode" not in dataset:
            raise ValueError(
                f"Dataset output file does not contain 'execution_mode': "
                f"{self.dataset_output_path}"
            )

        file_mode_raw = dataset["execution_mode"]
        if not isinstance(file_mode_raw, str) or not file_mode_raw.strip():
            raise ValueError(
                f"Dataset 'execution_mode' must be a non-empty string: "
                f"{self.dataset_output_path}"
            )

        cli_mode = cli_mode.strip().lower()
        file_mode = file_mode_raw.strip().lower()
        if file_mode != cli_mode:
            raise ValueError(
                f"Incompatible execution mode: CLI requested {cli_mode!r} "
                f"but dataset file contains execution_mode: {file_mode!r} "
                f"({self.dataset_output_path})"
            )

        if cli_mode not in MODES:
            raise ValueError(f"Unsupported execution mode: {cli_mode!r}")
        return cli_mode

    def resolve_client_dirs(self, mode: str, dataset: dict[str, Any]) -> str:
        if mode == "federated":
            raw_dirs = dataset.get("data_dirs")
            if not isinstance(raw_dirs, list) or not raw_dirs:
                raise ValueError(
                    f"Dataset 'data_dirs' must be a non-empty list: "
                    f"{self.dataset_output_path}"
                )
            paths = [
                str(p).strip().replace("\\", "/")
                for p in raw_dirs
                if str(p).strip()
            ]
            if not paths:
                raise ValueError(
                    f"Dataset 'data_dirs' list is empty: "
                    f"{self.dataset_output_path}"
                )
            return ",".join(paths)

        if mode in ("centralized", "simulation"):
            raw_dir = dataset.get("data_dir")
            if not isinstance(raw_dir, str) or not raw_dir.strip():
                raise ValueError(
                    f"Dataset 'data_dir' must be a non-empty string: "
                    f"{self.dataset_output_path}"
                )
            return raw_dir.strip().replace("\\", "/").strip("/")

        raise ValueError(f"Unsupported execution mode: {mode!r}")

    @staticmethod
    def _is_plugin_reference(value: object) -> bool:
        if not isinstance(value, str):
            return False
        text = value.strip()
        return text.startswith("plugins/") and text.endswith(".py")

    @staticmethod
    def _get_nested(mapping: dict[str, Any], keys: tuple[str, ...]) -> object | None:
        current: object = mapping
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    @staticmethod
    def _set_nested(mapping: dict[str, Any], keys: tuple[str, ...], value: str) -> None:
        current = mapping
        for key in keys[:-1]:
            nested = current.get(key)
            if not isinstance(nested, dict):
                return
            current = nested
        if keys[-1] in current:
            current[keys[-1]] = value

    def _stage_plugin_file(self, plugin_ref: str, staging_dir: Path) -> str:
        source = self.repo_root / Path(plugin_ref)
        if not source.is_file():
            raise ValueError(f"Plugin file not found: {source}")
        filename = source.name
        shutil.copy2(source, staging_dir / filename)
        return filename

    def _resolve_plugin_fields(self, fc_deep: dict[str, Any], staging_dir: Path) -> None:
        for keys in _PLUGIN_FIELD_PATHS:
            value = self._get_nested(fc_deep, keys)
            if not self._is_plugin_reference(value):
                continue
            plugin_ref = str(value).strip()
            filename = self._stage_plugin_file(plugin_ref, staging_dir)
            self._set_nested(fc_deep, keys, filename)

    def stage_config_for_featurecloud(self, mode: str) -> Path:
        """Copy mode config to data/tools/config and resolve plugin references."""
        source = self.source_config_path(mode)
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Generated config must be a YAML mapping: {source}")

        fc_deep = config.get("fc_deep")
        if not isinstance(fc_deep, dict):
            raise ValueError(f"Generated config must contain fc_deep mapping: {source}")

        staging_dir = self.staging_dir
        staging_dir.mkdir(parents=True, exist_ok=True)
        self._resolve_plugin_fields(fc_deep, staging_dir)

        header = f"# execution mode: {mode}\n"
        body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
        staged_path = self.staged_config_path
        staged_path.write_text(header + body, encoding="utf-8")
        return staged_path

    def clear_staging_dir(self) -> None:
        """Remove all staged files from data/tools/config."""
        staging_dir = self.staging_dir
        if not staging_dir.exists():
            return

        for path in staging_dir.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

        rel = staging_dir.relative_to(self.repo_root).as_posix()
        print(f"ConfigExecutor: cleared {rel}")

    def prepare(self, cli_execution_mode: str) -> ExecutionPlan:
        dataset = self.load_dataset()
        mode = self._assert_execution_mode(cli_execution_mode, dataset)
        client_dirs = self.resolve_client_dirs(mode, dataset)
        generated = self.stage_config_for_featurecloud(mode)

        return ExecutionPlan(
            mode=mode,
            client_dirs=client_dirs,
            generated_config_path=generated,
            generated_config_rel=generated.relative_to(self.repo_root).as_posix(),
            featurecloud_generic_dir=_FEATURECLOUD_GENERIC_DIR,
        )

    def is_config_file_available(self, plan: ExecutionPlan) -> None:
        """Ensure staged config.yml exists before running FeatureCloud."""
        if not plan.generated_config_path.exists():
            raise ValueError(
                f"Staged config not found: {plan.generated_config_path}"
            )

    def build_featurecloud_command(self, plan: ExecutionPlan) -> list[str]:
        return [
            "featurecloud",
            "test",
            "start",
            "--app-image",
            self.app_image,
            "--client-dirs",
            plan.client_dirs,
            "--generic-dir",
            plan.featurecloud_generic_dir,
            "--controller-host",
            self.controller_host,
        ]

    def run_featurecloud(self, plan: ExecutionPlan) -> int:
        command = self.build_featurecloud_command(plan)
        print("ConfigExecutor: running:")
        print(" ", " ".join(command))

        print(f"\nMode: {plan.mode}")
        print("Command:")
        print(" ".join(command))
        answer = input("\nAre you sure you want to run this command? (yes/no) ").strip().lower()
        if answer != "yes":
            print("ConfigExecutor: cancelled.")
            return 0

        completed = subprocess.run(command, cwd=self.repo_root, check=False)
        exit_code = int(completed.returncode or 0)
        if exit_code == 0 and not self.keep_staging:
            self.clear_staging_dir()
        return exit_code

    def execute(self, cli_execution_mode: str) -> int:
        plan = self.prepare(cli_execution_mode)
        self.is_config_file_available(plan)

        print(f"ConfigExecutor: mode={plan.mode}")
        print(f"ConfigExecutor: client-dirs={plan.client_dirs!r}")
        print(f"ConfigExecutor: generated-config={plan.generated_config_rel!r}")
        print(
            f"ConfigExecutor: featurecloud --generic-dir="
            f"{plan.featurecloud_generic_dir!r}"
        )

        return self.run_featurecloud(plan)
