"""DSPy config pipeline: bootstrap few-shot, extract intent, merge template, write YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from .artifact_discovery import discover_repo_artifacts
from .config import Settings, resolve_ollama_base_url, resolve_ollama_model
from .config_merger import merge_intent_into_template
from .generated_config_writer import GeneratedConfigWriter
from .optimizer import compile_module
from .template_loader import load_template_config
from .types import PipelineRunResult


class ConfigPluginPipeline:
    def __init__(
        self,
        repo_root: Path,
        settings: Settings,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        *,
        skip_bootstrap: bool = False,
    ) -> None:
        self._repo_root = repo_root
        self._settings = settings
        self._ollama_model_override = ollama_model
        self._ollama_base_url_override = ollama_base_url
        self._skip_bootstrap = skip_bootstrap
        self._artifacts = discover_repo_artifacts(repo_root)

    def _ollama_model(self) -> str:
        return resolve_ollama_model(self._ollama_model_override)

    def _ollama_base_url(self) -> str:
        return resolve_ollama_base_url(self._ollama_base_url_override)

    def _known_strings(self) -> dict[str, str]:
        return {key: str(self._artifacts.get(key, [])) for key in (
            "models",
            "trainers",
            "dataloaders",
            "aggregators",
            "losses",
            "optimizers",
            "devices",
        )}

    def run(self, request: str, output_dir: Path | None = None) -> PipelineRunResult:
        text = request.strip()
        if not text:
            raise ValueError("request must be non-empty")

        module = compile_module(
            self._settings,
            self._artifacts,
            self._ollama_model(),
            self._ollama_base_url(),
            skip_bootstrap=self._skip_bootstrap,
        )
        known = self._known_strings()
        extraction = module(
            user_request=text,
            known_models=known["models"],
            known_trainers=known["trainers"],
            known_dataloaders=known["dataloaders"],
            known_aggregators=known["aggregators"],
            known_losses=known["losses"],
            known_optimizers=known["optimizers"],
            known_devices=known["devices"],
        )

        template = load_template_config(self._repo_root, self._settings)
        plan = merge_intent_into_template(template, extraction.intent, self._artifacts)
        config_yaml = yaml.safe_dump(plan.config_data, sort_keys=False)

        writer = GeneratedConfigWriter(self._settings, self._repo_root)
        custom_path = None
        if output_dir is not None:
            custom_path = Path(output_dir) / self._settings.output_filename
        output_path = writer.write(
            plan.config_data,
            extraction.backend,
            extraction.model,
            extraction.base_url,
            output_path=custom_path,
        )

        return PipelineRunResult(
            approach=self._settings.pipeline_approach,
            template=self._settings.template_filename,
            backend=extraction.backend,
            model=extraction.model,
            base_url=extraction.base_url,
            output_config=str(output_path),
            config_yaml=config_yaml,
            intent=extraction.intent,
            output_files={"config": str(output_path)},
        )
