"""DSPy hyperparameter pipeline: bootstrap few-shot, extract fields, write YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    AGENT_OKAY_REPLY,
    HYPER_PARAMS_QUESTION,
    Settings,
    resolve_ollama_base_url,
    resolve_ollama_model,
)
from .optimizer import compile_module
from .output_writer import HyperParamsOutputWriter
from .types import FedHyperParamsConfig, FedHyperParamsConfigResult

_PACKAGE_DIR = Path(__file__).resolve().parent


class HyperParamsPipeline:
    HYPER_PARAMS_QUESTION = HYPER_PARAMS_QUESTION

    def __init__(
        self,
        settings: Settings,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self._settings = settings
        self._ollama_model_override = ollama_model
        self._ollama_base_url_override = ollama_base_url

    def _ollama_model(self) -> str:
        return resolve_ollama_model(self._ollama_model_override)

    def _ollama_base_url(self) -> str:
        return resolve_ollama_base_url(self._ollama_base_url_override)

    def extract_hyper_params_config(
        self,
        user_message: str,
        aggregator_artifacts: dict[str, str] | None = None,
    ) -> FedHyperParamsConfigResult:
        request = user_message.strip()
        if not request:
            raise ValueError(
                "Please provide max_iter, n_class (or n_classes), and federated_model."
            )

        try:
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
                aggregator_artifacts or {},
            )
            result = module(user_message=request)

            config = FedHyperParamsConfig(
                max_iter=result.max_iter,
                n_classes=result.n_classes,
                federated_model=result.federated_model,
                global_updates=self._settings.default_global_updates,
            )
            writer = HyperParamsOutputWriter(self._settings, _PACKAGE_DIR)
            output_path = writer.write(config)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Hyperparameter config pipeline extraction failed."
            ) from exc

        return FedHyperParamsConfigResult(
            max_iter=result.max_iter,
            n_classes=result.n_classes,
            federated_model=result.federated_model,
            global_updates=config.global_updates,
            agent_reply=AGENT_OKAY_REPLY,
            user_message=request,
            backend=result.backend,
            model=result.model,
            base_url=result.base_url,
            output_path=str(output_path),
        )

    def run(self, user_message: str, aggregator_artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        outcome = self.extract_hyper_params_config(
            user_message,
            aggregator_artifacts,
        )
        return {
            "approach": self._settings.pipeline_approach,
            "max_iter": outcome.max_iter,
            "n_classes": outcome.n_classes,
            "federated_model": outcome.federated_model,
            "global_updates": outcome.global_updates,
            "agent_reply": outcome.agent_reply,
            "backend": outcome.backend,
            "model": outcome.model,
            "base_url": outcome.base_url,
            "user_message": outcome.user_message,
            "output_path": outcome.output_path,
        }
