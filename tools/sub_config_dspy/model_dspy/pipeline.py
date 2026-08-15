"""DSPy model config pipeline: bootstrap few-shot, extract fields, write YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    AGENT_OKAY_REPLY,
    MODEL_QUESTION,
    Settings,
    resolve_ollama_base_url,
    resolve_ollama_model,
)
from .optimizer import compile_module
from .output_writer import ModelOutputWriter
from .types import ModelConfig, ModelConfigResult

_PACKAGE_DIR = Path(__file__).resolve().parent


class ModelPipeline:
    MODEL_QUESTION = MODEL_QUESTION

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

    def extract_model_config(
        self,
        user_message: str,
        model_artifacts: dict[str, str] | None = None,
    ) -> ModelConfigResult:
        request = user_message.strip()
        if not request:
            raise ValueError(
                "Please provide name, n_class (or n_classes), and in_features."
            )

        try:
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
                model_artifacts,
            )
            result = module(user_message=request)

            config = ModelConfig(
                name=result.name,
                n_classes=result.n_classes,
                in_features=result.in_features,
            )
            writer = ModelOutputWriter(self._settings, _PACKAGE_DIR)
            output_path = writer.write(config)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Model config pipeline extraction failed.") from exc

        return ModelConfigResult(
            name=result.name,
            n_classes=result.n_classes,
            in_features=result.in_features,
            agent_reply=AGENT_OKAY_REPLY,
            user_message=request,
            backend=result.backend,
            model=result.model,
            base_url=result.base_url,
            output_path=str(output_path),
        )

    def run(
        self,
        user_message: str,
        model_artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        outcome = self.extract_model_config(user_message, model_artifacts)
        return {
            "approach": self._settings.pipeline_approach,
            "name": outcome.name,
            "n_classes": outcome.n_classes,
            "in_features": outcome.in_features,
            "agent_reply": outcome.agent_reply,
            "backend": outcome.backend,
            "model": outcome.model,
            "base_url": outcome.base_url,
            "user_message": outcome.user_message,
            "output_path": outcome.output_path,
        }
