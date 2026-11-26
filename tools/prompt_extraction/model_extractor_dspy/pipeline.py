"""DSPy model extractor pipeline: one request, four copied fields."""

from __future__ import annotations

from typing import Any

from .config import (
    AGENT_OKAY_REPLY,
    MODEL_EXTRACTOR_QUESTION,
    Settings,
    resolve_ollama_base_url,
    resolve_ollama_model,
)
from .optimizer import compile_module
from .output_writer import ModelExtractorOutputWriter
from .types import ModelExtractorResult


class ModelExtractorPipeline:
    MODEL_EXTRACTOR_QUESTION = MODEL_EXTRACTOR_QUESTION

    def __init__(
        self,
        settings: Settings,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        recompile: bool | None = None,
    ) -> None:
        self._settings = settings
        self._ollama_model_override = ollama_model
        self._ollama_base_url_override = ollama_base_url
        self._recompile = recompile

    def _ollama_model(self) -> str:
        return resolve_ollama_model(self._ollama_model_override)

    def _ollama_base_url(self) -> str:
        return resolve_ollama_base_url(self._ollama_base_url_override)

    def extract_briefs(self, user_message: str) -> ModelExtractorResult:
        request = user_message.strip()
        if not request:
            raise ValueError("Please describe the model before continuing.")

        try:
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
                self._recompile,
            )
            result = module(user_message=request)
            output_path = ModelExtractorOutputWriter().write(result.briefs)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Model extractor pipeline extraction failed.") from exc

        return ModelExtractorResult(
            briefs=result.briefs,
            agent_reply=AGENT_OKAY_REPLY,
            user_message=request,
            backend=result.backend,
            model=result.model,
            base_url=result.base_url,
            output_path=str(output_path),
        )

    def run(self, user_message: str) -> dict[str, Any]:
        outcome = self.extract_briefs(user_message)
        payload = outcome.briefs.as_dict()
        payload.update(
            {
                "approach": self._settings.pipeline_approach,
                "agent_reply": outcome.agent_reply,
                "backend": outcome.backend,
                "model": outcome.model,
                "base_url": outcome.base_url,
                "user_message": outcome.user_message,
                "output_path": outcome.output_path,
            }
        )
        return payload
