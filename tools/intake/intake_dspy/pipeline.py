"""DSPy intake pipeline: one request, five briefs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    AGENT_OKAY_REPLY,
    INTAKE_QUESTION,
    Settings,
    resolve_ollama_base_url,
    resolve_ollama_model,
)
from .optimizer import compile_module
from .output_writer import IntakeOutputWriter
from .types import IntakeResult

_PACKAGE_DIR = Path(__file__).resolve().parent


class IntakePipeline:
    INTAKE_QUESTION = INTAKE_QUESTION

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

    def extract_briefs(self, user_message: str) -> IntakeResult:
        request = user_message.strip()
        if not request:
            raise ValueError("Please describe the experiment before continuing.")

        try:
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
                self._recompile,
            )
            result = module(user_message=request)
            output_path = IntakeOutputWriter(self._settings, _PACKAGE_DIR).write(result.briefs)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Intake pipeline extraction failed.") from exc

        return IntakeResult(
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
