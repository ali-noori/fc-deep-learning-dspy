"""DSPy execution-mode pipeline: bootstrap few-shot, select mode via LM, reply Okay!."""

from __future__ import annotations

from typing import Any

from .config import AGENT_OKAY_REPLY, MODE_QUESTION, Settings, resolve_ollama_base_url, resolve_ollama_model
from .optimizer import compile_module
from .types import ModeSelectionResult


class WorkflowPipeline:
    MODE_QUESTION = MODE_QUESTION

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

    def select_execution_mode(self, user_message: str) -> ModeSelectionResult:
        request = user_message.strip()
        if not request:
            raise ValueError(
                "Please type which mode you want (federated, simulation, or centralized)."
            )

        try:
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
            )
            result = module(user_message=request)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Execution mode pipeline mode selection failed.") from exc

        return ModeSelectionResult(
            execution_mode=result.execution_mode,
            agent_reply=AGENT_OKAY_REPLY,
            user_message=request,
            backend=result.backend,
            model=result.model,
            base_url=result.base_url,
        )

    def run(self, user_message: str) -> dict[str, Any]:
        outcome = self.select_execution_mode(user_message)
        return {
            "approach": self._settings.pipeline_approach,
            "execution_mode": outcome.execution_mode,
            "agent_reply": outcome.agent_reply,
            "backend": outcome.backend,
            "model": outcome.model,
            "base_url": outcome.base_url,
            "user_message": outcome.user_message,
        }
