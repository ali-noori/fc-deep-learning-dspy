"""Try Groq → Gemini → Cohere → Mistral → Ollama; register winner path on dspy.settings.lm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
    BACKEND_COHERE,
    BACKEND_GEMINI,
    BACKEND_GROQ,
    BACKEND_MISTRAL,
    BACKEND_OLLAMA,
    Settings,
)
from .dspy_support import dspy


@dataclass(frozen=True)
class BackendEndpoint:
    backend: str
    model_label: str
    lm: Any


class LiteLLMCascadeLM(dspy.BaseLM):
    """
    DSPy language model that tries each cloud/local backend in order.

    On success, sets last_backend / last_model / last_base_url for file headers.
    """

    def __init__(self, endpoints: list[BackendEndpoint], ollama_base_url: str) -> None:
        if not endpoints:
            raise RuntimeError("LiteLLMCascadeLM requires at least one backend endpoint.")
        primary = endpoints[0].lm
        super().__init__(
            model=getattr(primary, "model", "fc-architecture-cascade"),
            model_type=getattr(primary, "model_type", "chat"),
            temperature=getattr(primary, "kwargs", {}).get("temperature", 0.0),
            max_tokens=getattr(primary, "kwargs", {}).get("max_tokens", 4096),
            cache=getattr(primary, "cache", True),
        )
        self._endpoints = endpoints
        self._ollama_base_url = ollama_base_url
        self.last_backend: str | None = None
        self.last_model: str | None = None
        self.last_base_url: str | None = None

    def forward(self, prompt=None, messages=None, **kwargs):
        last_error: BaseException | None = None
        for endpoint in self._endpoints:
            try:
                response = endpoint.lm.forward(
                    prompt=prompt, messages=messages, **kwargs
                )
                self.last_backend = endpoint.backend
                self.last_model = endpoint.model_label
                self.last_base_url = (
                    self._ollama_base_url if endpoint.backend == BACKEND_OLLAMA else None
                )
                return response
            except BaseException as exc:
                last_error = exc
                continue
        raise RuntimeError(
            "All language-model backends failed (Groq, Gemini, Cohere, Mistral, Ollama). "
            f"Last error: {last_error!r}"
        ) from last_error

# TO create the list of candidate models
def build_backend_endpoints(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
) -> list[BackendEndpoint]:
    ollama_lm = dspy.LM(f"ollama/{ollama_model}", api_base=ollama_base_url)
    return [
        BackendEndpoint(
            BACKEND_GROQ,
            settings.groq_litellm_model.split("/")[-1],
            dspy.LM(settings.groq_litellm_model, api_key=settings.groq_api_key),
        ),
        BackendEndpoint(
            BACKEND_GEMINI,
            settings.gemini_litellm_model.split("/")[-1],
            dspy.LM(settings.gemini_litellm_model, api_key=settings.gemini_api_key),
        ),
        BackendEndpoint(
            BACKEND_COHERE,
            settings.cohere_litellm_model.split("/")[-1],
            dspy.LM(settings.cohere_litellm_model, api_key=settings.cohere_api_key),
        ),
        BackendEndpoint(
            BACKEND_MISTRAL,
            settings.mistral_litellm_model.split("/")[-1],
            dspy.LM(settings.mistral_litellm_model, api_key=settings.mistral_api_key),
        ),
        BackendEndpoint(
            BACKEND_OLLAMA,
            ollama_model,
            ollama_lm,
        ),
    ]


def configure_dspy_cascade(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
) -> LiteLLMCascadeLM:
    """Build cascade LM and set it as the global DSPy LM."""
    try:
        # Build the list of candidate models
        endpoints = build_backend_endpoints(settings, ollama_model, ollama_base_url)
        
        
        cascade = LiteLLMCascadeLM(endpoints, ollama_base_url)
        dspy.configure(lm=cascade)
        return cascade
    except Exception as exc:
        raise RuntimeError("Failed to configure DSPy with LiteLLM provider cascade.") from exc
