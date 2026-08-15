"""Try Groq → Gemini → Cohere → Mistral → Ollama; register winner on dspy.settings.lm."""

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
    """DSPy LM that tries each backend in order and records the winner for headers."""

    def __init__(self, endpoints: list[BackendEndpoint], ollama_base_url: str) -> None:
        if not endpoints:
            raise RuntimeError("LiteLLMCascadeLM requires at least one backend endpoint.")
        primary = endpoints[0].lm
        super().__init__(
            model=getattr(primary, "model", "fc-config-cascade"),
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


def _cloud_endpoint(
    backend: str,
    litellm_model: str,
    api_key: str,
) -> BackendEndpoint | None:
    if not (api_key or "").strip():
        return None
    return BackendEndpoint(
        backend,
        litellm_model.split("/")[-1],
        dspy.LM(litellm_model, api_key=api_key.strip()),
    )


def build_backend_endpoints(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
) -> list[BackendEndpoint]:
    ollama_lm = dspy.LM(f"ollama/{ollama_model}", api_base=ollama_base_url)
    endpoints: list[BackendEndpoint] = []

    for endpoint in (
        _cloud_endpoint(BACKEND_GROQ, settings.groq_litellm_model, settings.groq_api_key),
        _cloud_endpoint(BACKEND_GEMINI, settings.gemini_litellm_model, settings.gemini_api_key),
        _cloud_endpoint(BACKEND_COHERE, settings.cohere_litellm_model, settings.cohere_api_key),
        _cloud_endpoint(BACKEND_MISTRAL, settings.mistral_litellm_model, settings.mistral_api_key),
    ):
        if endpoint is not None:
            endpoints.append(endpoint)

    endpoints.append(
        BackendEndpoint(BACKEND_OLLAMA, ollama_model, ollama_lm),
    )
    return endpoints


def configure_dspy_cascade(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
) -> LiteLLMCascadeLM:
    endpoints = build_backend_endpoints(settings, ollama_model, ollama_base_url)
    cascade = LiteLLMCascadeLM(endpoints, ollama_base_url)
    dspy.configure(lm=cascade)
    return cascade
