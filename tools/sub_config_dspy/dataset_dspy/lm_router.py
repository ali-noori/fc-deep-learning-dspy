"""Try remote Ollama (llama4) → Groq → Gemini → Cohere → Mistral; register winner on dspy.settings.lm."""

from __future__ import annotations

import concurrent.futures
import sys
from dataclasses import dataclass
from typing import Any

# Seconds to wait for one backend before trying the next one.
# Set to None to restore the old wait-forever behavior.
LLM_CALL_TIMEOUT_S = 180

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
    def __init__(self, endpoints: list[BackendEndpoint], ollama_base_url: str) -> None:
        if not endpoints:
            raise RuntimeError("LiteLLMCascadeLM requires at least one backend endpoint.")
        primary = endpoints[0].lm
        super().__init__(
            model=getattr(primary, "model", "fc-dataset-cascade"),
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
        self._logged_failures: set[str] = set()

    def forward(self, prompt=None, messages=None, **kwargs):
        last_error: BaseException | None = None
        for endpoint in self._endpoints:
            try:
                if LLM_CALL_TIMEOUT_S is None:
                    response = endpoint.lm.forward(
                        prompt=prompt, messages=messages, **kwargs
                    )
                else:
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        future = pool.submit(
                            endpoint.lm.forward,
                            prompt=prompt,
                            messages=messages,
                            **kwargs,
                        )
                        response = future.result(timeout=LLM_CALL_TIMEOUT_S)
                    finally:
                        # Do not wait for a hung call; otherwise we never reach the next backend.
                        pool.shutdown(wait=False)
                self.last_backend = endpoint.backend
                self.last_model = endpoint.model_label
                self.last_base_url = (
                    self._ollama_base_url if endpoint.backend == BACKEND_OLLAMA else None
                )
                return response
            except BaseException as exc:
                last_error = exc
                skip_key = f"{endpoint.backend}/{endpoint.model_label}"
                if skip_key not in self._logged_failures:
                    self._logged_failures.add(skip_key)
                    print(
                        f"(skipped {skip_key}: {exc})",
                        file=sys.stderr,
                    )
                continue
        raise RuntimeError(
            "All language-model backends failed "
            "(Mistral, then COSY qwen3.5:122b, gpt-oss:120b, qwen3.6:27b, gemma4:31b, llama4:latest, "
            "then Groq, Gemini, Cohere). "
            f"Last error: {last_error!r}"
        ) from last_error


def build_backend_endpoints(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
) -> list[BackendEndpoint]:
    _ = ollama_model
    endpoints: list[BackendEndpoint] = [
        BackendEndpoint(
            BACKEND_MISTRAL,
            settings.mistral_litellm_model.split("/")[-1],
            dspy.LM(settings.mistral_litellm_model, api_key=settings.mistral_api_key),
        ),
    ]
    for cosy_name in settings.cosy_model_cascade:
        endpoints.append(
            BackendEndpoint(
                BACKEND_OLLAMA,
                cosy_name,
                dspy.LM(
                    f"openai/{cosy_name}",
                    api_base=ollama_base_url,
                    api_key=settings.cosy_api_key,
                ),
            )
        )
    endpoints.extend(
        [
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
        ]
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
