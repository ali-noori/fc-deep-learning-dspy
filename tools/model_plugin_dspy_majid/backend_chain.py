"""Build ordered list of language-model backends (Groq → … → Ollama)."""

from __future__ import annotations

from typing import Any

from .config import (
    BACKEND_COHERE,
    BACKEND_GEMINI,
    BACKEND_GROQ,
    BACKEND_MISTRAL,
    BACKEND_OLLAMA,
    Settings,
)


def build_backend_chain(
    dspy: Any,
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
) -> list[tuple[str, str, Any]]:
    """
    The models in the order of the backend chain
    """
    try:
        ollama_lm = dspy.LM(f"ollama/{ollama_model}", api_base=ollama_base_url)
        return [
            (
                BACKEND_GROQ,
                settings.groq_litellm_model.split("/")[-1],
                dspy.LM(settings.groq_litellm_model, api_key=settings.groq_api_key),
            ),
            (
                BACKEND_GEMINI,
                settings.gemini_litellm_model.split("/")[-1],
                dspy.LM(settings.gemini_litellm_model, api_key=settings.gemini_api_key),
            ),
            (
                BACKEND_COHERE,
                settings.cohere_litellm_model.split("/")[-1],
                dspy.LM(settings.cohere_litellm_model, api_key=settings.cohere_api_key),
            ),
            (
                BACKEND_MISTRAL,
                settings.mistral_litellm_model.split("/")[-1],
                dspy.LM(settings.mistral_litellm_model, api_key=settings.mistral_api_key),
            ),
            (BACKEND_OLLAMA, ollama_model, ollama_lm),
        ]
    except Exception as exc:
        raise RuntimeError("Failed to construct DSPy LM backend chain.") from exc
