"""Central configuration for experiment_splitter_dspy (no hard-coded values elsewhere)."""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_OLLAMA_BASE_URL = "FC_DSPY_OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "FC_DSPY_OLLAMA_MODEL"

DEFAULT_OLLAMA_BASE_URL = "https://dev.chat.cosy.bio/api"
DEFAULT_OLLAMA_MODEL = "llama4:latest"
COSY_MODEL_CASCADE: tuple[str, ...] = (
    "qwen3.5:122b",
    "gpt-oss:120b",
    "qwen3.6:27b",
    "gemma4:31b",
    "llama4:latest",
)

try:
    from tools.keys_secrets import (
        GROQ_API_KEY,
        GEMINI_API_KEY,
        COHERE_API_KEY,
        MISTRAL_API_KEY,
        COSY_API_KEY,
    )
except ImportError:
    GROQ_API_KEY = ""
    GEMINI_API_KEY = ""
    COHERE_API_KEY = ""
    MISTRAL_API_KEY = ""
    COSY_API_KEY = ""

GROQ_LITELLM_MODEL = "groq/mixtral-8x7b-32768"
GEMINI_LITELLM_MODEL = "gemini/gemini-1.5-pro"
COHERE_LITELLM_MODEL = "cohere/command-r-plus"
MISTRAL_LITELLM_MODEL = "mistral/codestral-latest"

BACKEND_GROQ = "groq"
BACKEND_GEMINI = "gemini"
BACKEND_COHERE = "cohere"
BACKEND_MISTRAL = "mistral"
BACKEND_OLLAMA = "ollama"

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    BACKEND_GROQ: "Groq",
    BACKEND_GEMINI: "Google Gemini",
    BACKEND_COHERE: "Cohere",
    BACKEND_MISTRAL: "Mistral AI",
    BACKEND_OLLAMA: "Ollama (COSY.BIO)",
}

EXPERIMENT_SPLITTER_QUESTION = """Please describe the whole experiment in one message.
Include how it should run (federated, simulation, or centralized), where the
data is, the model, the trainer, and the training settings. You do not need
to use YAML field names."""

AGENT_OKAY_REPLY = "Okay!"

PACKAGE_TOOL_NAME = "tools/prompt_extraction/experiment_splitter_dspy"
PIPELINE_APPROACH = "experiment_splitter_dspy_bootstrap_fewshot"

# Total few-shots in fewshot_examples.py: 20
BOOTSTRAP_MAX_BOOTSTRAPPED_DEMOS = 8
EXPERIMENT_SPLITTER_MAX_RETRIES = 3

RECOMPILE = True
ENV_RECOMPILE = "FC_DSPY_RECOMPILE"
COMPILED_PROGRAM_DIR_NAME = "compiled_agent"
COMPILED_PROGRAM_FILENAME = "compiled_agent.json"


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    gemini_api_key: str
    cohere_api_key: str
    mistral_api_key: str
    cosy_api_key: str
    groq_litellm_model: str
    gemini_litellm_model: str
    cohere_litellm_model: str
    mistral_litellm_model: str
    default_ollama_base_url: str
    default_ollama_model: str
    cosy_model_cascade: tuple[str, ...]
    package_tool_name: str
    pipeline_approach: str
    provider_display_names: dict[str, str]
    bootstrap_max_bootstrapped_demos: int
    experiment_splitter_max_retries: int


def load_settings() -> Settings:
    return Settings(
        groq_api_key=GROQ_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        cohere_api_key=COHERE_API_KEY,
        mistral_api_key=MISTRAL_API_KEY,
        cosy_api_key=COSY_API_KEY,
        groq_litellm_model=GROQ_LITELLM_MODEL,
        gemini_litellm_model=GEMINI_LITELLM_MODEL,
        cohere_litellm_model=COHERE_LITELLM_MODEL,
        mistral_litellm_model=MISTRAL_LITELLM_MODEL,
        default_ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
        default_ollama_model=DEFAULT_OLLAMA_MODEL,
        cosy_model_cascade=COSY_MODEL_CASCADE,
        package_tool_name=PACKAGE_TOOL_NAME,
        pipeline_approach=PIPELINE_APPROACH,
        provider_display_names=dict(PROVIDER_DISPLAY_NAMES),
        bootstrap_max_bootstrapped_demos=BOOTSTRAP_MAX_BOOTSTRAPPED_DEMOS,
        experiment_splitter_max_retries=EXPERIMENT_SPLITTER_MAX_RETRIES,
    )


def resolve_ollama_base_url(explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip().rstrip("/")
    try:
        raw = os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL)
        return str(raw).strip().rstrip("/")
    except Exception as exc:
        raise RuntimeError("Failed to resolve Ollama base URL from environment.") from exc


def resolve_ollama_model(explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    try:
        return str(os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL)).strip()
    except Exception as exc:
        raise RuntimeError("Failed to resolve Ollama model name from environment.") from exc


def resolve_recompile(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(ENV_RECOMPILE)
    if raw is not None and raw.strip():
        return raw.strip().lower() not in ("0", "false", "no")
    return RECOMPILE
