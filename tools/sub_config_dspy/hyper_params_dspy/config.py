"""Central configuration for hyper_params_dspy (no hard-coded values elsewhere)."""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_OLLAMA_BASE_URL = "FC_DSPY_OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "FC_DSPY_OLLAMA_MODEL"

# Open WebUI OpenAI-compatible API (POST {base}/chat/completions).
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

HYPER_PARAMS_QUESTION = """Please enter the parameter regarding the federated hyper parameters:
- max_iter
- n_class (or n_classes)
- federated_model

Built-in aggregator example:
max_iter: 10
n_class: 10
federated_model: FedAvg

Custom plugin aggregator example (use the custom aggregator):
max_iter: 10000
n_classes: 50
federated_model: FedAvg.py

IMPORTANT: Copy values exactly from your answer. Use FedAvg for built-in
aggregators; use FedAvg.py (or full plugin path) for custom plugins."""

AGENT_OKAY_REPLY = "Okay!"

# Built-in aggregators from utils.pytorch.optimizer (via utils.get_aggregator)
BUILTIN_AGGREGATORS: tuple[str, ...] = ("FedAvg",)

DEFAULT_MAX_ITER = 10
DEFAULT_GLOBAL_UPDATES = "WEIGHTS_STOPPING"

PACKAGE_TOOL_NAME = "tools/sub_config_dspy/hyper_params_dspy"
PIPELINE_APPROACH = "hyper_params_dspy_bootstrap_fewshot"
OUTPUT_DIR_NAME = "output"
OUTPUT_FILENAME = "output_fed_hyper_param.yml"

# Total few-shots in fewshot_examples.py: 40
BOOTSTRAP_MAX_BOOTSTRAPPED_DEMOS = 20
HYPER_PARAMS_EXTRACT_MAX_RETRIES = 3

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
    output_dir_name: str
    output_filename: str
    bootstrap_max_bootstrapped_demos: int
    hyper_params_extract_max_retries: int
    default_max_iter: int
    default_global_updates: str


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
        output_dir_name=OUTPUT_DIR_NAME,
        output_filename=OUTPUT_FILENAME,
        bootstrap_max_bootstrapped_demos=BOOTSTRAP_MAX_BOOTSTRAPPED_DEMOS,
        hyper_params_extract_max_retries=HYPER_PARAMS_EXTRACT_MAX_RETRIES,
        default_max_iter=DEFAULT_MAX_ITER,
        default_global_updates=DEFAULT_GLOBAL_UPDATES,
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
