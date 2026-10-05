"""Central configuration for execution_mode_dspy (no hard-coded values elsewhere)."""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_OLLAMA_BASE_URL = "FC_DSPY_OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "FC_DSPY_OLLAMA_MODEL"

# Open WebUI OpenAI-compatible API (POST {base}/chat/completions).
DEFAULT_OLLAMA_BASE_URL = "https://dev.chat.cosy.bio/api"
DEFAULT_OLLAMA_MODEL = "llama4:latest"
# COSY Open WebUI model names, tried in this order before Mistral/Groq/Gemini/Cohere.
COSY_MODEL_CASCADE: tuple[str, ...] = (
    "qwen3.5:122b",
    "gpt-oss:120b",
    "qwen3.6:27b",
    "gemma4:31b",
    "llama4:latest",
)

# api keys: read from tools/keys_secrets.py (gitignored)
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

# litellm models names for the providers
GROQ_LITELLM_MODEL = "groq/mixtral-8x7b-32768"
GEMINI_LITELLM_MODEL = "gemini/gemini-1.5-pro"
COHERE_LITELLM_MODEL = "cohere/command-r-plus"
MISTRAL_LITELLM_MODEL = "mistral/codestral-latest"


# backends for the providers
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

EXECUTION_MODES: tuple[str, ...] = ("federated", "simulation", "centralized")

MODE_QUESTION = (
    "What mode of app execution do you want? "
    "Please select among: federated, simulation, centralized."
)

AGENT_OKAY_REPLY = "Okay!"

PACKAGE_TOOL_NAME = "tools/execution_mode_dspy"
PIPELINE_APPROACH = "execution_mode_dspy_bootstrap_fewshot"

# max number of bootstrapped demos
# Total few-shots in fewshot_examples.py: 60
BOOTSTRAP_MAX_BOOTSTRAPPED_DEMOS = 4

# max number of retries for the mode selection
MODE_SELECT_MAX_RETRIES = 3

# Default used only if --recompile/--no-recompile is not passed on the CLI.
# True: re-run BootstrapFewShot from scratch and overwrite the saved compiled agent.
# False: skip compiling and load the already saved compiled agent (fails clearly if none exists yet).
RECOMPILE = True

# Where the compiled (optimized) DSPy program is saved/loaded for this agent only.
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
    mode_select_max_retries: int


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
        mode_select_max_retries=MODE_SELECT_MAX_RETRIES,
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
    """True: run BootstrapFewShot and overwrite the saved compiled program.
    False: skip compiling and load the previously saved compiled program.
    Override via explicit arg, else FC_DSPY_RECOMPILE env var, else RECOMPILE above."""
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(ENV_RECOMPILE)
    if raw is not None and raw.strip():
        return raw.strip().lower() not in ("0", "false", "no")
    return RECOMPILE
