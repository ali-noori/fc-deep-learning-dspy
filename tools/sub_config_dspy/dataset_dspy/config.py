"""Central configuration for dataset_dspy (no hard-coded values elsewhere)."""

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

DATASET_QUESTION_FEDERATED = """Please provide information regarding:
- data_dirs
- train_dataset_file_name
- test_dataset_file_name
- logic_dir

Hint:
data_dir1: [PROJECT_ROOT]/sample_data/c1
data_dir2: [PROJECT_ROOT]/sample_data/c2
data_dir3: ...

train_dataset_file_name: train.npz
test_dataset_file_name: test.npz

logic_dir: data"""

DATASET_QUESTION_CENTRALIZED = """Please provide information regarding:
- data_dir
- train_dataset_file_name
- test_dataset_file_name
- logic_dir

Hint:
data_dir: [PROJECT_ROOT]/sample_data_centralized
logic_dir: data"""

DATASET_QUESTION_SIMULATION = """Please provide information regarding:
- data_dir
- client_dirs
- train_dataset_file_name
- test_dataset_file_name
- logic_dir

Hint:
data_dir: [PROJECT_ROOT]/sample_data_simulation
cleint_dir1: c1
client_dir2: c2
client_dir3: ...

train_dataset_file_name: train.npz
test_dataset_file_name: test.npz

logic_dir: data
"""

AGENT_OKAY_REPLY = "Okay!"

PACKAGE_TOOL_NAME = "tools/sub_config_dspy/dataset_dspy"
PIPELINE_APPROACH = "dataset_dspy_bootstrap_fewshot"
OUTPUT_DIR_NAME = "output"
OUTPUT_FILENAME = "output_dataset_dspy.yml"

# Total few-shots in fewshot_examples.py: 90
# (30 federated + 30 centralized + 30 simulation)
BOOTSTRAP_MAX_BOOTSTRAPPED_DEMOS = 24
DATASET_EXTRACT_MAX_RETRIES = 3

# Default used only if --recompile is not passed on the CLI.
RECOMPILE = True
ENV_RECOMPILE = "FC_DSPY_RECOMPILE"
COMPILED_PROGRAM_DIR_NAME = "compiled_agent"


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
    dataset_extract_max_retries: int


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
        dataset_extract_max_retries=DATASET_EXTRACT_MAX_RETRIES,
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
    False: skip compiling and load the previously saved compiled program."""
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(ENV_RECOMPILE)
    if raw is not None and raw.strip():
        return raw.strip().lower() not in ("0", "false", "no")
    return RECOMPILE
