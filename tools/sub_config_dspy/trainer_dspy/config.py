"""Central configuration for trainer_dspy (no hard-coded values elsewhere)."""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_OLLAMA_BASE_URL = "FC_DSPY_OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "FC_DSPY_OLLAMA_MODEL"

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1"

try:
    from tools.keys_secrets import (
        GROQ_API_KEY,
        GEMINI_API_KEY,
        COHERE_API_KEY,
        MISTRAL_API_KEY,
    )
except ImportError:
    GROQ_API_KEY = ""
    GEMINI_API_KEY = ""
    COHERE_API_KEY = ""
    MISTRAL_API_KEY = ""

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
    BACKEND_OLLAMA: "Ollama (local)",
}

TRAINER_QUESTION = """Please enter the parameter regarding the trainer:
- name
- data_loader
- loss.name (or loss_name)
- n_class (or n_classes)

Built-in example:
name: BasicTrainer
data_loader: ImageLoader
loss.name: CrossEntropyLoss
n_class: 10

Custom plugin example:
name: FedMMb.py
data_loader: ImageLoader.py
loss.name: focal_loss.py
n_class: 10

IMPORTANT: Copy values exactly from your answer. Use built-in names without .py
(BasicTrainer, ImageLoader, CrossEntropyLoss). Plugin fields resolve to
repo-relative paths (e.g. plugins/trainers/FedMMb.py) in the output config."""

AGENT_OKAY_REPLY = "Okay!"

# Built-in trainer from utils.pytorch.DeepModel (via utils.get_trainer)
BUILTIN_TRAINERS: tuple[str, ...] = ("BasicTrainer",)

# Built-in dataloader from utils.pytorch.DataLoader (via utils.get_dataloader)
BUILTIN_DATALOADERS: tuple[str, ...] = ("ImageLoader",)

# Common torch.nn loss classes (via utils.get_loss_func)
BUILTIN_LOSSES: tuple[str, ...] = (
    "CrossEntropyLoss",
    "NLLLoss",
    "MSELoss",
    "BCEWithLogitsLoss",
)

DEFAULT_LOCAL_UPDATES = "WEIGHTS_N_SAMPLES"
DEFAULT_OPTIMIZER_NAME = "SGD"
DEFAULT_OPTIMIZER_LR = 0.1
DEFAULT_METRIC_NAME = "Accuracy"
DEFAULT_METRIC_PACKAGE = "torchmetrics.classification"
DEFAULT_METRIC_TASK = "multiclass"

PACKAGE_TOOL_NAME = "tools/sub_config_dspy/trainer_dspy"
PIPELINE_APPROACH = "trainer_dspy_bootstrap_fewshot"
OUTPUT_DIR_NAME = "output"
OUTPUT_FILENAME = "output_trainer_dspy.yml"

BOOTSTRAP_MAX_BOOTSTRAPPED_DEMOS = 30
TRAINER_EXTRACT_MAX_RETRIES = 3


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    gemini_api_key: str
    cohere_api_key: str
    mistral_api_key: str
    groq_litellm_model: str
    gemini_litellm_model: str
    cohere_litellm_model: str
    mistral_litellm_model: str
    default_ollama_base_url: str
    default_ollama_model: str
    package_tool_name: str
    pipeline_approach: str
    provider_display_names: dict[str, str]
    output_dir_name: str
    output_filename: str
    bootstrap_max_bootstrapped_demos: int
    trainer_extract_max_retries: int
    default_local_updates: str
    default_optimizer_name: str
    default_optimizer_lr: float
    default_metric_name: str
    default_metric_package: str
    default_metric_task: str


def load_settings() -> Settings:
    return Settings(
        groq_api_key=GROQ_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        cohere_api_key=COHERE_API_KEY,
        mistral_api_key=MISTRAL_API_KEY,
        groq_litellm_model=GROQ_LITELLM_MODEL,
        gemini_litellm_model=GEMINI_LITELLM_MODEL,
        cohere_litellm_model=COHERE_LITELLM_MODEL,
        mistral_litellm_model=MISTRAL_LITELLM_MODEL,
        default_ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
        default_ollama_model=DEFAULT_OLLAMA_MODEL,
        package_tool_name=PACKAGE_TOOL_NAME,
        pipeline_approach=PIPELINE_APPROACH,
        provider_display_names=dict(PROVIDER_DISPLAY_NAMES),
        output_dir_name=OUTPUT_DIR_NAME,
        output_filename=OUTPUT_FILENAME,
        bootstrap_max_bootstrapped_demos=BOOTSTRAP_MAX_BOOTSTRAPPED_DEMOS,
        trainer_extract_max_retries=TRAINER_EXTRACT_MAX_RETRIES,
        default_local_updates=DEFAULT_LOCAL_UPDATES,
        default_optimizer_name=DEFAULT_OPTIMIZER_NAME,
        default_optimizer_lr=DEFAULT_OPTIMIZER_LR,
        default_metric_name=DEFAULT_METRIC_NAME,
        default_metric_package=DEFAULT_METRIC_PACKAGE,
        default_metric_task=DEFAULT_METRIC_TASK,
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
