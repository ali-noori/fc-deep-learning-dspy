# If Groq isn't used, DSPy talks to Ollama here.
# FC_DSPY_OLLAMA_BASE_URL (default http://localhost:11434)
# FC_DSPY_OLLAMA_MODEL (default llama3.1)

from __future__ import annotations

import os


def get_ollama_base_url() -> str:
	base = os.environ.get("FC_DSPY_OLLAMA_BASE_URL", "http://localhost:11434").strip()
	return base.rstrip("/")


def get_ollama_model() -> str:
	return os.environ.get("FC_DSPY_OLLAMA_MODEL", "llama3.1").strip()
