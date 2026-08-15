# Groq key for DSPy: env vars first, then tools/local_secrets.py if it exists.

from __future__ import annotations

import os
from typing import Optional


def get_groq_api_key() -> Optional[str]:
	for name in ("GROQ_API_KEY", "FC_DSPY_GROQ_API_KEY"):
		v = os.environ.get(name)
		if v and str(v).strip():
			return str(v).strip()
	try:
		from tools.local_secrets import GROQ_API_KEY as local_key

		if local_key and str(local_key).strip():
			return str(local_key).strip()
	except ImportError:
		pass
	return None


def get_groq_model_spec() -> str:
	# LiteLLM format, default is Groq's fast 8b
	return os.environ.get("FC_DSPY_GROQ_MODEL", "groq/llama-3.1-8b-instant").strip()


def force_ollama_only() -> bool:
	return os.environ.get("FC_DSPY_FORCE_OLLAMA", "").strip().lower() in ("1", "true", "yes")
