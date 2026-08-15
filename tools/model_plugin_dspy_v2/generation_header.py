"""Build auto-generated file header comments."""

from __future__ import annotations

from .config import AUTO_GEN_HEADER_LINE, Settings


def build_generation_header(
    settings: Settings,
    backend: str,
    model: str,
    base_url: str | None,
) -> str:
    try:
        provider = settings.provider_display_names.get(backend, backend)
        lines = [
            AUTO_GEN_HEADER_LINE,
            f"# Organization / provider: {provider}",
            f"# Model: {model}",
        ]
        if base_url:
            lines.append(f"# Ollama API base: {base_url}")
        return "\n".join(lines) + "\n\n"
    except Exception as exc:
        raise RuntimeError("Failed to build generation header text.") from exc
