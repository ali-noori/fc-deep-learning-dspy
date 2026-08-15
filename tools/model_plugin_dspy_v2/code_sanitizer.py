"""Normalize raw LM output into executable Python module source."""

from __future__ import annotations

import re


def strip_code_fence(text: str) -> str:
    try:
        cleaned = text.strip()
        fence_match = re.search(
            r"```(?:python)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL
        )
        if fence_match:
            return fence_match.group(1).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        return cleaned.strip()
    except re.error as exc:
        raise RuntimeError("Failed to strip markdown code fences from model output.") from exc


def extract_python_code(text: str) -> str:
    try:
        cleaned = strip_code_fence(text)
        import_match = re.search(r"(?m)^(?:import|from)\s+", cleaned)
        if import_match:
            cleaned = cleaned[import_match.start() :]
        return cleaned.strip()
    except re.error as exc:
        raise RuntimeError("Failed to extract Python code from model output.") from exc
