"""Import DSPy once for this package (avoids repeating try/import in every file)."""

from __future__ import annotations

try:
    import dspy
except ImportError as exc:
    raise ImportError(
        "DSPy is not installed. Install it with: pip install dspy-ai"
    ) from exc
