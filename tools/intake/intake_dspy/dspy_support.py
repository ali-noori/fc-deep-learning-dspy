"""Import DSPy once for this package."""

from __future__ import annotations

try:
    import dspy
except ImportError as exc:
    raise ImportError(
        "DSPy is not installed. Install it with: pip install dspy-ai"
    ) from exc
