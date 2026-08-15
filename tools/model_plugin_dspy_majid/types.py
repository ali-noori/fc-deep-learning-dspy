"""Data shapes passed between pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureDescription:
    """Result of one successful language-model generation."""

    code: str
    backend: str
    model: str
    base_url: str | None
