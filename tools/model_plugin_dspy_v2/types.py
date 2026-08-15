"""Shared data shapes for the v2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureResult:
    """Sanitized PyTorch source plus provider metadata for file headers."""

    code: str
    backend: str
    model: str
    base_url: str | None
