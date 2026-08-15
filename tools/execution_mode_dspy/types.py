"""Shared data shapes for the execution_mode_dspy pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionModeResult:
    execution_mode: str
    backend: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class ModeSelectionResult:
    execution_mode: str
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
