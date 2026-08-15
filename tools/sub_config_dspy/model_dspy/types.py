"""Shared data shapes for the model_dspy pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_classes: int
    in_features: int


@dataclass(frozen=True)
class ModelExtractionResult:
    name: str
    n_classes: int
    in_features: int
    backend: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class ModelConfigResult:
    name: str
    n_classes: int
    in_features: int
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
