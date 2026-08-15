"""Shared data shapes for the hyper_params_dspy pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FedHyperParamsConfig:
    max_iter: int
    n_classes: int
    federated_model: str
    global_updates: str = "WEIGHTS_STOPPING"


@dataclass(frozen=True)
class FedHyperParamsExtractionResult:
    max_iter: int
    n_classes: int
    federated_model: str
    backend: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class FedHyperParamsConfigResult:
    max_iter: int
    n_classes: int
    federated_model: str
    global_updates: str
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
