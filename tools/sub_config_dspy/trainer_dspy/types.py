"""Shared data shapes for the trainer_dspy pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConfig:
    name: str
    data_loader: str
    loss_name: str
    num_classes: int
    local_updates: str = "WEIGHTS_N_SAMPLES"
    optimizer_name: str = "SGD"
    optimizer_lr: float = 0.1


@dataclass(frozen=True)
class TrainerExtractionResult:
    name: str
    data_loader: str
    loss_name: str
    num_classes: int
    backend: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class TrainerConfigResult:
    name: str
    data_loader: str
    loss_name: str
    num_classes: int
    local_updates: str
    optimizer_name: str
    optimizer_lr: float
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
