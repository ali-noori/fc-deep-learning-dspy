"""Shared data shapes for the train_config_dspy pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    verbose: bool
    batch_size: int
    test_batch_size: int
    epochs: int
    lr: float
    batch_count: int
    device: str = "cpu"


@dataclass(frozen=True)
class TrainConfigExtractionResult:
    verbose: bool
    batch_size: int
    test_batch_size: int
    epochs: int
    lr: float
    batch_count: int
    backend: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class TrainConfigResult:
    verbose: bool
    batch_size: int
    test_batch_size: int
    epochs: int
    lr: float
    batch_count: int
    device: str
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
