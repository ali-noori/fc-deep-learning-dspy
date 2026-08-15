"""Shared data shapes for the config plugin pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserIntent:
    """Structured fields extracted from natural language (maps onto fc_deep YAML)."""

    task: str
    model_name: str
    trainer_name: str
    aggregator_name: str
    loss_name: str
    data_loader_name: str
    raw_request: str


@dataclass
class ConfigPlan:
    config_data: dict[str, Any]


@dataclass(frozen=True)
class IntentExtractionResult:
    intent: UserIntent
    intent_json: str
    backend: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class ConfigGenerationResult:
    """Intent JSON plus merged config and provider metadata."""

    intent: UserIntent
    intent_json: str
    config_yaml: str
    backend: str
    model: str
    base_url: str | None


@dataclass
class PipelineRunResult:
    approach: str
    template: str
    backend: str
    model: str
    base_url: str | None
    output_config: str
    config_yaml: str
    intent: UserIntent
    output_files: dict[str, str] = field(default_factory=dict)
