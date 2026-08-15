"""Merge extracted intent into config.maximum.yml template (same paths as tools.dspy)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .types import ConfigPlan, UserIntent


def merge_intent_into_template(
    template: dict[str, Any],
    intent: UserIntent,
    artifacts: dict[str, list[str]] | None = None,
) -> ConfigPlan:
    """Deep-copy template and overwrite only non-empty intent fields."""
    _ = artifacts
    cfg = deepcopy(template)
    fc = cfg.setdefault("fc_deep", {})
    model = fc.setdefault("model", {})
    trainer = fc.setdefault("trainer", {})
    fed = fc.setdefault("fed_hyper_params", {})

    if intent.model_name:
        model["name"] = intent.model_name
    if intent.trainer_name:
        trainer["name"] = intent.trainer_name
    if intent.aggregator_name:
        fed["federated_model"] = intent.aggregator_name
    if intent.loss_name:
        loss = trainer.setdefault("loss", {})
        loss["name"] = intent.loss_name
    if intent.data_loader_name:
        trainer["data_loader"] = intent.data_loader_name

    return ConfigPlan(config_data=cfg)
