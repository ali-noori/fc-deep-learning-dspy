"""Training examples for BootstrapFewShot (from fewshot_examples.py)."""

from __future__ import annotations

import json

from .dspy_support import dspy
from .fewshot_examples import FEWSHOT_PAIRS

INPUT_FIELDS = (
    "user_request",
    "known_models",
    "known_trainers",
    "known_dataloaders",
    "known_aggregators",
    "known_losses",
    "known_optimizers",
    "known_devices",
)


def build_trainset(artifacts: dict[str, list[str]]) -> list:
    if not FEWSHOT_PAIRS:
        raise RuntimeError("FEWSHOT_PAIRS is empty; cannot build trainset.")

    known_models = str(artifacts.get("models", []))
    known_trainers = str(artifacts.get("trainers", []))
    known_dataloaders = str(artifacts.get("dataloaders", []))
    known_aggregators = str(artifacts.get("aggregators", []))
    known_losses = str(artifacts.get("losses", []))
    known_optimizers = str(artifacts.get("optimizers", []))
    known_devices = str(artifacts.get("devices", []))

    examples = []
    for user_request, intent_dict in FEWSHOT_PAIRS:
        example = dspy.Example(
            user_request=user_request,
            known_models=known_models,
            known_trainers=known_trainers,
            known_dataloaders=known_dataloaders,
            known_aggregators=known_aggregators,
            known_losses=known_losses,
            known_optimizers=known_optimizers,
            known_devices=known_devices,
            intent_json=json.dumps(intent_dict),
        ).with_inputs(*INPUT_FIELDS)
        examples.append(example)
    return examples
