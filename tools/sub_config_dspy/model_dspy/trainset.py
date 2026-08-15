"""Training examples for BootstrapFewShot (from fewshot_examples.py)."""

from __future__ import annotations

from .dspy_support import dspy
from .fewshot_examples import FEWSHOT_PAIRS
from .model_config_selector import format_known_models


def build_trainset(model_artifacts: dict[str, str] | None = None) -> list:
    if not FEWSHOT_PAIRS:
        raise RuntimeError("FEWSHOT_PAIRS is empty; cannot build trainset.")

    known_models = format_known_models(model_artifacts or {})
    examples = []
    for user_message, name, n_classes, in_features in FEWSHOT_PAIRS:
        example = dspy.Example(
            user_message=user_message,
            known_models=known_models,
            name=name,
            n_classes=n_classes,
            in_features=in_features,
        ).with_inputs("user_message", "known_models")
        examples.append(example)
    return examples
