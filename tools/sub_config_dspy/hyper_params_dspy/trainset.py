"""Training examples for BootstrapFewShot (from fewshot_examples.py)."""

from __future__ import annotations

from .dspy_support import dspy
from .fewshot_examples import FEWSHOT_PAIRS
from .hyper_params_selector import format_known_aggregators


def build_trainset(aggregator_artifacts: dict[str, str] | None = None) -> list:
    if not FEWSHOT_PAIRS:
        raise RuntimeError("FEWSHOT_PAIRS is empty; cannot build trainset.")

    known_aggregators = format_known_aggregators(aggregator_artifacts or {})
    examples = []
    for user_message, max_iter, n_classes, federated_model in FEWSHOT_PAIRS:
        example = dspy.Example(
            user_message=user_message,
            known_aggregators=known_aggregators,
            max_iter=max_iter,
            n_classes=n_classes,
            federated_model=federated_model,
        ).with_inputs("user_message", "known_aggregators")
        examples.append(example)
    return examples
