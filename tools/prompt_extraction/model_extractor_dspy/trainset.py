"""Training examples for BootstrapFewShot (from fewshot_examples.py)."""

from __future__ import annotations

from .dspy_support import dspy
from .fewshot_examples import FEWSHOT_PAIRS


def build_trainset() -> list:
    if not FEWSHOT_PAIRS:
        raise RuntimeError("FEWSHOT_PAIRS is empty; cannot build trainset.")

    examples = []
    for (
        user_message,
        architecture_description,
        model_selection,
        in_feature,
        n_classes,
    ) in FEWSHOT_PAIRS:
        example = dspy.Example(
            user_message=user_message,
            architecture_description=architecture_description,
            model_selection=model_selection,
            in_feature=in_feature,
            n_classes=n_classes,
        ).with_inputs("user_message")
        examples.append(example)
    return examples
