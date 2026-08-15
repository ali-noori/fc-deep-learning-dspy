"""Training examples for BootstrapFewShot (from fewshot_examples.py)."""

from __future__ import annotations

from .dspy_support import dspy
from .fewshot_examples import FEWSHOT_PAIRS


def build_trainset() -> list:
    if not FEWSHOT_PAIRS:
        raise RuntimeError("FEWSHOT_PAIRS is empty; cannot build trainset.")

    examples = []
    for user_message, execution_mode in FEWSHOT_PAIRS:
        example = dspy.Example(
            user_message=user_message,
            execution_mode=execution_mode,
        ).with_inputs("user_message")
        examples.append(example)
    return examples
