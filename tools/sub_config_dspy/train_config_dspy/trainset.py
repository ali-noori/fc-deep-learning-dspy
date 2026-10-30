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
        verbose,
        batch_size,
        test_batch_size,
        epochs,
        lr,
        batch_count,
    ) in FEWSHOT_PAIRS:
        example = dspy.Example(
            user_message=user_message,
            verbose=verbose,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            epochs=epochs,
            lr=lr,
            batch_count=batch_count,
        ).with_inputs("user_message")
        examples.append(example)
    return examples
