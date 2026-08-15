"""Training examples for BootstrapFewShot (from fewshot_examples.py)."""

from __future__ import annotations

from .dspy_support import dspy
from .fewshot_examples import FEWSHOT_PAIRS


def build_trainset() -> list:
    """Turn each (description, code) pair into a dspy.Example."""
    if not FEWSHOT_PAIRS:
        raise RuntimeError("FEWSHOT_PAIRS is empty; cannot build trainset.")

    examples = []
    for description, code in FEWSHOT_PAIRS:
        example = dspy.Example(
            architecture_description=description,
            architecture_code=code,
        ).with_inputs("architecture_description")
        examples.append(example)
    return examples
