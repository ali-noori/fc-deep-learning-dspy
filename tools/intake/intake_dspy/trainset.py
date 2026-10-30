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
        execution_mode_dspy,
        dataset_dspy,
        hyper_params_dspy,
        trainer_dspy,
        train_config_dspy,
    ) in FEWSHOT_PAIRS:
        example = dspy.Example(
            user_message=user_message,
            execution_mode_dspy=execution_mode_dspy,
            dataset_dspy=dataset_dspy,
            hyper_params_dspy=hyper_params_dspy,
            trainer_dspy=trainer_dspy,
            train_config_dspy=train_config_dspy,
        ).with_inputs("user_message")
        examples.append(example)
    return examples
