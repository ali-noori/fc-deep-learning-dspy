"""Training examples for BootstrapFewShot (from fewshot_examples.py)."""

from __future__ import annotations

from .dspy_support import dspy
from .fewshot_examples import FEWSHOT_PAIRS
from .trainer_selector import (
    format_known_dataloaders,
    format_known_losses,
    format_known_trainers,
)


def build_trainset(
    trainer_artifacts: dict[str, str] | None = None,
    dataloader_artifacts: dict[str, str] | None = None,
    loss_artifacts: dict[str, str] | None = None,
) -> list:
    if not FEWSHOT_PAIRS:
        raise RuntimeError("FEWSHOT_PAIRS is empty; cannot build trainset.")

    known_trainers = format_known_trainers(trainer_artifacts or {})
    known_dataloaders = format_known_dataloaders(dataloader_artifacts or {})
    known_losses = format_known_losses(loss_artifacts or {})
    examples = []
    for user_message, name, data_loader, loss_name, num_classes in FEWSHOT_PAIRS:
        example = dspy.Example(
            user_message=user_message,
            known_trainers=known_trainers,
            known_dataloaders=known_dataloaders,
            known_losses=known_losses,
            name=name,
            data_loader=data_loader,
            loss_name=loss_name,
            num_classes=num_classes,
        ).with_inputs(
            "user_message",
            "known_trainers",
            "known_dataloaders",
            "known_losses",
        )
        examples.append(example)
    return examples
