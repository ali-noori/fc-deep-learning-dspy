"""BootstrapFewShot: always compile TrainerConfigModule with few-shot demos."""

from __future__ import annotations

from functools import partial

from .config import Settings
from .dspy_support import dspy
from .lm_router import configure_dspy_cascade
from .metric import trainer_config_metric
from .trainset import build_trainset
from .trainer_selector import TrainerConfigModule


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
    trainer_artifacts: dict[str, str] | None = None,
    dataloader_artifacts: dict[str, str] | None = None,
    loss_artifacts: dict[str, str] | None = None,
) -> TrainerConfigModule:
    """
    Configure the LM cascade and run BootstrapFewShot on fewshot_examples.

    Same convention as tools/execution_mode_dspy: few-shot compile runs every time.
    """
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    trainset = build_trainset(
        trainer_artifacts,
        dataloader_artifacts,
        loss_artifacts,
    )
    trainer_module = TrainerConfigModule(
        settings,
        trainer_artifacts=trainer_artifacts or {},
        dataloader_artifacts=dataloader_artifacts or {},
        loss_artifacts=loss_artifacts or {},
    )

    teleprompter = dspy.BootstrapFewShot(
        metric=partial(
            trainer_config_metric,
            trainer_artifacts=trainer_artifacts or {},
            dataloader_artifacts=dataloader_artifacts or {},
            loss_artifacts=loss_artifacts or {},
        ),
        max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
        max_labeled_demos=len(trainset),
    )
    return teleprompter.compile(student=trainer_module, trainset=trainset)
