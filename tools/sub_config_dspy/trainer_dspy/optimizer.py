"""BootstrapFewShot: compile TrainerConfigModule, or reuse a saved compiled program."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from .config import (
    COMPILED_PROGRAM_DIR_NAME,
    COMPILED_PROGRAM_FILENAME,
    Settings,
    resolve_recompile,
)
from .dspy_support import dspy
from .lm_router import configure_dspy_cascade
from .metric import trainer_config_metric
from .trainset import build_trainset
from .trainer_selector import TrainerConfigModule

_PACKAGE_DIR = Path(__file__).resolve().parent
COMPILED_PROGRAM_PATH = _PACKAGE_DIR / COMPILED_PROGRAM_DIR_NAME / COMPILED_PROGRAM_FILENAME


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
    trainer_artifacts: dict[str, str] | None = None,
    dataloader_artifacts: dict[str, str] | None = None,
    loss_artifacts: dict[str, str] | None = None,
    recompile: bool | None = None,
) -> TrainerConfigModule:
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    student = TrainerConfigModule(
        settings,
        trainer_artifacts=trainer_artifacts or {},
        dataloader_artifacts=dataloader_artifacts or {},
        loss_artifacts=loss_artifacts or {},
    )

    if not resolve_recompile(recompile):
        if not COMPILED_PROGRAM_PATH.exists():
            raise RuntimeError(
                f"No compiled agent found at {COMPILED_PROGRAM_PATH}. "
                "Run once with --recompile=True to create it."
            )
        student.load(str(COMPILED_PROGRAM_PATH))
        return student

    trainset = build_trainset(
        trainer_artifacts,
        dataloader_artifacts,
        loss_artifacts,
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
    compiled = teleprompter.compile(student=student, trainset=trainset)
    COMPILED_PROGRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(COMPILED_PROGRAM_PATH))
    return compiled
