"""BootstrapFewShot: compile ExecutionModeSelectorModule, or reuse a saved compiled program."""

from __future__ import annotations

from pathlib import Path

from .config import (
    COMPILED_PROGRAM_DIR_NAME,
    COMPILED_PROGRAM_FILENAME,
    Settings,
    resolve_recompile,
)
from .dspy_support import dspy
from .execution_mode_selector import ExecutionModeSelectorModule
from .lm_router import configure_dspy_cascade
from .metric import execution_mode_metric
from .trainset import build_trainset

_PACKAGE_DIR = Path(__file__).resolve().parent
COMPILED_PROGRAM_PATH = _PACKAGE_DIR / COMPILED_PROGRAM_DIR_NAME / COMPILED_PROGRAM_FILENAME


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
    recompile: bool | None = None,
) -> ExecutionModeSelectorModule:
    """
    Configure the LM cascade, then either:
    - recompile=False: load the previously saved compiled agent (skip BootstrapFewShot), or
    - recompile=True: run BootstrapFewShot on fewshot_examples and save the result.

    `recompile` is normally passed in from the CLI (--recompile/--no-recompile).
    If not given, falls back to FC_DSPY_RECOMPILE env var, then RECOMPILE in config.py.
    """
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    student = ExecutionModeSelectorModule(settings)

    if not resolve_recompile(recompile):
        if not COMPILED_PROGRAM_PATH.exists():
            raise RuntimeError(
                f"No compiled agent found at {COMPILED_PROGRAM_PATH}. "
                "Run once with --recompile to create it."
            )
        student.load(str(COMPILED_PROGRAM_PATH))
        return student

    trainset = build_trainset()

    teleprompter = dspy.BootstrapFewShot(
        metric=execution_mode_metric,
        max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
        max_labeled_demos=len(trainset),
    )
    compiled = teleprompter.compile(student=student, trainset=trainset)

    COMPILED_PROGRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(COMPILED_PROGRAM_PATH))
    return compiled
