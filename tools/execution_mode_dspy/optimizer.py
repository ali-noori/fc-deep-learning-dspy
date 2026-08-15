"""BootstrapFewShot: always compile ExecutionModeSelectorModule with few-shot demos."""

from __future__ import annotations

from .config import Settings
from .dspy_support import dspy
from .execution_mode_selector import ExecutionModeSelectorModule
from .lm_router import configure_dspy_cascade
from .metric import execution_mode_metric
from .trainset import build_trainset


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
) -> ExecutionModeSelectorModule:
    """
    Configure the LM cascade and run BootstrapFewShot on fewshot_examples.

    Same convention as tools/model_plugin_dspy_v2: few-shot compile runs every time.
    """
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    trainset = build_trainset()
    student = ExecutionModeSelectorModule(settings)

    teleprompter = dspy.BootstrapFewShot(
        metric=execution_mode_metric,
        max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
        max_labeled_demos=len(trainset),
    )
    return teleprompter.compile(student=student, trainset=trainset)
