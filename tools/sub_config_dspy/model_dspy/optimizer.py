"""BootstrapFewShot: always compile ModelConfigModule with few-shot demos."""

from __future__ import annotations

from functools import partial

from .config import Settings
from .dspy_support import dspy
from .lm_router import configure_dspy_cascade
from .metric import model_config_metric
from .model_config_selector import ModelConfigModule
from .trainset import build_trainset


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
    model_artifacts: dict[str, str] | None = None,
) -> ModelConfigModule:
    """
    Configure the LM cascade and run BootstrapFewShot on fewshot_examples.

    Same convention as tools/execution_mode_dspy: few-shot compile runs every time.
    """
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    trainset = build_trainset(model_artifacts)
    ModelConfigModuleObj = ModelConfigModule(
        settings,
        model_artifacts=model_artifacts or {},
    )

    teleprompter = dspy.BootstrapFewShot(
        metric=partial(
            model_config_metric,
            model_artifacts=model_artifacts or {},
        ),
        max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
        max_labeled_demos=len(trainset),
    )
    return teleprompter.compile(student=ModelConfigModuleObj, trainset=trainset)
