"""BootstrapFewShot: always compile FedHyperParamsConfigModule with few-shot demos."""

from __future__ import annotations

from functools import partial

from .config import Settings
from .dspy_support import dspy
from .hyper_params_selector import FedHyperParamsConfigModule
from .lm_router import configure_dspy_cascade
from .metric import hyper_params_config_metric
from .trainset import build_trainset


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
    aggregator_artifacts: dict[str, str] | None = None,
) -> FedHyperParamsConfigModule:
    """
    Configure the LM cascade and run BootstrapFewShot on fewshot_examples.

    Same convention as tools/execution_mode_dspy: few-shot compile runs every time.
    """
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    trainset = build_trainset(aggregator_artifacts)
    FedHyperParamsConfigModuleObj = FedHyperParamsConfigModule(
        settings,
        aggregator_artifacts=aggregator_artifacts or {},
    )

    teleprompter = dspy.BootstrapFewShot(
        metric=partial(
            hyper_params_config_metric,
            aggregator_artifacts=aggregator_artifacts or {},
        ),
        max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
        max_labeled_demos=len(trainset),
    )
    return teleprompter.compile(student=FedHyperParamsConfigModuleObj, trainset=trainset)
