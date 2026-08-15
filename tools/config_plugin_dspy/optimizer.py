"""Compile ConfigIntentExtractorModule with BootstrapFewShot."""

from __future__ import annotations

from .config import Settings, resolve_ollama_base_url, resolve_ollama_model
from .dspy_support import dspy
from .intent_extractor import ConfigIntentExtractorModule
from .lm_router import configure_dspy_cascade
from .metric import intent_metric
from .trainset import build_trainset


def compile_module(
    settings: Settings,
    artifacts: dict[str, list[str]],
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    *,
    skip_bootstrap: bool = False,
) -> ConfigIntentExtractorModule:
    base_url = resolve_ollama_base_url(ollama_base_url)
    model = resolve_ollama_model(ollama_model)
    configure_dspy_cascade(settings, model, base_url)

    student = ConfigIntentExtractorModule(settings)
    if skip_bootstrap:
        return student

    trainset = build_trainset(artifacts)
    teleprompter = dspy.BootstrapFewShot(
        metric=intent_metric,
        max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
    )
    return teleprompter.compile(student=student, trainset=trainset)
