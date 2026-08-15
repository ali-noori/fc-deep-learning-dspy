"""BootstrapFewShot: learn which few-shot demos help, then run your prompt."""

from __future__ import annotations

from .architecture_generator import ArchitectureGeneratorModule
from .config import Settings
from .dspy_support import dspy
from .lm_router import configure_dspy_cascade
from .metric import architecture_metric
from .trainset import build_trainset


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
) -> ArchitectureGeneratorModule:
    """
    Run BootstrapFewShot on fewshot_examples and return the compiled module.

    Runs in memory only (nothing saved to disk).
    """
    
    # Create support for the model
    ## IMPORTANT: Not only one model is being selected but a list of candidate models is being selected.
    ## and during the execution of the pipeline, the appropriate model is being selected.
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    # build dspy examples
    trainset = build_trainset()
    
    # Create a raw generator
    generator = ArchitectureGeneratorModule()

    # 1. set metric
    # 2. set max_bootstrapped_demos
    # 3. set max_labeled_demos
    ## In here the immportant thing is that we set the metric
    teleprompter = dspy.teleprompt.BootstrapFewShot(
        metric=architecture_metric,
        max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
        max_labeled_demos=len(trainset),
    )
    # compile the generator
    ## It matches the raw generator with the candidate examples
    return teleprompter.compile(student=generator, trainset=trainset)
