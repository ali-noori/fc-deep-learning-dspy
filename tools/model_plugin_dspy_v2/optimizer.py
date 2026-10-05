"""BootstrapFewShot: learn which few-shot demos help, then run your prompt."""

from __future__ import annotations

from pathlib import Path

from .architecture_generator import ArchitectureGeneratorModule
from .config import (
    COMPILED_PROGRAM_DIR_NAME,
    COMPILED_PROGRAM_FILENAME,
    Settings,
    resolve_recompile,
)
from .dspy_support import dspy
from .lm_router import configure_dspy_cascade
from .metric import architecture_metric
from .trainset import build_trainset

_PACKAGE_DIR = Path(__file__).resolve().parent
COMPILED_PROGRAM_PATH = _PACKAGE_DIR / COMPILED_PROGRAM_DIR_NAME / COMPILED_PROGRAM_FILENAME


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
    recompile: bool | None = None,
    repo_root: Path | None = None,
) -> ArchitectureGeneratorModule:
    """
    Run BootstrapFewShot on fewshot_examples and return the compiled module,
    or load a previously saved compiled agent (same --recompile convention
    as the other DSPy agents, e.g. tools/sub_config_dspy/model_dspy).
    """
    
    # Create support for the model
    ## IMPORTANT: Not only one model is being selected but a list of candidate models is being selected.
    ## and during the execution of the pipeline, the appropriate model is being selected.
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    # Create a raw generator (repo_root is needed so RAG can find plugins/models)
    root = repo_root or _PACKAGE_DIR.parents[1]
    generator = ArchitectureGeneratorModule(settings=settings, repo_root=root)

    if not resolve_recompile(recompile):
        if not COMPILED_PROGRAM_PATH.exists():
            raise RuntimeError(
                f"No compiled agent found at {COMPILED_PROGRAM_PATH}. "
                "Run once with --recompile=True to create it."
            )
        generator.load(str(COMPILED_PROGRAM_PATH))
        return generator

    # build dspy examples
    trainset = build_trainset()

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
    compiled = teleprompter.compile(student=generator, trainset=trainset)
    compiled._settings = settings
    compiled._repo_root = root
    COMPILED_PROGRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(COMPILED_PROGRAM_PATH))
    return compiled
