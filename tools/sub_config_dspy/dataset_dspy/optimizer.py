"""BootstrapFewShot: compile a dataset module, or reuse a saved compiled program."""

from __future__ import annotations

from pathlib import Path

from .config import COMPILED_PROGRAM_DIR_NAME, Settings, resolve_recompile
from .dataset_config_selector import DatasetFederatedConfigModule
from .dataset_config_selector import DatasetCentralizedConfigModule
from .dataset_config_selector import DatasetSimulationConfigModule
from .dspy_support import dspy
from .lm_router import configure_dspy_cascade
from .metric import dataset_federated_config_metric
from .metric import dataset_centralized_config_metric
from .metric import dataset_simulation_config_metric
from .trainset import build_trainset

_PACKAGE_DIR = Path(__file__).resolve().parent


def _compiled_program_path(execution_mode: str) -> Path:
    return _PACKAGE_DIR / COMPILED_PROGRAM_DIR_NAME / f"compiled_agent_{execution_mode}.json"


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
    execution_mode: str,
    recompile: bool | None = None,
):
    """
    Configure the LM cascade, then either:
    - recompile=False: load the previously saved compiled agent for this mode, or
    - recompile=True: run BootstrapFewShot and save the result for this mode.
    """
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    if execution_mode == "federated":
        student = DatasetFederatedConfigModule(settings)
        metric = dataset_federated_config_metric
    elif execution_mode == "centralized":
        student = DatasetCentralizedConfigModule(settings)
        metric = dataset_centralized_config_metric
    elif execution_mode == "simulation":
        student = DatasetSimulationConfigModule(settings)
        metric = dataset_simulation_config_metric
    else:
        raise ValueError(f"Unknown execution mode: {execution_mode}")

    compiled_path = _compiled_program_path(execution_mode)

    if not resolve_recompile(recompile):
        if not compiled_path.exists():
            raise RuntimeError(
                f"No compiled agent found at {compiled_path}. "
                "Run once with --recompile=True to create it."
            )
        student.load(str(compiled_path))
        return student

    trainset = build_trainset(execution_mode)
    teleprompter = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
        max_labeled_demos=len(trainset),
    )
    compiled = teleprompter.compile(student=student, trainset=trainset)
    compiled_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(compiled_path))
    return compiled
