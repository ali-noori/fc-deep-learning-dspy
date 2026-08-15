"""BootstrapFewShot: always compile DatasetConfigSelectorModule with few-shot demos."""

from __future__ import annotations

from .config import Settings
from .dataset_config_selector import DatasetFederatedConfigModule
from .dataset_config_selector import DatasetCentralizedConfigModule 
from .dataset_config_selector import DatasetSimulationConfigModule
from .dspy_support import dspy
from .lm_router import configure_dspy_cascade
from .metric import dataset_federated_config_metric
from .metric import dataset_centralized_config_metric
from .metric import dataset_simulation_config_metric
from .trainset import build_trainset


def compile_module(
    settings: Settings,
    ollama_model: str,
    ollama_base_url: str,
    execution_mode: str,
):
    """
    Configure the LM cascade and run BootstrapFewShot on fewshot_examples.

    Same convention as tools/execution_mode_dspy: few-shot compile runs every time.
    """
    configure_dspy_cascade(settings, ollama_model, ollama_base_url)

    trainset = build_trainset(execution_mode)
    
    if execution_mode == "federated":
        datasetFederatedConfigModuleObj = DatasetFederatedConfigModule(settings)

        teleprompter = dspy.BootstrapFewShot(
            metric=dataset_federated_config_metric,
            max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
            max_labeled_demos=len(trainset),
        )
        return teleprompter.compile(student=datasetFederatedConfigModuleObj, trainset=trainset)
    elif execution_mode == "centralized":
        datasetCentralizedConfigModuleObj = DatasetCentralizedConfigModule(settings)
        
        teleprompter = dspy.BootstrapFewShot(
            metric=dataset_centralized_config_metric,
            max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
            max_labeled_demos=len(trainset),
        )
        return teleprompter.compile(student=datasetCentralizedConfigModuleObj, trainset=trainset)
    elif execution_mode == "simulation":
        datasetSimulationConfigModuleObj = DatasetSimulationConfigModule(settings)
        
        teleprompter = dspy.BootstrapFewShot(
            metric=dataset_simulation_config_metric,
            max_bootstrapped_demos=settings.bootstrap_max_bootstrapped_demos,
            max_labeled_demos=len(trainset),
        )
        return teleprompter.compile(student=datasetSimulationConfigModuleObj, trainset=trainset)