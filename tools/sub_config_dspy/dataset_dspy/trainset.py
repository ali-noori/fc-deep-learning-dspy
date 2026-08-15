"""Training examples for BootstrapFewShot (from fewshot_examples.py)."""

from __future__ import annotations

from .dspy_support import dspy
from .fewshot_examples import FEWSHOT_PAIRS_FEDERATED
from .fewshot_examples import FEWSHOT_PAIRS_CENTRALIZED
from .fewshot_examples import FEWSHOT_PAIRS_SIMULATION


def build_trainset(execution_mode: str) -> list:
    if not FEWSHOT_PAIRS_FEDERATED and not FEWSHOT_PAIRS_CENTRALIZED:
        raise RuntimeError("FEWSHOT_PAIRS_FEDERATED and FEWSHOT_PAIRS_CENTRALIZED are empty; cannot build trainset.")

    examples = []
    
    # Adding federated examples
    if execution_mode == "federated":
        for (
            user_message,
            data_dirs,
            train_dataset_file_name,
            test_dataset_file_name,
            logic_dir,
        ) in FEWSHOT_PAIRS_FEDERATED:
            example = dspy.Example(
                user_message=user_message,
                data_dirs=data_dirs,
                train_dataset_file_name=train_dataset_file_name,
                test_dataset_file_name=test_dataset_file_name,
                logic_dir=logic_dir,
            ).with_inputs("user_message")
            examples.append(example)
    
    # Adding centralized examples
    elif execution_mode == "centralized":
        for (
            user_message,
            data_dir,
            train_dataset_file_name,
            test_dataset_file_name,
            logic_dir,
        ) in FEWSHOT_PAIRS_CENTRALIZED:
            example = dspy.Example(
                user_message=user_message,
                data_dir=data_dir,
                train_dataset_file_name=train_dataset_file_name,
                test_dataset_file_name=test_dataset_file_name,
                logic_dir=logic_dir,
            ).with_inputs("user_message")
            examples.append(example)
    
    # Adding simulation examples
    elif execution_mode == "simulation":
        for (
            user_message,
            data_dir,
            client_dirs,
            train_dataset_file_name,
            test_dataset_file_name,
            logic_dir,
        ) in FEWSHOT_PAIRS_SIMULATION:
            example = dspy.Example(
                user_message=user_message,
                data_dir=data_dir,
                client_dirs=client_dirs,
                train_dataset_file_name=train_dataset_file_name,
                test_dataset_file_name=test_dataset_file_name,
                logic_dir=logic_dir,
            ).with_inputs("user_message")
            examples.append(example)
    else:
        raise ValueError(f"Unknown execution mode: {execution_mode}")
        
    return examples
