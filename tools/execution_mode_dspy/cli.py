"""CLI for execution_mode_dspy (DSPy + BootstrapFewShot, same convention as model_plugin_dspy_v2)."""

from __future__ import annotations

import subprocess
import sys

from .config import load_settings
from .pipeline import WorkflowPipeline


def _activate_next_agent(execution_mode: str) -> None:
    """Run the config agent for this execution mode (python -m tools.<agent>.cli)."""
    mode = execution_mode.strip().lower()

    if mode == "federated":
        subprocess.run(
            [sys.executable, "-m", "tools.federated_config_dspy.cli", "--execution-mode", mode],
            check=False,
        )
    elif mode == "simulation":
        subprocess.run(
            [sys.executable, "-m", "tools.simulation_config_dspy.cli", "--execution-mode", mode],
            check=False,
        )
    elif mode == "centralized":
        subprocess.run(
            [sys.executable, "-m", "tools.centralized_config_dspy.cli", "--execution-mode", mode],
            check=False,
        )
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
    
    # run config_executor to execute the final config file
    subprocess.run(
        [sys.executable, "-m", "tools.config_executor.cli", "--execution-mode", mode],
        check=False,
    )


def main() -> int:
    # load all the necessary information such as avaialable 
    # models that can be used online or offline(ollama), 
    # later will be used for dspy to know whihc llm model uses
    settings = load_settings()
    
    # just create an obj -> later is going to be used to call the 'run()' function
    pipeline = WorkflowPipeline(settings=settings)

    print(f"Agent: {pipeline.MODE_QUESTION}")
    try:
        user_input = input("You: ").strip()

        # Call 'select_execution_mode()':
        # 1. Read the user input message
        # 2. Create the module:
        # 2.1: (llm.py) call all the availabe potential models to find the one that is working and assign 
            # it as llm that dspy can use
        # 2.2: (trainset.py, fewshot_example.py) Create the trainset and assign it as example that dspy can use
        # 2.3: (execution_mode_selector.py, types.py) the "chain of toughts" and template for the out put of 
            # the prediction
        # 2.4: (mteric.py) to help dspy which example are btter by giving score to better examples
        # 2.5: Compbine all of the above deatils to create dspy agent
        # 2.6: give user message to created agent and get result based on the types defines already in 2.3
        # 2.7: create ModeSelectionResult class contains differen properties that can be used to crate the output 
            # for this agent in the next step
        result = pipeline.run(user_input)
    except ValueError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1

    print(f"Agent: {result['agent_reply']}")
    print(
        f"(mode={result['execution_mode']}, backend={result['backend']}, "
        f"model={result['model']})"
    )

    # based on the predicted mode, one of the federated, simulation or centrlized mode paths will be selected
    _activate_next_agent(result["execution_mode"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
