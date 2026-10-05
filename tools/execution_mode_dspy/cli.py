"""CLI for execution_mode_dspy (DSPy + BootstrapFewShot, same convention as model_plugin_dspy_v2)."""

from __future__ import annotations

import argparse
import subprocess
import sys

from .config import load_settings, resolve_recompile
from .pipeline import WorkflowPipeline


def _activate_next_agent(execution_mode: str, recompile: bool) -> None:
    """Run the config agent for this execution mode (python -m tools.<agent>.cli)."""
    mode = execution_mode.strip().lower()
    recompile_flag = f"--recompile={recompile}"

    if mode == "federated":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.federated_config_dspy.cli",
                "--execution-mode",
                mode,
                recompile_flag,
            ],
            check=False,
        )
    elif mode == "simulation":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.simulation_config_dspy.cli",
                "--execution-mode",
                mode,
                recompile_flag,
            ],
            check=False,
        )
    elif mode == "centralized":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.centralized_config_dspy.cli",
                "--execution-mode",
                mode,
                recompile_flag,
            ],
            check=False,
        )
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
    
    # run config_executor to execute the final config file
    subprocess.run(
        [sys.executable, "-m", "tools.config_executor.cli", "--execution-mode", mode],
        check=False,
    )


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execution mode agent.")
    parser.add_argument(
        "--recompile",
        type=_parse_bool,
        default=None,
        metavar="{True,False}",
        help=(
            "--recompile=True: run BootstrapFewShot from scratch and overwrite the saved "
            "compiled agent. --recompile=False: skip compiling and load the saved compiled "
            "agent (must already exist). If omitted, falls back to FC_DSPY_RECOMPILE env "
            "var, then RECOMPILE in config.py."
        ),
    )
    args = parser.parse_args()

    # load all the necessary information such as avaialable 
    # models that can be used online or offline(ollama), 
    # later will be used for dspy to know whihc llm model uses
    settings = load_settings()
    recompile = resolve_recompile(args.recompile)
    
    # just create an obj -> later is going to be used to call the 'run()' function
    pipeline = WorkflowPipeline(settings=settings, recompile=recompile)

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
    _activate_next_agent(result["execution_mode"], recompile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
