"""CLI for dataset_dspy (DSPy + BootstrapFewShot, same convention as execution_mode_dspy)."""

from __future__ import annotations

import sys
import argparse

from .config import load_settings
from .pipeline import DatasetPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset config agent.")
    parser.add_argument(
        "--execution-mode",
        help="Caller mode: federated, simulation, or centralized.",
    )
    args = parser.parse_args()
    execution_mode = (args.execution_mode or "").strip().lower()
    
    if execution_mode:
        print(f"(called from execution_mode={execution_mode})")
        
    allowed = {"federated", "simulation", "centralized"}
    if execution_mode and execution_mode not in allowed:
        print(f"error: unknown execution mode {execution_mode!r}", file=sys.stderr)
        return 2
    
    
    settings = load_settings()
    pipeline = DatasetPipeline(settings=settings)

    if execution_mode == "federated":
        print(f"Agent: {pipeline.DATASET_QUESTION_FEDERATED}")
    elif execution_mode == "centralized":
        print(f"Agent: {pipeline.DATASET_QUESTION_CENTRALIZED}")
    elif execution_mode == "simulation":
        print(f"Agent: {pipeline.DATASET_QUESTION_SIMULATION}")
    else:
        raise ValueError(f"Unknown execution mode: {execution_mode}")
    
    
    try:
        user_input = input("You: ").strip()
        result = pipeline.run(user_input, execution_mode)
    except ValueError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1

    print(f"Agent: {result['agent_reply']}")
    print(
        f"(output={result['output_path']}, backend={result['backend']}, "
        f"model={result['model']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
