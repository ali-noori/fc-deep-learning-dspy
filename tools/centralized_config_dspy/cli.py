"""CLI for centralized_config_dspy."""

from __future__ import annotations

import argparse
import subprocess
import sys

from tools.config_builder.builder import build_config, resolve_execution_mode, write_generated_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Centralized config agent.")
    parser.add_argument("--execution-mode", default="centralized")
    args = parser.parse_args()

    print("'centralized_config_agent' was selected")
    
    # 1. run dataset_dspy to create dataset part of the centralized mode config file
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.sub_config_dspy.dataset_dspy.cli",
            "--execution-mode",
            args.execution_mode,
        ],
        check=False,
    )
    
    # 2. run hyper_params_dspy to create fed_hyper_params part of the centralized mode config file
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.sub_config_dspy.hyper_params_dspy.cli",
            "--execution-mode",
            args.execution_mode,
        ],
        check=False,
    )
    
    # 3. run model_dspy to create model part of the centralized mode config file
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.sub_config_dspy.model_dspy.cli",
            "--execution-mode",
            args.execution_mode,
        ],
        check=False,
    )
    
    # 4. run trainer_dspy to create trainer part of the centralized mode config file
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.sub_config_dspy.trainer_dspy.cli",
            "--execution-mode",
            args.execution_mode,
        ],
        check=False,
    )
    
    # 5. run config_builder to build the final centralized mode config file
    mode = resolve_execution_mode(args.execution_mode)
    config = build_config(mode)
    out_path = write_generated_config(config, mode)
    print(f"Centralized config agent: wrote {out_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
