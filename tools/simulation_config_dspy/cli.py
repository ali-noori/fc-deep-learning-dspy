"""CLI for simulation_config_dspy."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from tools.artifact_discovery import ArtifactDiscovery
from tools.config_builder.builder import build_config, resolve_execution_mode, write_generated_config

# Distinguishes which step a Yes/No question belongs to: generating a new
# custom model vs. saving that generated model as a reusable plugin file.
QuestionType = Literal["generate_model", "save_model"]


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")

def _ask_yes_no(question: str, question_type: QuestionType) -> bool:
    while True:
        answer = input(f"{question} (Yes/No): ").strip().lower()
        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False
        print(f"Please answer Yes or No for the {question_type.replace('_', ' ')} question.")


def _ask_model_name(existing_names_lower: set[str]) -> str:
    while True:
        raw = input("Enter a name for this model (letters, digits, underscore only): ").strip()
        name = raw[:-3] if raw.lower().endswith(".py") else raw
        if not name or not all(c.isalnum() or c == "_" for c in name):
            print("Please enter a valid name (no spaces, slashes, or special characters).")
            continue
        # Case-insensitive check: reject cnn/CNN/Cnn, mlp/MLP, customized_model_v2, etc.
        if name.lower() in existing_names_lower:
            print(
                f"A model named '{name}' already exists under plugins/models. "
                "Please choose a different name."
            )
            continue
        return name


def _save_reusable_model() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "plugins" / "models" / "generated_architecture" / "customized_model_v2.py"
    if not src.exists():
        print(f"Simulation config agent: {src} not found; cannot save as reusable model.", file=sys.stderr)
        return

    models_dir = repo_root / "plugins" / "models"
    existing = ArtifactDiscovery(models_dir, repo_root=repo_root).discover()
    existing_names_lower = {Path(filename).stem.lower() for filename in existing}

    while True:
        name = _ask_model_name(existing_names_lower)
        dest = models_dir / f"{name}.py"
        if dest.exists() and not _ask_yes_no(
            f"{dest.name} already exists. Overwrite it?",
            question_type="save_model",
        ):
            continue
        shutil.copyfile(src, dest)
        print(f"Simulation config agent: saved reusable model to {dest}")
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="simulation config agent.")
    parser.add_argument("--execution-mode", default="simulation")
    parser.add_argument(
        "--recompile",
        type=_parse_bool,
        default=None,
        metavar="{True,False}",
        help="Forwarded to dataset, hyper-params, model, and trainer agents.",
    )
    args = parser.parse_args()

    print("'simulation_config_agent' was selected")
    
    def _sub_cmd(module: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            module,
            "--execution-mode",
            args.execution_mode,
        ]
        if args.recompile is not None:
            cmd.append(f"--recompile={args.recompile}")
        return cmd

    # 1.run dataset_dspy to create dataset part of the simulation mode config file
    subprocess.run(
        _sub_cmd("tools.sub_config_dspy.dataset_dspy.cli"),
        check=False,
    )
    
    # 2. run hyper_params_dspy to create fed_hyper_params part of the simulation mode config file
    subprocess.run(
        _sub_cmd("tools.sub_config_dspy.hyper_params_dspy.cli"),
        check=False,
    )
    
    # 2.5. ask user if they want to generate a new model
    if _ask_yes_no("Do you want to generate a new model?", question_type="generate_model"):
        model_plugin_cmd = [sys.executable, "-m", "tools.model_plugin_dspy_v2.cli"]
        if args.recompile is not None:
            model_plugin_cmd.append(f"--recompile={args.recompile}")
        subprocess.run(model_plugin_cmd, check=False)
        if _ask_yes_no(
            "Do you want to save this model as a reusable model?",
            question_type="save_model",
        ):
            _save_reusable_model()

    # 3. run model_dspy to create model part of the simulation mode config file
    subprocess.run(
        _sub_cmd("tools.sub_config_dspy.model_dspy.cli"),
        check=False,
    )
    
    # 4. run trainer_dspy to create trainer part of the simulation mode config file
    subprocess.run(
        _sub_cmd("tools.sub_config_dspy.trainer_dspy.cli"),
        check=False,
    )
    
    # 5. run config_builder to build the final simulation mode config file
    mode = resolve_execution_mode(args.execution_mode)
    config = build_config(mode)
    out_path = write_generated_config(config, mode)
    print(f"Simulation config agent: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
