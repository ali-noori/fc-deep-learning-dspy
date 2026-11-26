"""CLI for experiment_splitter_dspy. GUI by default; --no-gui reads PowerShell input."""

from __future__ import annotations

import argparse
import subprocess
import sys

from tools.prompt_extraction.gui.experiment_splitter_dspy_prompt import ask_experiment_splitter_dspy_prompt
from tools.prompt_extraction.gui.model_extractor_dspy_prompt import ModelPrompt, ask_model_extractor_dspy_prompt
from tools.prompt_extraction.intake_process import INTAKE_PROCESS_PATH, build_final_intake
from tools.prompt_extraction.model_extractor_dspy.briefs import briefs_path as model_briefs_path
from tools.prompt_extraction.model_extractor_dspy.config import load_settings as load_model_settings
from tools.prompt_extraction.model_extractor_dspy.pipeline import ModelExtractorPipeline

from .briefs import briefs_path
from .config import load_settings, resolve_recompile
from .pipeline import ExperimentSplitterPipeline


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")


def _read_prompt(use_gui: bool, question: str) -> str | None:
    if not use_gui:
        print(f"Agent: {question}")
        return input("You: ").strip()
    return ask_experiment_splitter_dspy_prompt()


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment splitter agent.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Read the request from PowerShell instead of the window.",
    )
    parser.add_argument(
        "--recompile",
        type=_parse_bool,
        default=None,
        metavar="{True,False}",
        help="--recompile=True: compile and save. --recompile=False: load saved agent.",
    )
    args = parser.parse_args()

    settings = load_settings()
    recompile = resolve_recompile(args.recompile)
    pipeline = ExperimentSplitterPipeline(settings=settings, recompile=recompile)

    try:
        user_input = _read_prompt(not args.no_gui, pipeline.EXPERIMENT_SPLITTER_QUESTION)
    except EOFError:
        print("Agent: no prompt was entered.", file=sys.stderr)
        return 1
    if user_input is None:
        return 0
    if not user_input.strip():
        print("Agent: please enter a prompt before continuing.", file=sys.stderr)
        return 1

    try:
        result = pipeline.run(user_input)
    except ValueError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1

    print(f"Agent: {result['agent_reply']}")
    print(f"Agent: wrote {briefs_path()}")
    print(
        f"(output={result['output_path']}, backend={result['backend']}, "
        f"model={result['model']})"
    )
    if not args.no_gui:
        return _run_model_extractor(ask_model_extractor_dspy_prompt(), recompile)
    return 0


def _run_model_extractor(form: ModelPrompt | None, recompile: bool) -> int:
    if form is None:
        return 0
    if not form.prompt.strip():
        print("Agent: please enter a prompt before continuing.", file=sys.stderr)
        return 1

    pipeline = ModelExtractorPipeline(settings=load_model_settings(), recompile=recompile)
    try:
        result = pipeline.run(form.prompt)
    except ValueError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1

    print(f"Agent: {result['agent_reply']}")
    print(f"Agent: wrote {model_briefs_path()}")
    build_final_intake(
        form.new_architecture,
        form.new_architecture_name,
        form.save_new_architecture,
    )
    print(f"Agent: wrote {INTAKE_PROCESS_PATH}")
    print(
        f"(output={result['output_path']}, backend={result['backend']}, "
        f"model={result['model']})"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.execution_mode_dspy.cli",
            "--from-gui",
            "--recompile=False",
        ],
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
