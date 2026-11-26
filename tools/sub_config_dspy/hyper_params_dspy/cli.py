"""CLI for hyper_params_dspy (DSPy + BootstrapFewShot, same convention as dataset_dspy)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.artifact_discovery import ArtifactDiscovery
from tools.prompt_extraction.intake_process import INTAKE_PROCESS_PATH

from .config import load_settings
from .pipeline import HyperParamsPipeline


def _intake_hyper_params_message() -> str:
    if not INTAKE_PROCESS_PATH.is_file():
        raise ValueError(f"Intake file not found: {INTAKE_PROCESS_PATH}")
    data = json.loads(INTAKE_PROCESS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Intake file is not an object: {INTAKE_PROCESS_PATH}")
    message = str(data.get("hyper_params_dspy", "")).strip()
    if not message:
        raise ValueError("Empty intake key: hyper_params_dspy")
    return message


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGGREGATORS_DIR = _REPO_ROOT / "plugins" / "aggregators"


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyper-params config agent.")
    parser.add_argument("--execution-mode", help="Unused; accepted when called from a mode wrapper.")
    parser.add_argument(
        "--recompile",
        type=_parse_bool,
        default=None,
        metavar="{True,False}",
        help="--recompile=True: compile and save. --recompile=False: load saved agent.",
    )
    parser.add_argument(
        "--from-gui",
        action="store_true",
        help="Use the hyper_params_dspy value from intake_process.json.",
    )
    args = parser.parse_args()

    settings = load_settings()
    pipeline = HyperParamsPipeline(settings=settings, recompile=args.recompile)

    # Get the valid artifacts for aggragator
    valid_aggregators = ArtifactDiscovery(
        _AGGREGATORS_DIR,
        repo_root=_REPO_ROOT,
    ).discover()

    if not args.from_gui:
        print(f"Agent: {pipeline.HYPER_PARAMS_QUESTION}")
    if valid_aggregators:
        print(f"Agent: Discovered aggregator plugins: {valid_aggregators}")
    try:
        if args.from_gui:
            user_input = _intake_hyper_params_message()
        else:
            user_input = input("You: ").strip()
        result = pipeline.run(user_input, valid_aggregators)
    except json.JSONDecodeError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1
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


