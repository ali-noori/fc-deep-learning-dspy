"""CLI for trainer_dspy (DSPy + BootstrapFewShot, same convention as model_dspy)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.artifact_discovery import ArtifactDiscovery
from tools.prompt_extraction.intake_process import INTAKE_PROCESS_PATH

from .config import load_settings
from .pipeline import TrainerPipeline


def _intake_trainer_message() -> str:
    if not INTAKE_PROCESS_PATH.is_file():
        raise ValueError(f"Intake file not found: {INTAKE_PROCESS_PATH}")
    data = json.loads(INTAKE_PROCESS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Intake file is not an object: {INTAKE_PROCESS_PATH}")
    message = str(data.get("trainer_dspy", "")).strip()
    if not message:
        raise ValueError("Empty intake key: trainer_dspy")
    return message


# trainer artifact discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINERS_DIR = _REPO_ROOT / "plugins" / "trainers"

# dataloader artifact discovery
_DATALOADERS_DIR = _REPO_ROOT / "plugins" / "dataloaders"

# loss artifact discovery
_LOSSES_DIR = _REPO_ROOT / "plugins" / "loss"


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Trainer config agent.")
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
        help="Use the trainer_dspy value from intake_process.json.",
    )
    args = parser.parse_args()

    settings = load_settings()
    pipeline = TrainerPipeline(settings=settings, recompile=args.recompile)

    if not args.from_gui:
        print(f"Agent: {pipeline.TRAINER_QUESTION}")
    try:
        
        valid_trainers = ArtifactDiscovery(
            _TRAINERS_DIR,
            repo_root=_REPO_ROOT,
        ).discover()

        valid_dataloaders = ArtifactDiscovery(
            _DATALOADERS_DIR,
            repo_root=_REPO_ROOT,
        ).discover()
        
        valid_losses = ArtifactDiscovery(
            _LOSSES_DIR,
            repo_root=_REPO_ROOT,
        ).discover()

        if valid_trainers:
            print(f"Agent: Discovered trainer plugins: {valid_trainers}")
        if valid_dataloaders:
            print(f"Agent: Discovered dataloader plugins: {valid_dataloaders}")
        if valid_losses:
            print(f"Agent: Discovered loss plugins: {valid_losses}")

        if args.from_gui:
            user_input = _intake_trainer_message()
        else:
            user_input = input("You: ").strip()
        result = pipeline.run(
            user_input,
            valid_trainers,
            valid_dataloaders,
            valid_losses,
        )
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
