"""CLI for trainer_dspy (DSPy + BootstrapFewShot, same convention as model_dspy)."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.artifact_discovery import ArtifactDiscovery

from .config import load_settings
from .pipeline import TrainerPipeline


# trainer artifact discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINERS_DIR = _REPO_ROOT / "plugins" / "trainers"

# dataloader artifact discovery
_DATALOADERS_DIR = _REPO_ROOT / "plugins" / "dataloaders"

# loss artifact discovery
_LOSSES_DIR = _REPO_ROOT / "plugins" / "loss"


def main() -> int:
    settings = load_settings()
    pipeline = TrainerPipeline(settings=settings)

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

        user_input = input("You: ").strip()
        result = pipeline.run(
            user_input,
            valid_trainers,
            valid_dataloaders,
            valid_losses,
        )
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
