"""CLI for model_dspy (DSPy + BootstrapFewShot, same convention as hyper_params_dspy)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.artifact_discovery import ArtifactDiscovery

from .config import load_settings
from .pipeline import ModelPipeline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _REPO_ROOT / "plugins" / "models"


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Model config agent.")
    parser.add_argument("--execution-mode", help="Unused; accepted when called from a mode wrapper.")
    parser.add_argument(
        "--recompile",
        type=_parse_bool,
        default=None,
        metavar="{True,False}",
        help="--recompile=True: compile and save. --recompile=False: load saved agent.",
    )
    args = parser.parse_args()

    settings = load_settings()
    pipeline = ModelPipeline(settings=settings, recompile=args.recompile)

    print(f"Agent: {pipeline.MODEL_QUESTION}")
    try:
        valid_models = ArtifactDiscovery(
            _MODELS_DIR,
            repo_root=_REPO_ROOT,
        ).discover()

        if valid_models:
            print(f"Agent: Discovered model plugins: {valid_models}")
        user_input = input("You: ").strip()
        result = pipeline.run(user_input, valid_models)
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
