"""CLI for model_dspy (DSPy + BootstrapFewShot, same convention as hyper_params_dspy)."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.artifact_discovery import ArtifactDiscovery

from .config import load_settings
from .pipeline import ModelPipeline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _REPO_ROOT / "plugins" / "models"


def main() -> int:
    settings = load_settings()
    pipeline = ModelPipeline(settings=settings)

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
