"""
Create a model plugin from a single natural-language --request.

Example:
python -m tools.model_plugin_dspy.cli --request "$(cat prompt.txt)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import ModelPluginPipeline


def main() -> int:
    p = argparse.ArgumentParser(
        description="DSPy model-plugin generator: --request must contain all configuration."
    )
    p.add_argument(
        "--request",
        required=True,
        help=(
            "Full specification: task, n_classes / class count, input shape / in_features, "
            "layer blocks, transfer backbone name (if any), etc."
        ),
    )
    args = p.parse_args()
    if not str(args.request).strip():
        print("error: --request must be non-empty", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    pipeline = ModelPluginPipeline(repo_root=str(repo_root))
    summary = pipeline.run(request=args.request)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
