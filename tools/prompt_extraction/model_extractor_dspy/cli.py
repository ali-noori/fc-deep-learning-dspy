"""CLI for model_extractor_dspy. Reads one model request from PowerShell."""

from __future__ import annotations

import argparse
import sys

from .briefs import briefs_path
from .config import load_settings, resolve_recompile
from .pipeline import ModelExtractorPipeline


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Model extractor agent.")
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
    pipeline = ModelExtractorPipeline(settings=settings, recompile=recompile)

    print(f"Agent: {pipeline.MODEL_EXTRACTOR_QUESTION}")
    try:
        user_input = input("You: ").strip()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
