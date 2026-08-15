"""CLI for config_plugin_dspy (natural language → config YAML)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .pipeline import ConfigPluginPipeline


def main() -> int:
    repo_default = Path(__file__).resolve().parents[2]
    settings = load_settings()
    default_out = repo_default.joinpath(
        *settings.output_path_parts,
        settings.output_filename,
    )

    parser = argparse.ArgumentParser(
        description="Turn a sentence into FeatureCloud config YAML (DSPy + config.maximum.yml)",
    )
    parser.add_argument("--repo-root", default=str(repo_default))
    parser.add_argument("--request", "--prompt", dest="request", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated YAML (default: tools/output/config/generated/)",
    )
    parser.add_argument(
        "--output-config",
        default=str(default_out),
        help="Path shown in summary JSON (default matches write location)",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip BootstrapFewShot (fewer LM calls; recommended for Groq rate limits)",
    )
    args = parser.parse_args()

    if not args.request.strip():
        print("error: --request must be non-empty", file=sys.stderr)
        return 2

    pipeline = ConfigPluginPipeline(
        repo_root=Path(args.repo_root),
        settings=settings,
        skip_bootstrap=args.no_bootstrap,
    )
    result = pipeline.run(
        args.request,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    print(f"Backend: {result.backend} ({result.model})")
    print(f"Wrote config to: {result.output_config}\n")

    summary = {
        "approach": result.approach,
        "template": result.template,
        "backend": result.backend,
        "model": result.model,
        "base_url": result.base_url,
        "output_files": result.output_files,
        "output_config": args.output_config,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
