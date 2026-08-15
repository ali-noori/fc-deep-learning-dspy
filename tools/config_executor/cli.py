"""CLI for config_executor."""

from __future__ import annotations

import argparse
import sys

from .execute import MODES, ConfigExecutor


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the final generated config.")
    parser.add_argument(
        "--execution-mode",
        required=True,
        choices=MODES,
        help="Execution mode from execution_mode_dspy.",
    )
    parser.add_argument(
        "--app-image",
        default="featurecloud.ai/fc_deep_networks1",
        help="FeatureCloud app image.",
    )
    parser.add_argument(
        "--controller-host",
        default="http://localhost:8000",
        help="FeatureCloud controller URL.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep staged files in data/tools/config after a successful run.",
    )
    args = parser.parse_args()

    executor = ConfigExecutor(
        app_image=args.app_image,
        controller_host=args.controller_host,
        keep_staging=args.keep_staging,
    )

    try:
        return executor.execute(args.execution_mode)
    except ValueError as exc:
        print(f"ConfigExecutor: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
