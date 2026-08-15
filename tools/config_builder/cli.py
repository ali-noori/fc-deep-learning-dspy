"""CLI entrypoint for building final generated config."""

from __future__ import annotations

import argparse
import sys

from .builder import MODES, build_config, resolve_execution_mode, write_generated_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final generated config.yml files.")
    parser.add_argument(
        "--execution-mode",
        choices=MODES,
        help=(
            "Optional override; must match execution_mode in "
            "tools/sub_config_dspy/dataset_dspy/output/output_dataset_dspy.yml. "
            "If omitted, mode is read from that file."
        ),
    )
    args = parser.parse_args()

    try:
        mode = resolve_execution_mode(args.execution_mode)
        config = build_config(mode)
        out_path = write_generated_config(config, mode)
        print(f"ConfigBuilder: built mode={mode}")
        print(f"ConfigBuilder: wrote {out_path}")
    except ValueError as exc:
        print(f"ConfigBuilder: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ConfigBuilder: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

