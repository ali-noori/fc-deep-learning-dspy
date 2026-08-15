"""
Command-line entry: ``python -m tools.dspy.cli --request "..."``.

Parses flags, runs ``ConfigAgentPipeline``, prints JSON paths. All DSPy behavior
lives in ``pipeline.py`` / ``modules.py``. Longer explanation: DSPY_IMPLEMENTATION.md.
"""
import argparse
import json
from pathlib import Path

from .optimization import GENERATED_CONFIG_FILENAME, default_generated_output_dir
from .pipeline import ConfigAgentPipeline


def main():
	repo_default = Path(__file__).resolve().parents[2]
	default_cfg = default_generated_output_dir(repo_default) / GENERATED_CONFIG_FILENAME

	p = argparse.ArgumentParser(description="Turn a sentence into config yaml (DSPy + template)")
	p.add_argument("--repo-root", default=str(repo_default))
	p.add_argument("--request", required=True, help="what you want changed in plain English")
	p.add_argument(
		"--output-dir",
		default=None,
		help="where to write config.generated.dspy.yml (default under tools/output/...)",
	)
	p.add_argument("--output-config", default=str(default_cfg), help="only used for printing summary path")
	p.add_argument(
		"--no-bootstrap",
		action="store_true",
		help="don't run BootstrapFewShot (way fewer API calls to Groq)",
	)

	args = p.parse_args()

	skip = True if args.no_bootstrap else None
	pipeline = ConfigAgentPipeline(repo_root=args.repo_root, skip_bootstrap=skip)
	pipeline.run(args.request, output_dir=args.output_dir)

	if args.output_dir is None:
		out_dir = default_generated_output_dir(args.repo_root)
	else:
		out_dir = Path(args.output_dir)

	print(
		json.dumps(
			{
				"approach": "dspy",
				"template": "config.maximum.yml",
				"output_files": {"config": str(out_dir / GENERATED_CONFIG_FILENAME)},
			},
			indent=2,
		)
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
