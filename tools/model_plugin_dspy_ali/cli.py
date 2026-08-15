"""
Command-line front door.

You type a prompt in the terminal (or paste one). This script hands that text to the
pipeline, then prints where the new Python file was saved—or the full JSON if you ask.
It is the same idea as clicking "Run" in an app, but for people who like the console.
"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import ModelPluginAliPipeline


def main() -> int:
    # Read what the user typed after.
    parser = argparse.ArgumentParser(
        description="Generate a PyTorch model architecture via DSPy (Groq → Gemini → Cohere → Mistral → Ollama)."
    )
    parser.add_argument("--request", "--prompt", dest="prompt", required=True)
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = parser.parse_args()

    if not args.prompt.strip():
        print("error: --request must be non-empty", file=sys.stderr)
        return 2

    # The pipeline asks the AI helpers and writes the `.py` file on disk.
    pipeline = ModelPluginAliPipeline()
    result = pipeline.run(args.prompt)

    if args.json:
        # Machine-friendly: everything in one JSON blob for scripts or logs.
        print(json.dumps(result, indent=2))
    else:
        # Human-friendly: short lines, then the raw generated code.
        print(f"Backend: {result['backend']} ({result['model']})")
        print(f"Wrote model to: {result['output_model']}\n")
        print(result["architecture"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
