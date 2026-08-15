from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .pipeline import ModelPluginPipeline


def main() -> int:
    # Parser to parse the arguments
    parser = argparse.ArgumentParser(
        description="Generate a PyTorch model architecture via DSPy (Groq → Gemini → Cohere → Mistral → Ollama)."
    )
    parser.add_argument("--request", "--prompt", dest="prompt", required=True)
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = parser.parse_args()

    if not args.prompt.strip():
        # If the prompt is empty, raise an error
        print("error: --request must be non-empty", file=sys.stderr)
        return 2

    # Get back to directoty back
    repo_root = Path(__file__).resolve().parents[2]
    
    # Create pipeline
    pipeline = ModelPluginPipeline(repo_root=repo_root, settings=load_settings())
    result = pipeline.run(args.prompt)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Backend: {result['backend']} ({result['model']})")
        print(f"Wrote model to: {result['output_model']}\n")
        print(result["architecture"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
