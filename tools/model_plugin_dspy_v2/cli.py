"""CLI for DSPy v2 architecture generation (same flags as majid/ali)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .pipeline import ModelPluginV2Pipeline


def main() -> int:
    # Create a parser to parse the arguments
    parser = argparse.ArgumentParser(
        description=(
            "Generate an architecture for a PyTorch model."
        )
    )
    #"--request", "--prompt": 
    # These define the flags the user can type in the terminal. 
    # They are aliases, meaning the user can run the script using 
    # either python script.py --request "hello" or python script.py --prompt "hello"
    
    # dest="prompt": 
    # This specifies the variable name where the input will be stored. 
    # Regardless of whether the user typed --request or --prompt, 
    # the value will be saved inside the parsed arguments object as .prompt (e.g., args.prompt).
    parser.add_argument("--request", "--prompt", dest="prompt", required=True)
    args = parser.parse_args()

    if not args.prompt.strip():
        print("error: --request must be non-empty", file=sys.stderr)
        return 2

    # parents[2]: This navigates up the folder tree.
    ## parents[0] is the folder containing the current script.
    ## parents[1] is the grandparent folder (one level up).
    ## parents[2] is the great-grandparent folder (two levels up).
    repo_root = Path(__file__).resolve().parents[2]
    
    # some paramters like the candidate model names and their urls
    settings = load_settings()
    
    # Create pipeline
    pipeline = ModelPluginV2Pipeline(
        repo_root=repo_root,
        settings=settings,
    )

    # Run the pipeline
    # eventually it creates model implementation
    result = pipeline.run(args.prompt)

    # Print the result
    print(f"Backend: {result['backend']} ({result['model']})")
    print(f"Wrote model to: {result['output_model']}\n")
    print(result["architecture"])
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
