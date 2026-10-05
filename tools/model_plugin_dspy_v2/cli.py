"""CLI for DSPy v2 architecture generation (interactive, same convention as the other agents)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_settings
from .pipeline import ModelPluginV2Pipeline


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")

# Question printed before reading the architecture description from the user.
# Kept here (not in config.py) since this is the only place that prints it.
ARCHITECTURE_QUESTION = (
    "Please describe the model architecture you want "
    "(layers, in_features, n_classes, activations, etc.).\n"
    "You can type/paste multiple lines; finish with an empty line."
)


def _read_multiline_input() -> str:
    """
    Read the developer's answer interactively (like the other DSPy agents),
    but allow multiple lines, since architecture prompts here are typically
    written one layer per line (see fewshot_examples.py). Reading stops at
    the first blank line after some text was typed, or at EOF.
    """
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            # No content yet: allow leading blank lines instead of stopping immediately.
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> int:
    # Same --recompile convention as the other DSPy agents (e.g. model_dspy):
    # True = compile with BootstrapFewShot and save; False = load saved agent.
    parser = argparse.ArgumentParser(description="Generate a PyTorch model architecture.")
    parser.add_argument(
        "--recompile",
        type=_parse_bool,
        default=None,
        metavar="{True,False}",
        help="--recompile=True: compile and save. --recompile=False: load saved agent.",
    )
    args = parser.parse_args()

    print(f"Agent: {ARCHITECTURE_QUESTION}")

    # Same idea as the other DSPy agents (e.g. trainer_dspy, model_dspy):
    # read the developer's free-text answer interactively instead of a CLI flag.
    print("You: ", end="", flush=True)
    user_input = _read_multiline_input()

    if not user_input:
        print("Agent: architecture description must be non-empty", file=sys.stderr)
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
        recompile=args.recompile,
    )

    # Run the pipeline
    # eventually it creates model implementation
    result = pipeline.run(user_input)

    # Print the result
    print(f"Backend: {result['backend']} ({result['model']})")
    retrieved = result.get("retrieved_paths") or []
    if retrieved:
        print("Retrieved plugins: " + ", ".join(retrieved))
    print(f"Wrote model to: {result['output_model']}\n")
    print(result["architecture"])
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
