"""Join both intake briefs and the second-window choices into one JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from tools.intake.intake_dspy.briefs import load_briefs as load_intake_briefs
from tools.intake.intake_model_dspy.briefs import load_briefs as load_model_briefs

INTAKE_PROCESS_PATH = Path(__file__).resolve().parent / "output" / "intake_process.json"


def check_compatible(merged: dict[str, str]) -> None:
    new_architecture = merged["new_architecture"]
    description = merged["architecture_description"]
    selection = merged["model_selection"]
    name = merged["new_architecture_name"]
    save = merged["save_new_architecture"]
    if new_architecture == "Yes":
        compatible = bool(description) and not selection and bool(name) and save == "Yes"
    elif new_architecture == "No":
        compatible = not description and bool(selection) and not name and save == "No"
    else:
        compatible = False
    if not compatible:
        raise ValueError(
            "Incompatible intake choices: "
            f"new_architecture={new_architecture!r}, "
            f"architecture_description={description!r}, "
            f"model_selection={selection!r}, "
            f"new_architecture_name={name!r}, "
            f"save_new_architecture={save!r}."
        )


def build_final_intake(
    new_architecture: str,
    new_architecture_name: str,
    save_new_architecture: str,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    merged.update(load_intake_briefs())
    merged.update(load_model_briefs())
    merged["new_architecture"] = new_architecture.strip()
    merged["new_architecture_name"] = new_architecture_name.strip()
    merged["save_new_architecture"] = save_new_architecture.strip()
    check_compatible(merged)
    if merged["new_architecture"] == "Yes":
        model_name = merged["new_architecture_name"]
    else:
        model_name = merged["model_selection"]
    merged["model_dspy"] = (
        f"Use model = {model_name} and input_feature = {merged['in_feature']} "
        f"and n_classes = {merged['n_classes']}"
    )
    del merged["model_selection"]
    del merged["in_feature"]
    del merged["n_classes"]
    # With an existing model ("No"), these two keys are empty by design.
    allowed_empty = (
        {"architecture_description", "new_architecture_name"}
        if merged["new_architecture"] == "No"
        else set()
    )
    empty_keys = [
        key
        for key, value in merged.items()
        if key not in allowed_empty and not str(value).strip()
    ]
    if empty_keys:
        raise ValueError("Empty intake key: " + ", ".join(empty_keys))

    INTAKE_PROCESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTAKE_PROCESS_PATH.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merged
