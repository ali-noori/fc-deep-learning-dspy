"""Load config.maximum.yml as the merge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import Settings


def load_template_config(repo_root: Path, settings: Settings) -> dict[str, Any]:
    cfg_path = repo_root / settings.template_filename
    if not cfg_path.exists():
        return {"fc_deep": {}}

    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if "fc_deep" not in loaded:
        loaded = {"fc_deep": loaded}
    return loaded
