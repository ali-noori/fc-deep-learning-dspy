"""Write the five experiment splitter briefs to the JSON handoff file."""

from __future__ import annotations

from pathlib import Path

from .briefs import save_briefs
from .types import ExperimentSplitterBriefs


class ExperimentSplitterOutputWriter:
    def write(self, briefs: ExperimentSplitterBriefs) -> Path:
        return save_briefs(briefs.as_dict())
