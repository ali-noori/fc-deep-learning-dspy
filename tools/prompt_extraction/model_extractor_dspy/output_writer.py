"""Write the four model extractor fields to the JSON handoff file."""

from __future__ import annotations

from pathlib import Path

from .briefs import save_briefs
from .types import ModelExtractorBriefs


class ModelExtractorOutputWriter:
    def write(self, briefs: ModelExtractorBriefs) -> Path:
        return save_briefs(briefs.as_dict())
