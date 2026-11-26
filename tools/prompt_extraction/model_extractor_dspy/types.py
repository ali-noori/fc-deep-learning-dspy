"""Shared data shapes for the model_extractor_dspy pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelExtractorBriefs:
    architecture_description: str
    model_selection: str
    in_feature: str
    n_classes: str

    def as_dict(self) -> dict[str, str]:
        return {
            "architecture_description": self.architecture_description,
            "model_selection": self.model_selection,
            "in_feature": self.in_feature,
            "n_classes": self.n_classes,
        }


@dataclass(frozen=True)
class ModelExtractorExtractionResult:
    briefs: ModelExtractorBriefs
    backend: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class ModelExtractorResult:
    briefs: ModelExtractorBriefs
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
