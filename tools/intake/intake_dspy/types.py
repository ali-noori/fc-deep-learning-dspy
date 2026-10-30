"""Shared data shapes for the intake_dspy pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntakeBriefs:
    execution_mode_dspy: str
    dataset_dspy: str
    hyper_params_dspy: str
    trainer_dspy: str
    train_config_dspy: str

    def as_dict(self) -> dict[str, str]:
        return {
            "execution_mode_dspy": self.execution_mode_dspy,
            "dataset_dspy": self.dataset_dspy,
            "hyper_params_dspy": self.hyper_params_dspy,
            "trainer_dspy": self.trainer_dspy,
            "train_config_dspy": self.train_config_dspy,
        }


@dataclass(frozen=True)
class IntakeExtractionResult:
    briefs: IntakeBriefs
    backend: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class IntakeResult:
    briefs: IntakeBriefs
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
