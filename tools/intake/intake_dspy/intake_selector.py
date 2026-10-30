"""DSPy Signature + Module: one user request -> briefs for the other agents."""

from __future__ import annotations

from .config import Settings
from .dspy_support import dspy
from .types import IntakeBriefs, IntakeExtractionResult


class ParseIntakeSignature(dspy.Signature):
    """Gather the user's words into one brief per downstream agent.

    Do not rewrite the request into YAML. Copy the relevant phrases, even when
    they are scattered through the message. Keep enough of the original wording
    that the next agent can still interpret it.

    Outputs:
    - execution_mode_dspy: phrases about federated, simulation, or centralized.
    - dataset_dspy: folders, file names, and split/fold layout.
    - hyper_params_dspy: communication rounds or max_iter, class count, aggregator.
    - trainer_dspy: trainer, data loader, and loss.
    Ignore the model name, input channels, and whether a new architecture should
    be generated. Those facts are not outputs of this agent.
    - train_config_dspy: verbose, batch sizes, local epochs, learning rate, batch_count.
      Do not put communication rounds into this brief.
    """

    user_message: str = dspy.InputField(
        desc="The full natural-language experiment request. Facts may be out of order."
    )
    execution_mode_dspy: str = dspy.OutputField(
        desc="User phrases about federated, simulation, or one-machine centralized training."
    )
    dataset_dspy: str = dspy.OutputField(
        desc="User phrases about data folders, train/test file names, and fold directory."
    )
    hyper_params_dspy: str = dspy.OutputField(
        desc="User phrases about rounds, number of classes, and FedAvg or another aggregator."
    )
    trainer_dspy: str = dspy.OutputField(
        desc="User phrases about trainer, data loader, and loss."
    )
    train_config_dspy: str = dspy.OutputField(
        desc="User phrases about logging, batch size, epochs, learning rate, and batch_count."
    )


def _clean(raw: object) -> str:
    return str(raw or "").strip().strip('"').strip("'")


def _require_text(raw: object, field_name: str) -> str:
    text = _clean(raw)
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    return text


def _validate_extraction(
    execution_mode_dspy: object,
    dataset_dspy: object,
    hyper_params_dspy: object,
    trainer_dspy: object,
    train_config_dspy: object,
) -> IntakeBriefs:
    return IntakeBriefs(
        execution_mode_dspy=_require_text(execution_mode_dspy, "execution_mode_dspy"),
        dataset_dspy=_require_text(dataset_dspy, "dataset_dspy"),
        hyper_params_dspy=_require_text(hyper_params_dspy, "hyper_params_dspy"),
        trainer_dspy=_require_text(trainer_dspy, "trainer_dspy"),
        train_config_dspy=_require_text(train_config_dspy, "train_config_dspy"),
    )


class IntakeModule(dspy.Module):
    """Chain-of-Thought extractor that only gathers briefs."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self.extract = dspy.ChainOfThought(ParseIntakeSignature)

    def forward(self, user_message: str) -> IntakeExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")

        result: IntakeExtractionResult | None = None
        last_error: Exception | None = None

        for _ in range(self._settings.intake_extract_max_retries):
            try:
                prediction = self.extract(user_message=current_request)
                briefs = _validate_extraction(
                    getattr(prediction, "execution_mode_dspy", ""),
                    getattr(prediction, "dataset_dspy", ""),
                    getattr(prediction, "hyper_params_dspy", ""),
                    getattr(prediction, "trainer_dspy", ""),
                    getattr(prediction, "train_config_dspy", ""),
                )
                lm = dspy.settings.lm
                result = IntakeExtractionResult(
                    briefs=briefs,
                    backend=getattr(lm, "last_backend", None) or "unknown",
                    model=getattr(lm, "last_model", None) or getattr(lm, "model", "unknown"),
                    base_url=getattr(lm, "last_base_url", None),
                )
                break
            except Exception as exc:
                last_error = exc
                current_request = (
                    current_request
                    + "\n\nYour last answer was invalid: "
                    + str(exc)
                    + "\nCopy the user's phrases into each brief. "
                    + "Do not answer with a model name or a new-architecture yes or no."
                )

        if result is None:
            raise RuntimeError("Intake extraction failed after LM retries.") from last_error
        return result
