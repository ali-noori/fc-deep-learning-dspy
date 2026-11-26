"""DSPy Signature + Module: one model request -> four copied fields."""

from __future__ import annotations

from .config import Settings
from .dspy_support import dspy
from .types import ModelExtractorBriefs, ModelExtractorExtractionResult


class ParseModelExtractorSignature(dspy.Signature):
    """Map the user's words into four fields. Copy only what they wrote.

    Do not investigate the message, do not assume a missing fact, and do not
    draw a conclusion the user did not write. Do not add .py and do not turn
    their words into Python.
    The user does not have to say whether they want a new model.
    Put layer wording they wrote in architecture_description.
    Put a model name they wrote in model_selection, copied as written.
    Put the input-feature value they wrote in in_feature, copied as written.
    Put the class-count value they wrote in n_classes, copied as written.
    A number may be digits or words. "number of input is 1 and number of class is ten" maps to 1 and ten.
    "three input channels" maps to three. "seven hundred eighty-four" stays the whole phrase.
    Do not turn a word into a digit, and do not turn a digit into a word.
    Leave a field empty when they did not write that fact.
    """

    user_message: str = dspy.InputField(
        desc="The user's words. Copy the model facts they wrote. Do not infer facts they did not write."
    )
    architecture_description: str = dspy.OutputField(
        desc="Layer wording copied from the user. Empty when they did not write any."
    )
    model_selection: str = dspy.OutputField(
        desc="Model name copied as written. Empty when they did not write one."
    )
    in_feature: str = dspy.OutputField(
        desc="Input-feature value copied as written, such as 1, one, or seven hundred eighty-four."
    )
    n_classes: str = dspy.OutputField(
        desc="Class-count value copied as written, such as 10, ten, or thirty-two."
    )


def _clean(raw: object) -> str:
    return str(raw or "").strip().strip('"').strip("'")


def _require_text(raw: object, field_name: str) -> str:
    text = _clean(raw)
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    return text


def _validate_extraction(
    architecture_description: object,
    model_selection: object,
    in_feature: object,
    n_classes: object,
) -> ModelExtractorBriefs:
    return ModelExtractorBriefs(
        architecture_description=_clean(architecture_description),
        model_selection=_clean(model_selection),
        in_feature=_require_text(in_feature, "in_feature"),
        n_classes=_require_text(n_classes, "n_classes"),
    )


class ModelExtractorModule(dspy.Module):
    """Chain-of-Thought extractor that only copies model fields."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self.extract = dspy.ChainOfThought(ParseModelExtractorSignature)

    def forward(self, user_message: str) -> ModelExtractorExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")

        result: ModelExtractorExtractionResult | None = None
        last_error: Exception | None = None

        for _ in range(self._settings.model_extractor_max_retries):
            try:
                prediction = self.extract(user_message=current_request)
                briefs = _validate_extraction(
                    getattr(prediction, "architecture_description", ""),
                    getattr(prediction, "model_selection", ""),
                    getattr(prediction, "in_feature", ""),
                    getattr(prediction, "n_classes", ""),
                )
                lm = dspy.settings.lm
                result = ModelExtractorExtractionResult(
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
                    + "\nCopy only the words the user wrote. Leave a field empty "
                    + "when they did not write it. Do not add .py. "
                    + "Copy a number word as written. Do not turn ten into 10."
                )

        if result is None:
            raise RuntimeError("Model extractor extraction failed after LM retries.") from last_error
        return result
