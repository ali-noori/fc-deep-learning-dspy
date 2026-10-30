"""DSPy Signature + Module: natural language → train_config fields (LM only)."""

from __future__ import annotations

from .config import Settings
from .dspy_support import dspy
from .types import TrainConfigExtractionResult


class ParseTrainConfigSignature(dspy.Signature):
    """Extract train_config fields from the developer's answer.

    Outputs:
    - verbose: true/false (logging during training)
    - batch_size: positive integer training batch size
    - test_batch_size: positive integer test batch size
    - epochs: positive integer local epochs
    - lr: positive float learning rate
    - batch_count: positive integer (used by FedMMb)

    Decision rules:
    - Ignore typos, informal wording, and incomplete phrasing; infer the intended values.
    - If the user mentions unrelated extras (dataset paths, model architecture, FedAvg,
      trainer name), ignore them.
    - Do not invent missing required fields; only extract what is stated or clearly implied.
    - Synonyms: verbose / logging / print progress → verbose;
      batch / train batch / mini-batch → batch_size;
      test batch / eval batch → test_batch_size;
      epoch / local epochs → epochs;
      learning rate / step size → lr;
      batch_count / mini-batch count / FedMMb batches → batch_count.
    - verbose: true/yes/on → true; false/no/off/quiet → false.
    """

    user_message: str = dspy.InputField(
        desc=(
            "Developer natural-language answer about train_config; "
            "may be short, long, messy, incomplete, or misspelled, and may not "
            "use exact field names"
        )
    )
    verbose: str = dspy.OutputField(
        desc="Boolean as true or false: whether to print training logs."
    )
    batch_size: str = dspy.OutputField(
        desc="Positive integer: training batch size (e.g. 32)."
    )
    test_batch_size: str = dspy.OutputField(
        desc="Positive integer: test/eval batch size (e.g. 32)."
    )
    epochs: str = dspy.OutputField(
        desc="Positive integer: local training epochs (e.g. 1)."
    )
    lr: str = dspy.OutputField(
        desc="Positive float: learning rate (e.g. 0.1)."
    )
    batch_count: str = dspy.OutputField(
        desc="Positive integer: batch_count for FedMMb (e.g. 1)."
    )


def _parse_bool(raw: str) -> bool:
    text = str(raw or "").strip().strip('"').strip("'").lower()
    if text in ("true", "1", "yes", "y", "on"):
        return True
    if text in ("false", "0", "no", "n", "off", "quiet"):
        return False
    raise ValueError("verbose must be true or false.")


def _parse_positive_int(raw: str, field_name: str) -> int:
    text = str(raw or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError(f"{field_name} must be a positive integer.")
    try:
        value = int(float(text))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _parse_positive_float(raw: str, field_name: str) -> float:
    text = str(raw or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError(f"{field_name} must be a positive number.")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a positive number.") from exc
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive number.")
    return value


def _validate_extraction(
    verbose_raw: str,
    batch_size_raw: str,
    test_batch_size_raw: str,
    epochs_raw: str,
    lr_raw: str,
    batch_count_raw: str,
) -> tuple[bool, int, int, int, float, int]:
    verbose = _parse_bool(verbose_raw)
    batch_size = _parse_positive_int(batch_size_raw, "batch_size")
    test_batch_size = _parse_positive_int(test_batch_size_raw, "test_batch_size")
    epochs = _parse_positive_int(epochs_raw, "epochs")
    lr = _parse_positive_float(lr_raw, "lr")
    batch_count = _parse_positive_int(batch_count_raw, "batch_count")
    return verbose, batch_size, test_batch_size, epochs, lr, batch_count


class TrainConfigModule(dspy.Module):
    """Chain-of-Thought extractor for train_config fields."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self.extract = dspy.ChainOfThought(ParseTrainConfigSignature)

    def forward(self, user_message: str) -> TrainConfigExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")

        result: TrainConfigExtractionResult | None = None
        last_error: Exception | None = None

        for _ in range(self._settings.train_config_extract_max_retries):
            try:
                prediction = self.extract(user_message=current_request)
                (
                    verbose,
                    batch_size,
                    test_batch_size,
                    epochs,
                    lr,
                    batch_count,
                ) = _validate_extraction(
                    str(getattr(prediction, "verbose", "") or ""),
                    str(getattr(prediction, "batch_size", "") or ""),
                    str(getattr(prediction, "test_batch_size", "") or ""),
                    str(getattr(prediction, "epochs", "") or ""),
                    str(getattr(prediction, "lr", "") or ""),
                    str(getattr(prediction, "batch_count", "") or ""),
                )

                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                result = TrainConfigExtractionResult(
                    verbose=verbose,
                    batch_size=batch_size,
                    test_batch_size=test_batch_size,
                    epochs=epochs,
                    lr=lr,
                    batch_count=batch_count,
                    backend=backend,
                    model=model,
                    base_url=base_url,
                )
                break
            except Exception as exc:
                last_error = exc
                current_request = (
                    current_request
                    + "\n\nYour last answer was invalid: "
                    + str(exc)
                    + "\nReply with verbose, batch_size, test_batch_size, "
                    + "epochs, lr, and batch_count copied exactly from the "
                    + "developer answer."
                )

        if result is None:
            raise RuntimeError(
                "Train config extraction failed after LM retries."
            ) from last_error

        return result
