"""DSPy Signature + Module: natural language → fed_hyper_params fields (LM only)."""

from __future__ import annotations

from .config import BUILTIN_AGGREGATORS, Settings
from .dspy_support import dspy
from .types import FedHyperParamsExtractionResult


def format_known_aggregators(aggregator_artifacts: dict[str, str]) -> str:
    """Format built-in and discovered plugin aggregators for the LM."""
    builtins = ", ".join(BUILTIN_AGGREGATORS)
    builtin_hint = f"built-in names without .py ({builtins})"
    if not aggregator_artifacts:
        return builtin_hint
    plugins = ", ".join(
        f"{name} -> {path}" for name, path in aggregator_artifacts.items()
    )
    return f"{builtin_hint}; plugins: {plugins}"


class ParseFedHyperParamsSignature(dspy.Signature):
    """Extract federated hyperparameter fields from the developer's answer.

    Outputs:
    - max_iter: positive integer federated communication rounds
    - n_classes: positive integer class count (n_class is a synonym)
    - federated_model: built-in aggregator name OR custom plugin reference

    Decision rules:
    - Ignore typos, informal wording, and incomplete phrasing; infer the intended values.
    - If the user mentions unrelated extras (dataset paths, model architecture, loss, trainer), ignore them.
    - Do not invent missing required fields; only extract what is stated or clearly implied.
    - Synonyms: rounds / communication rounds / FL iterations → max_iter;
      classes / n_class / n_classes → n_classes.
    - Built-in vs plugin (critical):
      * FedAvg / "built-in" / "library" / "not the .py file" → FedAvg
      * FedAvg.py / "custom" / "plugin" / plugins/aggregators/FedAvg.py → plugin
        (FedAvg.py or full path; resolution may map to plugins/aggregators/FedAvg.py)
      * Bare "FedAvg" without .py/plugin/custom wording defaults to built-in FedAvg.
      * Do not convert built-in FedAvg into FedAvg.py unless the user asks for the plugin.
    - Choose federated_model only from known built-ins or discovered plugins in known_aggregators.
    - max_iter and n_classes must be positive integers.
    """

    user_message: str = dspy.InputField(
        desc=(
            "Developer natural-language answer about fed_hyper_params; "
            "may be short, long, messy, incomplete, or misspelled, and may not "
            "use exact field names"
        )
    )
    known_aggregators: str = dspy.InputField(
        desc=(
            "Allowed aggregators: built-in names without .py (e.g. FedAvg) and "
            "discovered plugins as filename -> repo-relative path "
            "(e.g. FedAvg.py -> plugins/aggregators/FedAvg.py)"
        )
    )
    max_iter: str = dspy.OutputField(
        desc="Positive integer: number of federated communication rounds (e.g. 10)."
    )
    n_classes: str = dspy.OutputField(
        desc="Positive integer: number of classes (e.g. 10). Accept n_class as synonym."
    )
    federated_model: str = dspy.OutputField(
        desc=(
            "Built-in aggregator name (e.g. FedAvg) OR plugin filename/path "
            "(e.g. FedAvg.py or plugins/aggregators/FedAvg.py). "
            "Do not confuse built-in FedAvg with plugin FedAvg.py."
        )
    )


def _looks_like_plugin_reference(model_text: str) -> bool:
    return model_text.endswith(".py") or "/" in model_text or "\\" in model_text


def _resolve_federated_model(
    federated_model_raw: str,
    aggregator_artifacts: dict[str, str],
) -> str:
    federated_model_raw_text = str(federated_model_raw or "").strip().strip('"').strip("'")
    if not federated_model_raw_text:
        raise ValueError("federated_model must not be empty.")

    # Plugin filename (dict key), e.g. FedAvg.py -> plugins/aggregators/FedAvg.py
    if federated_model_raw_text in aggregator_artifacts:
        return aggregator_artifacts[federated_model_raw_text]

    # Already a full plugin path (dict value)
    if federated_model_raw_text in aggregator_artifacts.values():
        return federated_model_raw_text

    if _looks_like_plugin_reference(federated_model_raw_text):
        if aggregator_artifacts:
            allowed = ", ".join(aggregator_artifacts.keys())
            raise ValueError(
                f"federated_model {federated_model_raw_text!r} is not a discovered plugin. "
                f"Allowed plugins: {allowed}"
            )
        raise ValueError(
            f"federated_model {federated_model_raw_text!r} looks like a plugin but none were "
            "discovered under plugins/aggregators."
        )
    if federated_model_raw_text in BUILTIN_AGGREGATORS:
        return federated_model_raw_text

    # federated_model_raw_text is neither in BUILTIN_AGGREGATORS nor in the aggregator_artifacts list
    allowed_builtins = ", ".join(BUILTIN_AGGREGATORS)
    allowed_plugins = ", ".join(aggregator_artifacts.keys()) or "(none)"
    raise ValueError(
        f"federated_model {federated_model_raw_text!r} is not a valid aggregator. "
        f"Allowed built-ins: {allowed_builtins}. Allowed plugins: {allowed_plugins}"
    )


def _validate_extraction(
    max_iter_raw: str,
    n_classes_raw: str,
    federated_model_raw: str,
    aggregator_artifacts: dict[str, str] | None = None,
) -> tuple[int, int, str]:
    max_iter_text = str(max_iter_raw or "").strip()
    n_classes_text = str(n_classes_raw or "").strip()
    model_text = str(federated_model_raw or "").strip().strip('"').strip("'")

    if not max_iter_text or not n_classes_text or not model_text:
        raise ValueError(
            "Missing max_iter, n_classes (or n_class), or federated_model."
        )

    try:
        max_iter = int(float(max_iter_text))
        n_classes = int(float(n_classes_text))
    except ValueError as exc:
        raise ValueError("max_iter and n_classes must be positive integers.") from exc

    if max_iter < 1 or n_classes < 1:
        raise ValueError("max_iter and n_classes must be positive integers.")

    # Check if the returned aggregator name is a valid aggregator
    federated_model = _resolve_federated_model(
        federated_model_raw,
        aggregator_artifacts or {},
    )

    return max_iter, n_classes, federated_model


class FedHyperParamsConfigModule(dspy.Module):
    """Chain-of-Thought extractor for fed_hyper_params config fields."""

    def __init__(
        self,
        settings: Settings,
        aggregator_artifacts: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._aggregator_artifacts = dict(aggregator_artifacts or {})
        self.extract = dspy.ChainOfThought(ParseFedHyperParamsSignature)

    def forward(
        self,
        user_message: str,
        known_aggregators: str = "",
    ) -> FedHyperParamsExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")

        known = known_aggregators.strip() or format_known_aggregators(
            self._aggregator_artifacts
        )

        result: FedHyperParamsExtractionResult | None = None
        last_error: Exception | None = None

        for _ in range(self._settings.hyper_params_extract_max_retries):
            try:
                prediction = self.extract(
                    user_message=current_request,
                    known_aggregators=known,
                )
                max_iter, n_classes, federated_model = _validate_extraction(
                    str(getattr(prediction, "max_iter", "") or ""),
                    str(getattr(prediction, "n_classes", "") or ""),
                    str(getattr(prediction, "federated_model", "") or ""),
                    self._aggregator_artifacts,
                )

                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                result = FedHyperParamsExtractionResult(
                    max_iter=max_iter,
                    n_classes=n_classes,
                    federated_model=federated_model,
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
                    + "\nReply with max_iter, n_class (or n_classes), and "
                    + "federated_model copied exactly from the developer answer. "
                    + f"Allowed aggregators: {known}"
                )

        if result is None:
            raise RuntimeError(
                "Hyperparameter config extraction failed after LM retries."
            ) from last_error

        return result
