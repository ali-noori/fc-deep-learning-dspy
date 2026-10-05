"""DSPy Signature + Module: natural language → execution mode (LM only)."""

from __future__ import annotations

from .config import EXECUTION_MODES, Settings
from .dspy_support import dspy
from .types import ExecutionModeResult


class ParseExecutionModeSignature(dspy.Signature):
    """Choose exactly one FeatureCloud execution mode from the developer's answer.

    Modes:
    - federated: real multi-client / distributed training across separate clients or sites
    - simulation: local fake multi-client / dry-run testing without real remote clients
    - centralized: single-machine / single-server training with one dataset (no federation)

    Decision rules:
    - Ignore typos, incomplete words, informal wording; infer the intended mode.
    - If the user mentions many requirements (model, loss, data, etc.), still pick only the execution mode.
    - If they want federated behavior but with local/fake clients or a dry-run first → simulation.
    - If they want real separate clients/sites that keep data local → federated.
    - If they want one machine / one dataset / no clients → centralized.
    - Reply with exactly one label word only.
    """

    user_message: str = dspy.InputField(
        desc=(
            "Developer's natural-language answer about how to run the app; "
            "may be short, long, informal, incomplete, or misspelled"
        )
    )
    execution_mode: str = dspy.OutputField(
        desc=(
            "Exactly one lowercase label only: federated, simulation, or centralized"
        )
    )


def _validate_lm_mode(raw_mode: str) -> str:
    """Accept only a valid mode string returned by the LM (no keyword scan of user text)."""
    cleaned = str(raw_mode or "").strip().lower()
    if cleaned in EXECUTION_MODES:
        return cleaned
    raise ValueError(
        f"Language model returned invalid mode {raw_mode!r}. "
        f"Expected one of: {', '.join(EXECUTION_MODES)}."
    )


class ExecutionModeSelectorModule(dspy.Module):
    """Chain-of-Thought selector for federated | simulation | centralized."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self.select = dspy.ChainOfThought(ParseExecutionModeSignature)

    def forward(self, user_message: str) -> ExecutionModeResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")

        result: ExecutionModeResult | None = None
        last_error: Exception | None = None

        # Retry with feedback technique:
        # This is a technique to improve the accuracy of the model by providing feedback to the model
        # The model will be retried multiple times with the feedback until it is able to provide a valid answer
        for _ in range(self._settings.mode_select_max_retries):
            try:
                prediction = self.select(user_message=current_request)
                raw = getattr(prediction, "execution_mode", None)
                mode = _validate_lm_mode(str(raw or ""))

                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                result = ExecutionModeResult(
                    execution_mode=mode,
                    backend=backend,
                    model=model,
                    base_url=base_url,
                )
                break
            except Exception as exc:
                last_error = exc
                current_request = (
                    current_request
                    + "\n\nYour last answer was invalid. Reply with exactly one word: "
                    "federated, simulation, or centralized."
                )

        if result is None:
            raise RuntimeError(
                "Execution mode selection failed after LM retries."
            ) from last_error

        return result
