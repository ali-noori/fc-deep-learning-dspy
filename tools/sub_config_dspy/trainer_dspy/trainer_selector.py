"""DSPy Signature + Module: natural language -> trainer config fields (LM only)."""

from __future__ import annotations

from .config import (
    BUILTIN_DATALOADERS,
    BUILTIN_LOSSES,
    BUILTIN_TRAINERS,
    Settings,
)
from .dspy_support import dspy
from .types import TrainerExtractionResult


def format_known_trainers(trainer_artifacts: dict[str, str]) -> str:
    """Format built-in and discovered trainer plugins for the LM."""
    builtins = ", ".join(BUILTIN_TRAINERS)
    builtin_hint = f"built-in names without .py ({builtins})"
    if not trainer_artifacts:
        return builtin_hint
    plugins = ", ".join(
        f"{name} -> {path}" for name, path in trainer_artifacts.items()
    )
    return f"{builtin_hint}; plugins: {plugins}"


def format_known_dataloaders(dataloader_artifacts: dict[str, str]) -> str:
    """Format built-in and discovered dataloader plugins for the LM."""
    builtins = ", ".join(BUILTIN_DATALOADERS)
    builtin_hint = f"built-in names without .py ({builtins})"
    if not dataloader_artifacts:
        return builtin_hint
    plugins = ", ".join(
        f"{name} -> {path}" for name, path in dataloader_artifacts.items()
    )
    return f"{builtin_hint}; plugins: {plugins}"


def format_known_losses(loss_artifacts: dict[str, str]) -> str:
    """Format built-in and discovered loss plugins for the LM."""
    builtins = ", ".join(BUILTIN_LOSSES)
    builtin_hint = f"built-in torch.nn class names without .py ({builtins})"
    if not loss_artifacts:
        return builtin_hint
    plugins = ", ".join(
        f"{name} -> {path}" for name, path in loss_artifacts.items()
    )
    return f"{builtin_hint}; plugins: {plugins}"


class ParseTrainerConfigSignature(dspy.Signature):
    """Extract trainer fields from the developer's answer.

    Outputs:
    - name: built-in trainer OR trainer plugin path
    - data_loader: built-in dataloader OR dataloader plugin path
    - loss_name: built-in torch.nn loss OR loss plugin path
    - num_classes: positive integer (n_class / n_classes synonym)

    Decision rules:
    - Ignore typos, informal wording, and incomplete phrasing; infer the intended values.
    - If the user mentions unrelated extras (dataset paths, model architecture, FedAvg), ignore them.
    - Do not invent missing required fields; only extract what is stated or clearly implied.
    - Synonyms: trainer/name → name; loader/data_loader → data_loader;
      loss / loss.name / loss_name → loss_name; classes / n_class → num_classes.

    Built-in vs plugin (critical — apply independently to each of the three components):
    - Trainer:
      * BasicTrainer / "built-in trainer" / "not the .py file" → BasicTrainer
      * FedMMb.py / "custom trainer plugin" / plugins/trainers/FedMMb.py → plugin trainer
      * Bare BasicTrainer without .py/plugin/custom wording defaults to built-in.
    - Data loader:
      * ImageLoader / "built-in loader" → ImageLoader
      * ImageLoader.py / RnaLoader.py / plugins/dataloaders/... → plugin dataloader
      * Do not convert ImageLoader into ImageLoader.py unless the user asks for the plugin.
    - Loss:
      * CrossEntropyLoss / NLLLoss / MSELoss / BCEWithLogitsLoss → built-in torch.nn names
      * focal_loss.py / plugins/loss/focal_loss.py → plugin loss
      * Do not convert CrossEntropyLoss into a .py plugin unless requested.

    Mixed configs are allowed (e.g. built-in trainer + plugin loader + built-in loss).
    Choose each field only from the corresponding known_* built-ins/plugins list.
    num_classes must be a positive integer.
    """

    user_message: str = dspy.InputField(
        desc=(
            "Developer natural-language answer about trainer/data_loader/loss; "
            "may be short, long, messy, incomplete, or misspelled, and may not "
            "use exact field names; components may be mixed built-in/plugin"
        )
    )
    known_trainers: str = dspy.InputField(
        desc=(
            "Allowed trainers: built-in names without .py (e.g. BasicTrainer) and "
            "discovered plugins as filename -> path (e.g. FedMMb.py -> plugins/trainers/FedMMb.py)"
        )
    )
    known_dataloaders: str = dspy.InputField(
        desc=(
            "Allowed dataloaders: built-in names without .py (e.g. ImageLoader) and "
            "discovered plugins as filename -> path "
            "(e.g. ImageLoader.py -> plugins/dataloaders/ImageLoader.py)"
        )
    )
    known_losses: str = dspy.InputField(
        desc=(
            "Allowed losses: built-in torch.nn class names without .py "
            "(e.g. CrossEntropyLoss) and discovered plugins as filename -> path "
            "(e.g. focal_loss.py -> plugins/loss/focal_loss.py)"
        )
    )
    name: str = dspy.OutputField(
        desc=(
            "Built-in trainer (e.g. BasicTrainer) OR plugin filename/path "
            "(e.g. FedMMb.py or plugins/trainers/FedMMb.py). "
            "Do not confuse BasicTrainer with FedMMb.py."
        )
    )
    data_loader: str = dspy.OutputField(
        desc=(
            "Built-in dataloader (e.g. ImageLoader) OR plugin filename/path "
            "(e.g. ImageLoader.py, RnaLoader.py, or plugins/dataloaders/...). "
            "Do not confuse ImageLoader with ImageLoader.py."
        )
    )
    loss_name: str = dspy.OutputField(
        desc=(
            "Built-in torch.nn loss (e.g. CrossEntropyLoss) OR plugin filename/path "
            "(e.g. focal_loss.py or plugins/loss/focal_loss.py). "
            "Do not confuse CrossEntropyLoss with focal_loss.py."
        )
    )
    num_classes: str = dspy.OutputField(
        desc="Positive integer: number of classes (e.g. 10). Accept n_class as synonym."
    )


def _looks_like_plugin_reference(text: str) -> bool:
    return text.endswith(".py") or "/" in text or "\\" in text


def _resolve_artifact_name(
    raw: str,
    artifacts: dict[str, str],
    builtins: tuple[str, ...],
    field_label: str,
    plugins_dir_hint: str,
) -> str:
    text = str(raw or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError(f"{field_label} must not be empty.")

    # Plugin filename (dict key), e.g. FedMMb.py -> plugins/trainers/FedMMb.py
    if text in artifacts:
        return artifacts[text]

    # Already a full plugin path (dict value)
    if text in artifacts.values():
        return text

    if _looks_like_plugin_reference(text):
        if artifacts:
            allowed = ", ".join(artifacts.keys())
            raise ValueError(
                f"{field_label} {text!r} is not a discovered plugin. "
                f"Allowed plugins: {allowed}"
            )
        raise ValueError(
            f"{field_label} {text!r} looks like a plugin but none were "
            f"discovered under {plugins_dir_hint}."
        )

    if text in builtins:
        return text

    allowed_builtins = ", ".join(builtins)
    allowed_plugins = ", ".join(artifacts.keys()) or "(none)"
    raise ValueError(
        f"{field_label} {text!r} is not valid. "
        f"Allowed built-ins: {allowed_builtins}. "
        f"Allowed plugins: {allowed_plugins}"
    )


def _resolve_trainer_name(
    name_raw: str,
    trainer_artifacts: dict[str, str],
) -> str:
    return _resolve_artifact_name(
        name_raw,
        trainer_artifacts,
        BUILTIN_TRAINERS,
        "trainer name",
        "plugins/trainers",
    )


def _resolve_dataloader_name(
    data_loader_raw: str,
    dataloader_artifacts: dict[str, str],
) -> str:
    return _resolve_artifact_name(
        data_loader_raw,
        dataloader_artifacts,
        BUILTIN_DATALOADERS,
        "data_loader",
        "plugins/dataloaders",
    )


def _resolve_loss_name(
    loss_name_raw: str,
    loss_artifacts: dict[str, str],
) -> str:
    return _resolve_artifact_name(
        loss_name_raw,
        loss_artifacts,
        BUILTIN_LOSSES,
        "loss.name",
        "plugins/loss",
    )


def _validate_extraction(
    name_raw: str,
    data_loader_raw: str,
    loss_name_raw: str,
    num_classes_raw: str,
    trainer_artifacts: dict[str, str] | None = None,
    dataloader_artifacts: dict[str, str] | None = None,
    loss_artifacts: dict[str, str] | None = None,
) -> tuple[str, str, str, int]:
    classes_text = str(num_classes_raw or "").strip()

    if (
        not str(name_raw or "").strip()
        or not str(data_loader_raw or "").strip()
        or not str(loss_name_raw or "").strip()
        or not classes_text
    ):
        raise ValueError("Missing name, data_loader, loss.name, or n_class.")

    try:
        num_classes = int(float(classes_text))
    except ValueError as exc:
        raise ValueError("n_class must be a positive integer.") from exc

    if num_classes < 1:
        raise ValueError("n_class must be a positive integer.")

    name = _resolve_trainer_name(name_raw, trainer_artifacts or {})
    data_loader = _resolve_dataloader_name(data_loader_raw, dataloader_artifacts or {})
    loss_name = _resolve_loss_name(loss_name_raw, loss_artifacts or {})
    return name, data_loader, loss_name, num_classes


class TrainerConfigModule(dspy.Module):
    """Chain-of-Thought extractor for trainer config fields."""

    def __init__(
        self,
        settings: Settings,
        trainer_artifacts: dict[str, str] | None = None,
        dataloader_artifacts: dict[str, str] | None = None,
        loss_artifacts: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._trainer_artifacts = dict(trainer_artifacts or {})
        self._dataloader_artifacts = dict(dataloader_artifacts or {})
        self._loss_artifacts = dict(loss_artifacts or {})
        self.extract = dspy.ChainOfThought(ParseTrainerConfigSignature)

    def forward(
        self,
        user_message: str,
        known_trainers: str = "",
        known_dataloaders: str = "",
        known_losses: str = "",
    ) -> TrainerExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")

        known_trainer_hint = known_trainers.strip() or format_known_trainers(
            self._trainer_artifacts
        )
        known_dataloader_hint = known_dataloaders.strip() or format_known_dataloaders(
            self._dataloader_artifacts
        )
        known_loss_hint = known_losses.strip() or format_known_losses(
            self._loss_artifacts
        )

        result: TrainerExtractionResult | None = None
        last_error: Exception | None = None

        for _ in range(self._settings.trainer_extract_max_retries):
            try:
                prediction = self.extract(
                    user_message=current_request,
                    known_trainers=known_trainer_hint,
                    known_dataloaders=known_dataloader_hint,
                    known_losses=known_loss_hint,
                )
                name, data_loader, loss_name, num_classes = _validate_extraction(
                    str(getattr(prediction, "name", "") or ""),
                    str(getattr(prediction, "data_loader", "") or ""),
                    str(getattr(prediction, "loss_name", "") or ""),
                    str(getattr(prediction, "num_classes", "") or ""),
                    self._trainer_artifacts,
                    self._dataloader_artifacts,
                    self._loss_artifacts,
                )

                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                result = TrainerExtractionResult(
                    name=name,
                    data_loader=data_loader,
                    loss_name=loss_name,
                    num_classes=num_classes,
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
                    + "\nReply with name, data_loader, loss.name, and n_class "
                    + "copied exactly from the developer answer. "
                    + f"Allowed trainers: {known_trainer_hint}. "
                    + f"Allowed dataloaders: {known_dataloader_hint}. "
                    + f"Allowed losses: {known_loss_hint}"
                )

        if result is None:
            raise RuntimeError(
                "Trainer config extraction failed after LM retries."
            ) from last_error

        return result
