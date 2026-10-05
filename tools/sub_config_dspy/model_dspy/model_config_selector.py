"""DSPy Signature + Module: natural language → model config fields (LM only)."""

from __future__ import annotations

from .config import BUILTIN_MODELS, Settings
from .dspy_support import dspy
from .types import ModelExtractionResult


def format_known_models(model_artifacts: dict[str, str]) -> str:
    """Format built-in and discovered model plugins for the LM."""
    builtins = ", ".join(BUILTIN_MODELS)
    builtin_hint = f"built-in names without .py ({builtins})"
    if not model_artifacts:
        return builtin_hint
    plugins = ", ".join(
        f"{name} -> {path}" for name, path in model_artifacts.items()
    )
    return f"{builtin_hint}; plugins: {plugins}"


class ParseModelConfigSignature(dspy.Signature):
    """Extract model architecture fields from the developer's answer.

    Outputs:
    - name: built-in model name OR custom plugin reference (resolved path in gold/config)
    - n_classes: positive integer output class count (n_class is a synonym)
    - in_features: positive integer input size / channel count

    Decision rules:
    - Ignore typos, informal wording, and incomplete phrasing; infer the intended values.
    - If the user mentions unrelated extras (dataset paths, trainer, loss, FedAvg), ignore them.
    - Do not invent missing required fields; only extract what is stated or clearly implied.
    - Synonyms: classes / n_class / n_classes → n_classes;
      input features / channels / input size → in_features.
    - Built-in vs plugin (critical):
      * CNN / "built-in" / "library" / "not the .py file" → CNN
      * cnn.py / "custom plugin cnn" / plugins/models/cnn.py → plugin cnn
      * mlp.py / plugins/models/mlp.py → plugin mlp
      * Bare "CNN" without .py/plugin/custom wording defaults to built-in CNN.
      * Do not convert built-in CNN into cnn.py unless the user asks for the plugin.
    - Choose name only from known built-ins or discovered plugins in known_models.
    - Plugins are not limited to cnn.py/mlp.py: any filename listed in
      known_models is a valid plugin name, including previously generated
      and saved custom models (e.g. custom_model.py). Echo the plugin name
      exactly as it appears in known_models; do not force it to cnn.py/mlp.py.
    - Pretrained/architecture-only plugins (matched from known_models the same
      way as any other plugin filename; never invent one that is not listed):
      * resnet / resnet18 / "ResNet-18" -> resnet18.py
        (pretrained CNN backbone, image data, ready to fine-tune)
      * efficientnet / "efficientnet-b0" / "EfficientNet B0" -> efficientnet_b0.py
        (pretrained CNN backbone, image data, ready to fine-tune)
      * mobilenet / mobilenetv3 / "MobileNetV3-Small" / "mobile net small" -> mobilenetv3_small.py
        (pretrained CNN backbone, image data, ready to fine-tune)
      * tabnet / "tab net" -> tabnet.py
        (tabular; attention-based, sparse feature selection; trained from scratch)
      * "ft-transformer" / "ft transformer" / "feature tokenizer transformer" / fttransformer -> ft_transformer.py
        (tabular; transformer-based; strong general tabular performance; trained from scratch)
      * tabtransformer / "tab transformer" -> tabtransformer.py
        (tabular; transformer-based; especially good with categorical features; trained from scratch)
      * These aliases only apply when the matching filename is present in
        known_models; if it is absent, do not output it anyway.
    - n_classes and in_features must be positive integers.
    """

    user_message: str = dspy.InputField(
        desc=(
            "Developer natural-language answer about the model block; "
            "may be short, long, messy, incomplete, or misspelled, and may not "
            "use exact field names"
        )
    )
    known_models: str = dspy.InputField(
        desc=(
            "Allowed models: built-in names without .py (e.g. CNN) and "
            "discovered plugins as filename -> repo-relative path "
            "(e.g. cnn.py -> plugins/models/cnn.py, mlp.py -> plugins/models/mlp.py)"
        )
    )
    name: str = dspy.OutputField(
        desc=(
            "Built-in model name (e.g. CNN) OR plugin filename/path "
            "(e.g. cnn.py, mlp.py, resnet18.py, tabnet.py, or a full "
            "plugins/models/... path). Do not confuse built-in CNN with "
            "plugin cnn.py, and never invent a plugin name not present in "
            "known_models."
        )
    )
    n_classes: str = dspy.OutputField(
        desc="Positive integer: number of output classes (e.g. 10). Accept n_class as synonym."
    )
    in_features: str = dspy.OutputField(
        desc=(
            "Positive integer: input size / channel count "
            "(e.g. 1 for MNIST grayscale, 784 for flattened MLP input)."
        )
    )


def _looks_like_plugin_reference(model_text: str) -> bool:
    return model_text.endswith(".py") or "/" in model_text or "\\" in model_text


def _resolve_model_name(
    name_raw: str,
    model_artifacts: dict[str, str],
) -> str:
    name_text = str(name_raw or "").strip().strip('"').strip("'")
    if not name_text:
        raise ValueError("model name must not be empty.")

    # Plugin filename (dict key), e.g. cnn.py -> plugins/models/cnn.py
    if name_text in model_artifacts:
        return model_artifacts[name_text]

    # Already a full plugin path (dict value)
    if name_text in model_artifacts.values():
        return name_text

    if _looks_like_plugin_reference(name_text):
        if model_artifacts:
            allowed = ", ".join(model_artifacts.keys())
            raise ValueError(
                f"model name {name_text!r} is not a discovered plugin. "
                f"Allowed plugins: {allowed}"
            )
        raise ValueError(
            f"model name {name_text!r} looks like a plugin but none were "
            "discovered under plugins/models."
        )

    if name_text in BUILTIN_MODELS:
        return name_text

    allowed_builtins = ", ".join(BUILTIN_MODELS)
    allowed_plugins = ", ".join(model_artifacts.keys()) or "(none)"
    raise ValueError(
        f"model name {name_text!r} is not a valid model. "
        f"Allowed built-ins: {allowed_builtins}. Allowed plugins: {allowed_plugins}"
    )


def _validate_extraction(
    name_raw: str,
    n_classes_raw: str,
    in_features_raw: str,
    model_artifacts: dict[str, str] | None = None,
) -> tuple[str, int, int]:
    n_classes_text = str(n_classes_raw or "").strip()
    in_features_text = str(in_features_raw or "").strip()

    if not str(name_raw or "").strip() or not n_classes_text or not in_features_text:
        raise ValueError("Missing name, n_classes (or n_class), or in_features.")

    try:
        n_classes = int(float(n_classes_text))
        in_features = int(float(in_features_text))
    except ValueError as exc:
        raise ValueError("n_classes and in_features must be positive integers.") from exc

    if n_classes < 1 or in_features < 1:
        raise ValueError("n_classes and in_features must be positive integers.")

    name = _resolve_model_name(name_raw, model_artifacts or {})
    return name, n_classes, in_features


class ModelConfigModule(dspy.Module):
    """Chain-of-Thought extractor for model config fields."""

    def __init__(
        self,
        settings: Settings,
        model_artifacts: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._model_artifacts = dict(model_artifacts or {})
        self.extract = dspy.ChainOfThought(ParseModelConfigSignature)

    def forward(
        self,
        user_message: str,
        known_models: str = "",
    ) -> ModelExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")

        known = known_models.strip() or format_known_models(self._model_artifacts)

        result: ModelExtractionResult | None = None
        last_error: Exception | None = None

        for _ in range(self._settings.model_extract_max_retries):
            try:
                prediction = self.extract(
                    user_message=current_request,
                    known_models=known,
                )
                name, n_classes, in_features = _validate_extraction(
                    str(getattr(prediction, "name", "") or ""),
                    str(getattr(prediction, "n_classes", "") or ""),
                    str(getattr(prediction, "in_features", "") or ""),
                    self._model_artifacts,
                )

                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                result = ModelExtractionResult(
                    name=name,
                    n_classes=n_classes,
                    in_features=in_features,
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
                    + "\nReply with name, n_class (or n_classes), and in_features "
                    + "copied exactly from the developer answer. "
                    + f"Allowed models: {known}"
                )

        if result is None:
            raise RuntimeError(
                "Model config extraction failed after LM retries."
            ) from last_error

        return result
