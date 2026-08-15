"""DSPy teleprompt metric: score generated architecture code as valid or invalid."""

from __future__ import annotations

import ast
import re

from .code_sanitizer import extract_python_code


def architecture_metric(example, predicted, trace=None) -> float:
    try:
        raw = getattr(predicted, "architecture_code", None)
        if raw is None and isinstance(predicted, dict):
            raw = predicted.get("architecture_code")
        if not raw or not str(raw).strip():
            return 0.0

        code = extract_python_code(str(raw))

        # Step 1: keyword check
        if "import torch" not in code:
            return 0.0
        if "nn.Module" not in code:
            return 0.0
        if "def forward" not in code:
            return 0.0

        # Step 2: syntax check (valid Python)
        try:
            ast.parse(code)
        except SyntaxError:
            return 0.0

        # Step 3: semantic check (request type must match code layers)
        request = _get_request_text(example)
        if not _matches_architecture_semantics(request, code):
            return 0.0

        # Step 4: dynamic execution (instantiate model and run forward pass)
        if not _passes_execution_check(request, code):
            return 0.0

        return 1.0
    except Exception as exc:
        raise RuntimeError("architecture_metric evaluation failed.") from exc


def _get_request_text(example) -> str:
    if example is None:
        return ""
    if hasattr(example, "architecture_description"):
        return str(example.architecture_description)
    if isinstance(example, dict):
        return str(example.get("architecture_description", ""))
    return ""


def _is_cnn_request(request: str) -> bool:
    text = request.lower()
    cnn_words = (
        "cnn",
        "conv2d",
        "conv1d",
        "convolution",
        "convolutional",
        "image",
        "vision",
        "maxpool",
    )
    return any(word in text for word in cnn_words)


def _is_mlp_request(request: str) -> bool:
    text = request.lower()
    mlp_words = (
        "mlp",
        "fully connected",
        "fully-connected",
        "multilayer perceptron",
    )
    if any(word in text for word in mlp_words):
        return True
    if "linear:" in text and "conv" not in text:
        return True
    return False


def _is_sequence_request(request: str) -> bool:
    text = request.lower()
    sequence_words = ("lstm", "rnn", "gru", "sequence", "sequential", "transformer")
    return any(word in text for word in sequence_words)


def _code_has_conv_layers(code: str) -> bool:
    text = code.lower()
    conv_markers = ("conv2d", "conv1d", "conv3d", "nn.conv")
    return any(marker in text for marker in conv_markers)


def _code_has_linear_layers(code: str) -> bool:
    text = code.lower()
    return "nn.linear" in text or "linear(" in text


def _code_looks_sequential(code: str) -> bool:
    text = code.lower()
    markers = ("lstm", "rnn", "gru", "transformer")
    return any(marker in text for marker in markers)


def _matches_architecture_semantics(request: str, code: str) -> bool:
    """CNN/Image/Vision requests need conv layers; MLP requests need Linear, not conv."""
    if _is_cnn_request(request):
        return _code_has_conv_layers(code)

    if _is_mlp_request(request):
        if not _code_has_linear_layers(code):
            return False
        if _code_has_conv_layers(code):
            return False
        return True

    return True


def _guess_in_features(request: str) -> int:
    """Default feature/channel count for dummy tensors."""
    if _is_cnn_request(request):
        return 1
    return 10


def _guess_image_size(request: str) -> tuple[int, int]:
    """Read sizes like 28x28 from the request; default to 28x28."""
    match = re.search(r"(\d+)\s*x\s*(\d+)", request.lower())
    if match:
        return int(match.group(1)), int(match.group(2))
    return 28, 28


def _make_dummy_input(request: str, code: str, in_features: int):
    import torch

    batch_size = 2

    if _is_sequence_request(request) or _code_looks_sequential(code):
        seq_len = 10
        return torch.randn(batch_size, seq_len, in_features)

    if _is_cnn_request(request) or _code_has_conv_layers(code):
        height, width = _guess_image_size(request)
        return torch.randn(batch_size, in_features, height, width)

    return torch.randn(batch_size, in_features)


def _find_model_class(namespace: dict, nn_module) -> type | None:
    model_cls = namespace.get("Model")
    if isinstance(model_cls, type) and issubclass(model_cls, nn_module.Module):
        return model_cls

    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, nn_module.Module):
            continue
        if obj is nn_module.Module:
            continue
        return obj

    return None


def _passes_execution_check(request: str, code: str) -> bool:
    """
    Load code, build the Model, run one forward pass with dummy data.
    Any error returns False so the DSPy pipeline keeps running.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        namespace = {
            "torch": torch,
            "nn": nn,
            "F": F,
        }
        exec(code, namespace)

        model_cls = _find_model_class(namespace, nn)
        if model_cls is None:
            return False

        n_classes = 10
        in_features = _guess_in_features(request)
        model = model_cls(n_classes=n_classes, in_features=in_features)
        model.eval()

        dummy_input = _make_dummy_input(request, code, in_features)

        with torch.no_grad():
            output = model(dummy_input)

        return output is not None
    except Exception:
        return False
