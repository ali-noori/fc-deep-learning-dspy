"""DSPy Signature + Module: natural language → PyTorch Model(nn.Module) source."""

from __future__ import annotations

import re

from .code_sanitizer import extract_python_code
from .dspy_support import dspy
from .types import ArchitectureResult


def _validate_forward_pass(request: str, code: str) -> None:
    """
    Run generated code and one forward pass; raise on failure.
    IMPORTANT: the main purpose if that we just check the genearted architecture and 
    not the exaxt promtp that we have provided. So we might have different in_features and n_classes.
    for example mybe in the prompt it is for 512 layer as input_feature and n_classes is 100.
    but we test it with deefault values for example for cnn it is 1 and for the n_classes it is 10.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(code, namespace)

    model_cls = namespace.get("Model")
    if not isinstance(model_cls, type) or not issubclass(model_cls, nn.Module):
        for obj in namespace.values():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                model_cls = obj
                break
        else:
            raise RuntimeError("No nn.Module subclass found in generated code.")

    text = request.lower()
    is_cnn = any(w in text for w in ("cnn", "conv2d", "image", "vision", "convolution"))
    in_features = 1 if is_cnn else 10
    batch = 2

    if is_cnn or "conv2d" in code.lower():
        match = re.search(r"(\d+)\s*x\s*(\d+)", text)
        h, w = (int(match.group(1)), int(match.group(2))) if match else (28, 28)
        dummy = torch.randn(batch, in_features, h, w)
    else:
        dummy = torch.randn(batch, in_features)

    model = model_cls(n_classes=10, in_features=in_features)
    model.eval()
    with torch.no_grad():
        output = model(dummy)
    if output is None:
        raise RuntimeError("forward() returned None.")


class GenerateArchitectureSignature(dspy.Signature):
    """Generate a FeatureCloud-compatible PyTorch nn.Module plugin from a natural-language spec."""

    architecture_description: str = dspy.InputField(
        desc=(
            "Natural-language architecture description including layer order, "
            "in_features, n_classes, activations, pooling, flatten size, and output rule."
        )
    )
    architecture_code: str = dspy.OutputField(
        desc=(
            "Valid Python source for one class Model(nn.Module) with "
            "__init__(self, n_classes, in_features) and forward(self, x). "
            "Include required torch imports. Match few-shot demo style."
        )
    )


class ArchitectureGeneratorModule(dspy.Module):
    """Chain-of-Thought generator for FeatureCloud model plugins."""

    def __init__(self) -> None:
        super().__init__()
        # Strategy of learnign the the example
        self.generate = dspy.ChainOfThought(GenerateArchitectureSignature)

    def forward(self, architecture_description: str) -> ArchitectureResult:
        max_retries = 3
        current_request = architecture_description.strip()
        if not current_request:
            raise ValueError("architecture_description must be non-empty")

        try:
            result = None
            for attempt in range(max_retries):
                prediction = self.generate(architecture_description=current_request)
                code = extract_python_code(prediction.architecture_code)

                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                result = ArchitectureResult(
                    code=code,
                    backend=backend,
                    model=model,
                    base_url=base_url,
                )

                try:
                    _validate_forward_pass(current_request, code)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        break
                    current_request = (
                        current_request
                        + "\n\nPrevious generation failed validation. Fix the code. Error:\n"
                        + str(e)
                    )

            return result
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError("ArchitectureGeneratorModule.forward failed.") from exc
