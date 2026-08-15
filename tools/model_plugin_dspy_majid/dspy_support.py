"""Everything that requires the optional `dspy` package (import + DSPy types)."""

from __future__ import annotations

from typing import Any


def import_dspy():
    """Return the dspy module or raise a clear error."""
    try:
        import dspy
    except ImportError as exc:
        raise RuntimeError(
            "DSPy is not installed. Install the `dspy-ai` package in your environment "
            "to use tools.model_plugin_dspy_majid."
        ) from exc
    return dspy


def build_architecture_signature_class(dspy: Any):
    """Create the signiture class"""

    class ArchitectureSignature(dspy.Signature):

        # Input: Natural-language architecture
        prompt = dspy.InputField(
            desc=(
                "Natural-language model architecture description. May include image size, "
                "in_features, n_classes, ordered layer list, activations, flatten size, and "
                "softmax / no-softmax output rule."
            )
        )
        # Output: Python source for class Model.
        architecture_code = dspy.OutputField(
            desc=(
                "Python source code only. Define exactly one class Model(nn.Module) with "
                "__init__(self, n_classes, in_features) and forward(self, x). Follow the "
                "demo Prompt/Architecture pairs for naming, layer order, flatten style, "
                "dropout handling, and softmax behavior."
            )
        )

    return ArchitectureSignature


def build_fewshot_demos(dspy: Any, pairs: list[tuple[str, str]]) -> list:
    """Process raw tuples into DSPy examples"""
    try:
        fewshot_dataset = []
        for prompt_text, target_code in pairs:
            
            # Map the raw text to the DSPy Output and Input fields
            example_instance = dspy.Example(
                prompt=prompt_text, 
                architecture_code=target_code
            )
            
            # Explicitly designate the independent variable
            configured_example = example_instance.with_inputs("prompt")
            
            # Add the configured instance to the final dataset
            fewshot_dataset.append(configured_example)
            
        return fewshot_dataset
    
    except Exception as exc:
        
        raise RuntimeError("Failed to build DSPy few-shot demonstration examples.") from exc
