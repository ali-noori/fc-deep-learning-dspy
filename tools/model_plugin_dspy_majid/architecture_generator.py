"""Calls DSPy + LLM backends to produce cleaned Python source for class Model."""

from __future__ import annotations

from .backend_chain import build_backend_chain
from .code_sanitizer import extract_python_code
from .config import BACKEND_OLLAMA, Settings
from .dspy_support import (
    build_architecture_signature_class,
    build_fewshot_demos,
    import_dspy,
)
from .fewshot_examples import FEWSHOT_PAIRS
from .types import ArchitectureDescription


class ArchitectureGenerator:
    """Turn a natural-language prompt into an ArchitectureDescription."""

    def __init__(self, settings: Settings, ollama_model: str, ollama_base_url: str) -> None:
        self._settings = settings
        self._ollama_model = ollama_model
        self._ollama_base_url = ollama_base_url

    def generate(self, prompt: str) -> ArchitectureDescription:
        """Check if dspy is availabe"""
        dspy = import_dspy()

        try:
            # Build the signature class
            signature_cls = build_architecture_signature_class(dspy)
            
            # Build the fewshot demos
            demos = build_fewshot_demos(dspy, FEWSHOT_PAIRS)
            
            # Build the backend chain
            chain = build_backend_chain(
                dspy,
                self._settings,
                self._ollama_model,
                self._ollama_base_url,
            )
            
            '''Find LM set the model''' 
            # Configure the dspy
            # [0] is the first backend in the chain
            # [2] is the dspy.LM instance
            dspy.configure(lm=chain[0][2])
            
            # Build the predictor -> using chain of thought and the signature class
            predictor = dspy.ChainOfThought(signature_cls)
            
            # Add the fewshot demos to the predictor for few-shot learning
            predictor.demos = demos
        
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Failed to initialize DSPy Chain-of-Thought predictor.") from exc

        request = prompt.strip()
        last_error: BaseException | None = None

        for backend, model_label, lm in chain:
            try:
                # Genearte the code using the predictor
                result = predictor(prompt=request, lm=lm)
                
                # Extract the code from the result
                raw_code = result.architecture_code
                
                # Extract the code from the result
                code = extract_python_code(raw_code)
                
                # Set the base URL for the Ollama model to ge the used model URL
                base_url = self._ollama_base_url if backend == BACKEND_OLLAMA else None
                return ArchitectureDescription(
                    code=code,
                    backend=backend,
                    model=model_label,
                    base_url=base_url,
                )
            except BaseException as exc:
                last_error = exc
                continue

        raise RuntimeError(
            "All language-model backends failed (Groq, Gemini, Cohere, Mistral, Ollama). "
            f"Last error: {last_error!r}"
        ) from last_error
