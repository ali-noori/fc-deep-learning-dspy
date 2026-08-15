"""DSPy v2 pipeline: bootstrap few-shot demos, generate architecture, write plugin file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings, resolve_ollama_base_url, resolve_ollama_model
from .generated_model_writer import GeneratedModelWriter
from .optimizer import compile_module


class ModelPluginV2Pipeline:
    def __init__(
        self,
        repo_root: Path,
        settings: Settings,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._settings = settings
        self._ollama_model_override = ollama_model
        self._ollama_base_url_override = ollama_base_url

    def _ollama_model(self) -> str:
        return resolve_ollama_model(self._ollama_model_override)

    def _ollama_base_url(self) -> str:
        return resolve_ollama_base_url(self._ollama_base_url_override)

    def run(self, prompt: str) -> dict[str, Any]:
        request = prompt.strip()
        if not request:
            raise ValueError("prompt must be non-empty")

        try:
            '''
            After running these few lines of code, 
            you have a “fully prepared and optimized brain.” 
            This brain now knows what examples to use, 
            how to evaluate its code, and what support model to fall back 
            to if the internet goes down. 
            Now the module is just waiting for you to give it a text sentence 
            to deliver the Python code to you!
            '''
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
            )
            # Ask the moodule to generate code for the given prompt
            result = module(architecture_description=request)

            writer = GeneratedModelWriter(self._settings, self._repo_root)
            output_path = writer.write(
                result.code,
                result.backend,
                result.model,
                result.base_url,
            )
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Model plugin v2 pipeline run failed.") from exc

        return {
            "approach": self._settings.pipeline_approach,
            "backend": result.backend,
            "model": result.model,
            "base_url": result.base_url,
            "output_model": str(output_path),
            "architecture": result.code,
        }
