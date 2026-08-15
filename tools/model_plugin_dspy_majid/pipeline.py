"""Orchestrates generation and file write (high-level workflow)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .architecture_generator import ArchitectureGenerator
from .config import Settings, resolve_ollama_base_url, resolve_ollama_model
from .generated_model_writer import GeneratedModelWriter


class ModelPluginPipeline:
    """Same contract as model_plugin_dspy_ali: run(prompt) -> dict and file on disk."""

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
        self._generator: ArchitectureGenerator | None = None

    # Lazy initialization of the generator
    # with the settings
    def _get_generator(self) -> ArchitectureGenerator:
        if self._generator is None:
            self._generator = ArchitectureGenerator(
                self._settings,
                resolve_ollama_model(self._ollama_model_override),
                resolve_ollama_base_url(self._ollama_base_url_override),
            )
        return self._generator

    def run(self, prompt: str) -> dict[str, Any]:
        request = prompt.strip()
        if not request:
            raise ValueError("prompt must be non-empty")

        try:
            # Build architecture
            architecture_python_code = self._get_generator().generate(request)
            
            
            writer = GeneratedModelWriter(self._settings, self._repo_root)
            output_path = writer.write(
                architecture_python_code.code,
                architecture_python_code.backend,
                architecture_python_code.model,
                architecture_python_code.base_url,
            )
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Model plugin pipeline run failed.") from exc

        return {
            "approach": self._settings.pipeline_approach,
            "backend": architecture_python_code.backend,
            "model": architecture_python_code.model,
            "base_url": architecture_python_code.base_url,
            "output_model": str(output_path),
            "architecture": architecture_python_code.code,
        }
