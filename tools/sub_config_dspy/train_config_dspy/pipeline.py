"""DSPy train_config pipeline: bootstrap few-shot, extract fields, write YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    AGENT_OKAY_REPLY,
    TRAIN_CONFIG_QUESTION,
    Settings,
    resolve_ollama_base_url,
    resolve_ollama_model,
)
from .optimizer import compile_module
from .output_writer import TrainConfigOutputWriter
from .types import TrainConfig, TrainConfigResult

_PACKAGE_DIR = Path(__file__).resolve().parent


class TrainConfigPipeline:
    TRAIN_CONFIG_QUESTION = TRAIN_CONFIG_QUESTION

    def __init__(
        self,
        settings: Settings,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        recompile: bool | None = None,
    ) -> None:
        self._settings = settings
        self._ollama_model_override = ollama_model
        self._ollama_base_url_override = ollama_base_url
        self._recompile = recompile

    def _ollama_model(self) -> str:
        return resolve_ollama_model(self._ollama_model_override)

    def _ollama_base_url(self) -> str:
        return resolve_ollama_base_url(self._ollama_base_url_override)

    def extract_train_config(self, user_message: str) -> TrainConfigResult:
        request = user_message.strip()
        if not request:
            raise ValueError(
                "Please provide verbose, batch_size, test_batch_size, epochs, lr, "
                "and batch_count."
            )

        try:
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
                self._recompile,
            )
            result = module(user_message=request)

            config = TrainConfig(
                verbose=result.verbose,
                batch_size=result.batch_size,
                test_batch_size=result.test_batch_size,
                epochs=result.epochs,
                lr=result.lr,
                batch_count=result.batch_count,
                device=self._settings.default_device,
            )
            writer = TrainConfigOutputWriter(self._settings, _PACKAGE_DIR)
            output_path = writer.write(config)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Train config pipeline extraction failed."
            ) from exc

        return TrainConfigResult(
            verbose=result.verbose,
            batch_size=result.batch_size,
            test_batch_size=result.test_batch_size,
            epochs=result.epochs,
            lr=result.lr,
            batch_count=result.batch_count,
            device=config.device,
            agent_reply=AGENT_OKAY_REPLY,
            user_message=request,
            backend=result.backend,
            model=result.model,
            base_url=result.base_url,
            output_path=str(output_path),
        )

    def run(self, user_message: str) -> dict[str, Any]:
        outcome = self.extract_train_config(user_message)
        return {
            "approach": self._settings.pipeline_approach,
            "verbose": outcome.verbose,
            "batch_size": outcome.batch_size,
            "test_batch_size": outcome.test_batch_size,
            "epochs": outcome.epochs,
            "lr": outcome.lr,
            "batch_count": outcome.batch_count,
            "device": outcome.device,
            "agent_reply": outcome.agent_reply,
            "backend": outcome.backend,
            "model": outcome.model,
            "base_url": outcome.base_url,
            "user_message": outcome.user_message,
            "output_path": outcome.output_path,
        }
