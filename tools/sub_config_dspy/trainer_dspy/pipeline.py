"""DSPy trainer config pipeline: bootstrap few-shot, extract fields, write YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    AGENT_OKAY_REPLY,
    TRAINER_QUESTION,
    Settings,
    resolve_ollama_base_url,
    resolve_ollama_model,
)
from .optimizer import compile_module
from .output_writer import TrainerOutputWriter
from .types import TrainerConfig, TrainerConfigResult

_PACKAGE_DIR = Path(__file__).resolve().parent


class TrainerPipeline:
    TRAINER_QUESTION = TRAINER_QUESTION

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

    def extract_trainer_config(
        self,
        user_message: str,
        trainer_artifacts: dict[str, str] | None = None,
        dataloader_artifacts: dict[str, str] | None = None,
        loss_artifacts: dict[str, str] | None = None,
    ) -> TrainerConfigResult:
        request = user_message.strip()
        if not request:
            raise ValueError("Please provide name, data_loader, loss.name, and n_class.")

        try:
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
                trainer_artifacts,
                dataloader_artifacts,
                loss_artifacts,
                self._recompile,
            )
            result = module(user_message=request)

            config = TrainerConfig(
                name=result.name,
                data_loader=result.data_loader,
                loss_name=result.loss_name,
                num_classes=result.num_classes,
                local_updates=self._settings.default_local_updates,
                optimizer_name=self._settings.default_optimizer_name,
                optimizer_lr=self._settings.default_optimizer_lr,
            )
            writer = TrainerOutputWriter(self._settings, _PACKAGE_DIR)
            output_path = writer.write(config)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Trainer config pipeline extraction failed.") from exc

        return TrainerConfigResult(
            name=result.name,
            data_loader=result.data_loader,
            loss_name=result.loss_name,
            num_classes=result.num_classes,
            local_updates=config.local_updates,
            optimizer_name=config.optimizer_name,
            optimizer_lr=config.optimizer_lr,
            agent_reply=AGENT_OKAY_REPLY,
            user_message=request,
            backend=result.backend,
            model=result.model,
            base_url=result.base_url,
            output_path=str(output_path),
        )

    def run(
        self,
        user_message: str,
        trainer_artifacts: dict[str, str] | None = None,
        dataloader_artifacts: dict[str, str] | None = None,
        loss_artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        outcome = self.extract_trainer_config(
            user_message,
            trainer_artifacts,
            dataloader_artifacts,
            loss_artifacts,
        )
        return {
            "approach": self._settings.pipeline_approach,
            "name": outcome.name,
            "data_loader": outcome.data_loader,
            "loss_name": outcome.loss_name,
            "num_classes": outcome.num_classes,
            "local_updates": outcome.local_updates,
            "optimizer_name": outcome.optimizer_name,
            "optimizer_lr": outcome.optimizer_lr,
            "agent_reply": outcome.agent_reply,
            "backend": outcome.backend,
            "model": outcome.model,
            "base_url": outcome.base_url,
            "user_message": outcome.user_message,
            "output_path": outcome.output_path,
        }
