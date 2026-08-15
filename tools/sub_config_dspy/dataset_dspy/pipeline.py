"""DSPy dataset config pipeline: bootstrap few-shot, extract fields, write YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AGENT_OKAY_REPLY, DATASET_QUESTION_FEDERATED, DATASET_QUESTION_CENTRALIZED, DATASET_QUESTION_SIMULATION, Settings, resolve_ollama_base_url, resolve_ollama_model
from .optimizer import compile_module
from .output_writer import DatasetOutputWriter
from .types import DatasetFederatedConfig, DatasetCentralizedConfig, DatasetSimulationConfig
from .types import DatasetFederatedConfigResult, DatasetCentralizedConfigResult, DatasetSimulationConfigResult


_PACKAGE_DIR = Path(__file__).resolve().parent


class DatasetPipeline:
    DATASET_QUESTION_FEDERATED = DATASET_QUESTION_FEDERATED
    DATASET_QUESTION_CENTRALIZED = DATASET_QUESTION_CENTRALIZED
    DATASET_QUESTION_SIMULATION = DATASET_QUESTION_SIMULATION

    def __init__(
        self,
        settings: Settings,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self._settings = settings
        self._ollama_model_override = ollama_model
        self._ollama_base_url_override = ollama_base_url

    def _ollama_model(self) -> str:
        return resolve_ollama_model(self._ollama_model_override)

    def _ollama_base_url(self) -> str:
        return resolve_ollama_base_url(self._ollama_base_url_override)

    def extract_dataset_config(self, user_message: str, execution_mode: str) -> DatasetFederatedConfigResult | DatasetCentralizedConfigResult:
        request = user_message.strip()
        if not request:
            raise ValueError(
                "Please provide data_dir, train_dataset_file_name, "
                "test_dataset_file_name, and logic_dir."
            )

        try:
            module = compile_module(
                self._settings,
                self._ollama_model(),
                self._ollama_base_url(),
                execution_mode,
            )
            result = module(user_message=request)

            # Create buidong block necessary deatils 
            # to complete dataset part of the federated mode config file
            if execution_mode == "federated":
                datasetFederatedConfigObj = DatasetFederatedConfig(
                    data_dirs=result.data_dirs,
                    train_dataset_file_name=result.train_dataset_file_name,
                    test_dataset_file_name=result.test_dataset_file_name,
                    logic_dir=result.logic_dir,
                )
                writer = DatasetOutputWriter(self._settings, _PACKAGE_DIR)
                output_path = writer.write_federated(datasetFederatedConfigObj, execution_mode)
            elif execution_mode == "centralized":
                datasetCentralizedConfigObj = DatasetCentralizedConfig(
                    data_dir=result.data_dir,
                    train_dataset_file_name=result.train_dataset_file_name,
                    test_dataset_file_name=result.test_dataset_file_name,
                    logic_dir=result.logic_dir,
                )
                writer = DatasetOutputWriter(self._settings, _PACKAGE_DIR)
                output_path = writer.write_centralized(datasetCentralizedConfigObj, execution_mode)
            elif execution_mode == "simulation":
                datasetSimulationConfigObj = DatasetSimulationConfig(
                    data_dir=result.data_dir,
                    client_dirs=result.client_dirs,
                    train_dataset_file_name=result.train_dataset_file_name,
                    test_dataset_file_name=result.test_dataset_file_name,
                    logic_dir=result.logic_dir,
                )
                writer = DatasetOutputWriter(self._settings, _PACKAGE_DIR)
                output_path = writer.write_simulation(datasetSimulationConfigObj, execution_mode)
                
                
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Dataset config pipeline extraction failed.") from exc

        if execution_mode == "federated":
            return DatasetFederatedConfigResult(
                data_dirs=result.data_dirs,
                train_dataset_file_name=result.train_dataset_file_name,
                test_dataset_file_name=result.test_dataset_file_name,
                logic_dir=result.logic_dir,
                agent_reply=AGENT_OKAY_REPLY,
                user_message=request,
                backend=result.backend,
                model=result.model,
                base_url=result.base_url,
                output_path=str(output_path),
            )
        elif execution_mode == "centralized":
            return DatasetCentralizedConfigResult(
                data_dir=result.data_dir,
                train_dataset_file_name=result.train_dataset_file_name,
                test_dataset_file_name=result.test_dataset_file_name,
                logic_dir=result.logic_dir,
                agent_reply=AGENT_OKAY_REPLY,
                user_message=request,
                backend=result.backend,
                model=result.model,
                base_url=result.base_url,
                output_path=str(output_path),
            )
        elif execution_mode == "simulation":
            return DatasetSimulationConfigResult(
                data_dir=result.data_dir,
                client_dirs=result.client_dirs,
                train_dataset_file_name=result.train_dataset_file_name,
                test_dataset_file_name=result.test_dataset_file_name,
                logic_dir=result.logic_dir,
                agent_reply=AGENT_OKAY_REPLY,
                user_message=request,
                backend=result.backend,
                model=result.model,
                base_url=result.base_url,
                output_path=str(output_path),
            )

    def run(self, user_message: str, execution_mode: str) -> dict[str, Any]:
        outcome = self.extract_dataset_config(user_message, execution_mode)
        
        result = {
            "approach": self._settings.pipeline_approach,
            "train_dataset_file_name": outcome.train_dataset_file_name,
            "test_dataset_file_name": outcome.test_dataset_file_name,
            "logic_dir": outcome.logic_dir,
            "agent_reply": outcome.agent_reply,
            "backend": outcome.backend,
            "model": outcome.model,
            "base_url": outcome.base_url,
            "user_message": outcome.user_message,
            "output_path": outcome.output_path,
        }
        
        if execution_mode == "federated":
            result["data_dirs"] = list(outcome.data_dirs)
        elif execution_mode == "centralized":
            result["data_dir"] = outcome.data_dir
        elif execution_mode == "simulation":
            result["data_dir"] = outcome.data_dir
            result["client_dirs"] = list(outcome.client_dirs)
            
        return result