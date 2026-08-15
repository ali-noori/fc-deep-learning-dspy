"""Shared data shapes for the dataset_dspy pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetFederatedConfig:
    data_dirs: tuple[str, ...]
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str
    
@dataclass(frozen=True)
class DatasetCentralizedConfig:
    data_dir: str
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str
    
@dataclass(frozen=True)
class DatasetSimulationConfig:
    data_dir: str
    client_dirs: tuple[str, ...]
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str


@dataclass(frozen=True)
class DatasetFederatedExtractionResult:
    data_dirs: tuple[str, ...]
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str
    backend: str
    model: str
    base_url: str | None
    
@dataclass(frozen=True)
class DatasetCentralizedExtractionResult:
    data_dir: str
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str
    backend: str
    model: str
    base_url: str | None
    
@dataclass(frozen=True)
class DatasetSimulationExtractionResult:
    data_dir: str
    client_dirs: tuple[str, ...]
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str
    backend: str
    model: str
    base_url: str | None

@dataclass(frozen=True)
class DatasetFederatedConfigResult:
    data_dirs: tuple[str, ...]
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
    
@dataclass(frozen=True)
class DatasetCentralizedConfigResult:
    data_dir: str
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
    
@dataclass(frozen=True)
class DatasetSimulationConfigResult:
    data_dir: str
    client_dirs: tuple[str, ...]
    train_dataset_file_name: str
    test_dataset_file_name: str
    logic_dir: str
    agent_reply: str
    user_message: str
    backend: str
    model: str
    base_url: str | None
    output_path: str
