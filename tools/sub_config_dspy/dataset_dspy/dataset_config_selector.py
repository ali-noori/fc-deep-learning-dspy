"""DSPy Signature + Module: natural language → dataset config fields (LM only)."""

from __future__ import annotations

import re
from pathlib import Path

from .config import Settings
from .dspy_support import dspy
from .types import DatasetFederatedExtractionResult
from .types import DatasetCentralizedExtractionResult 
from .types import DatasetSimulationExtractionResult


class ParseDatasetFederatedConfigSignature(dspy.Signature):
    """Extract federated dataset layout fields from the developer's answer.

    Federated means real multi-client folders: each client has its own data_dir path.

    Decision rules:
    - Ignore typos, informal wording, and incomplete phrasing; infer the intended values.
    - If the user mentions unrelated extras (model, loss, aggregator, etc.), ignore them.
    - Do not invent missing required fields; only extract what is stated or clearly implied.
    - Synonyms: client folders / sites / data_dir1,data_dir2 → data_dirs.
    - data_dirs must be relative to the FeatureCloud mounted data/ directory (for --client-dirs).
      If the user gives an absolute path containing .../data/<rest>, output only <rest>
      (keep subfolders, e.g. .../data/sample_data/c1 → sample_data/c1;
      .../data/siteA → siteA). If already relative (sample_data/c1), keep it.
      Strip a leading ./ . Do not emit Windows drive letters or full host paths.
      If the user writes data/<name> meaning under the mount, output <name> (avoid data/data/...).
    - Join data_dirs with commas in stated order.
    - train/test values must be file names only (basename), never directories and never prefixed with data/.
      Keep the exact basename stem the user stated. For the common pair train.npz/test.npz: if the user
      writes train.npz but mistypes the other as test.npy (or the reverse), correct the mistyped one to .npz.
      Only keep .npy when the user clearly intends numpy files (e.g. both train.npy and test.npy).
    - logic_dir is the folder name under mnt/input (often "data"), not a host path and not part of train/test.
    """

    user_message: str = dspy.InputField(
        desc=(
            "Developer natural-language answer about federated dataset layout; "
            "may be short, long, messy, incomplete, or misspelled, and may not "
            "use exact field names; may include long project paths containing \\data\\..."
        )
    )
    data_dirs: str = dspy.OutputField(
        desc=(
            "Comma-separated client dirs relative to the FeatureCloud data/ mount "
            "(e.g. sample_data/c1,sample_data/c2), not absolute host paths"
        )
    )
    train_dataset_file_name: str = dspy.OutputField(
        desc=(
            "Train file name only (e.g. train.npz). Correct common .npy typos to .npz "
            "when the user meant the standard npz pair; not a full path."
        )
    )
    test_dataset_file_name: str = dspy.OutputField(
        desc=(
            "Test file name only (e.g. test.npz). If user wrote test.npy but train is train.npz "
            "as a misspelling of the usual pair, output test.npz; not a full path."
        )
    )
    logic_dir: str = dspy.OutputField(
        desc="Folder name under mnt/input that contains split subfolders (e.g. data)."
    )


class ParseDatasetCentralizedConfigSignature(dspy.Signature):
    """Extract centralized dataset layout fields from the developer's answer.

    Centralized means a single data directory on one machine (no multi-client federation).

    Decision rules:
    - Ignore typos, informal wording, and incomplete phrasing; infer the intended values.
    - If the user mentions unrelated extras (model, loss, etc.), ignore them.
    - Do not invent missing required fields; only extract what is stated or clearly implied.
    - Exactly one data_dir; never invent multiple clients.
    - If the user says "client" but clearly gives only one folder, treat it as data_dir.
    - data_dir must be relative to the FeatureCloud mounted data/ directory (for --client-dirs).
      Absolute .../data/<rest> → output <rest> (e.g. .../data/sample_data_centralized → sample_data_centralized).
      Already relative → keep it. Strip leading ./. No drive letters or full host paths.
      If the user writes data/<name> meaning under the mount, output <name>.
    - train/test values must be file names only (basename), never directories and never prefixed with data/.
      Keep the exact basename stem the user stated. For the common pair train.npz/test.npz: if the user
      writes train.npz but mistypes the other as test.npy (or the reverse), correct the mistyped one to .npz.
      Only keep .npy when the user clearly intends numpy files (e.g. both train.npy and test.npy).
    - logic_dir is the folder name under mnt/input (often "data"), not a host path and not part of train/test.
    """

    user_message: str = dspy.InputField(
        desc=(
            "Developer natural-language answer about centralized dataset layout; "
            "may be short, long, messy, incomplete, or misspelled, and may not "
            "use exact field names; may include long project paths containing \\data\\..."
        )
    )
    data_dir: str = dspy.OutputField(
        desc=(
            "Single data directory relative to the FeatureCloud data/ mount "
            "(e.g. sample_data_centralized), not an absolute host path"
        )
    )
    train_dataset_file_name: str = dspy.OutputField(
        desc=(
            "Train file name only (e.g. train.npz). Correct common .npy typos to .npz "
            "when the user meant the standard npz pair; not a full path."
        )
    )
    test_dataset_file_name: str = dspy.OutputField(
        desc=(
            "Test file name only (e.g. test.npz). If user wrote test.npy but train is train.npz "
            "as a misspelling of the usual pair, output test.npz; not a full path."
        )
    )
    logic_dir: str = dspy.OutputField(
        desc="Folder name under mnt/input that contains split subfolders (e.g. data)."
    )


class ParseDatasetSimulationConfigSignature(dspy.Signature):
    """Extract simulation dataset layout fields from the developer's answer.

    Simulation means one parent data_dir plus local fake client subfolders (not real remote clients).

    Decision rules:
    - Ignore typos, informal wording, and incomplete phrasing; infer the intended values.
    - If the user mentions unrelated extras (model, loss, etc.), ignore them.
    - Do not invent missing required fields; only extract what is stated or clearly implied.
    - data_dir is the parent simulation root relative to the FeatureCloud mounted data/ directory.
      Absolute .../data/<rest> → output <rest>
      (e.g. .../data/sample_data_simulation → sample_data_simulation).
      Already relative → keep it. Strip leading ./. No drive letters or full host paths.
    - client_dirs are local fake client folder names under that root (e.g. c1,c2). If the user
      gives full paths, output only the client folder names (basenames), comma-separated.
    - train/test values must be file names only (basename), never directories and never prefixed with data/.
      Keep the exact basename stem the user stated. For the common pair train.npz/test.npz: if the user
      writes train.npz but mistypes the other as test.npy (or the reverse), correct the mistyped one to .npz.
      Only keep .npy when the user clearly intends numpy files (e.g. both train.npy and test.npy).
    - logic_dir is the folder name under mnt/input (often "data"), not a host path and not part of train/test.
    """

    user_message: str = dspy.InputField(
        desc=(
            "Developer natural-language answer about simulation dataset layout; "
            "may be short, long, messy, incomplete, or misspelled, and may not "
            "use exact field names; may include long project paths containing \\data\\..."
        )
    )
    data_dir: str = dspy.OutputField(
        desc=(
            "Parent simulation directory relative to the FeatureCloud data/ mount "
            "(e.g. sample_data_simulation), not an absolute host path"
        )
    )
    client_dirs: str = dspy.OutputField(
        desc=(
            "Comma-separated local fake client folder names under data_dir "
            "(e.g. c1,c2), usually basenames rather than full host paths"
        )
    )
    train_dataset_file_name: str = dspy.OutputField(
        desc=(
            "Train file name only (e.g. train.npz). Correct common .npy typos to .npz "
            "when the user meant the standard npz pair; not a full path."
        )
    )
    test_dataset_file_name: str = dspy.OutputField(
        desc=(
            "Test file name only (e.g. test.npz). If user wrote test.npy but train is train.npz "
            "as a misspelling of the usual pair, output test.npz; not a full path."
        )
    )
    logic_dir: str = dspy.OutputField(
        desc="Folder name under mnt/input that contains split subfolders (e.g. data)."
    )


def _split_data_dirs(raw: str) -> tuple[str, ...]:
    parts = re.split(r"[,;\n]+", raw)
    return tuple(p.strip().strip('"').strip("'") for p in parts if p.strip())


def _split_client_dirs(raw: str) -> tuple[str, ...]:
    parts = re.split(r"[,;\n]+", raw)
    return tuple(p.strip().strip('"').strip("'") for p in parts if p.strip())

def _validate_federated_extraction(
    data_dirs_raw: str,
    train_raw: str,
    test_raw: str,
    logic_raw: str,
) -> tuple[tuple[str, ...], str, str, str]:
    data_dirs = _split_data_dirs(data_dirs_raw)
    if not data_dirs:
        raise ValueError("Could not extract any data_dir paths from the response.")

    train = train_raw.strip().strip('"').strip("'")
    test = test_raw.strip().strip('"').strip("'")
    logic_dir = logic_raw.strip().strip('"').strip("'").strip("/\\")

    if not train or not test or not logic_dir:
        raise ValueError(
            "Missing train_dataset_file_name, test_dataset_file_name, or logic_dir."
        )

    return (
        data_dirs,
        Path(train).name,
        Path(test).name,
        logic_dir,
    )
    
def _validate_centralized_extraction(
    data_dir_raw: str,
    train_raw: str,
    test_raw: str,
    logic_raw: str,
) -> tuple[str, str, str, str]:
    data_dir = data_dir_raw.strip().strip('"').strip("'")
    train = train_raw.strip().strip('"').strip("'")
    test = test_raw.strip().strip('"').strip("'")
    logic_dir = logic_raw.strip().strip('"').strip("'").strip("/\\")

    if not data_dir:
        raise ValueError(
            "Could not extract any data_dir path from the response."
        )

    train = train_raw.strip().strip('"').strip("'")
    test = test_raw.strip().strip('"').strip("'")
    logic_dir = logic_raw.strip().strip('"').strip("'").strip("/\\")

    if not train or not test or not logic_dir:
        raise ValueError(
            "Missing train_dataset_file_name, test_dataset_file_name, or logic_dir."
        )

    return (
        data_dir,
        Path(train).name,
        Path(test).name,
        logic_dir,
    )
    
def _validate_simulation_extraction(
    data_dir_raw: str,
    client_dirs_raw: str,
    train_raw: str,
    test_raw: str,
    logic_raw: str,
) -> tuple[str, tuple[str, ...], str, str, str]:
    data_dir = data_dir_raw.strip().strip('"').strip("'")
    client_dirs = _split_client_dirs(client_dirs_raw)
    if not client_dirs:
        raise ValueError("Could not extract any client_dir paths from the response.")
    train = train_raw.strip().strip('"').strip("'")
    test = test_raw.strip().strip('"').strip("'")
    logic_dir = logic_raw.strip().strip('"').strip("'").strip("/\\")

    if not train or not test or not logic_dir:
        raise ValueError(
            "Missing train_dataset_file_name, test_dataset_file_name, or logic_dir."
        )

    return (
        data_dir,
        client_dirs,
        Path(train).name,
        Path(test).name,
        logic_dir,
    )


class DatasetFederatedConfigModule(dspy.Module):
    """Chain-of-Thought extractor for dataset config fields."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self.extract = dspy.ChainOfThought(ParseDatasetFederatedConfigSignature)

    def forward(self, user_message: str) -> DatasetFederatedExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")

        result: DatasetFederatedExtractionResult | None = None
        last_error: Exception | None = None

        for _ in range(self._settings.dataset_extract_max_retries):
            try:
                prediction = self.extract(user_message=current_request)
                data_dirs, train_name, test_name, logic_dir = _validate_federated_extraction(
                    str(getattr(prediction, "data_dirs", "") or ""),
                    str(getattr(prediction, "train_dataset_file_name", "") or ""),
                    str(getattr(prediction, "test_dataset_file_name", "") or ""),
                    str(getattr(prediction, "logic_dir", "") or ""),
                )

                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                result = DatasetFederatedExtractionResult(
                    data_dirs=data_dirs,
                    train_dataset_file_name=train_name,
                    test_dataset_file_name=test_name,
                    logic_dir=logic_dir,
                    backend=backend,
                    model=model,
                    base_url=base_url,
                )
                break
            except Exception as exc:
                last_error = exc
                current_request = (
                    current_request
                    + "\n\nYour last answer was incomplete or invalid. "
                    "Reply with data_dir paths, train_dataset_file_name, "
                    "test_dataset_file_name, and logic_dir."
                )

        if result is None:
            raise RuntimeError(
                "Dataset config extraction failed after LM retries."
            ) from last_error

        return result
    
class DatasetCentralizedConfigModule(dspy.Module):
    """Chain-of-Thought extractor for centralized dataset config fields."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self.extract = dspy.ChainOfThought(ParseDatasetCentralizedConfigSignature)

    def forward(self, user_message: str) -> DatasetCentralizedExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")
        
        result: DatasetCentralizedExtractionResult | None = None
        last_error: Exception | None = None
        
        for _ in range(self._settings.dataset_extract_max_retries):
            try:
                prediction = self.extract(user_message=current_request)
                data_dir, train_name, test_name, logic_dir = _validate_centralized_extraction(
                    str(getattr(prediction, "data_dir", "") or ""),
                    str(getattr(prediction, "train_dataset_file_name", "") or ""),
                    str(getattr(prediction, "test_dataset_file_name", "") or ""),
                    str(getattr(prediction, "logic_dir", "") or ""),
                )
                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)

                result = DatasetCentralizedExtractionResult(
                    data_dir=data_dir,
                    train_dataset_file_name=train_name,
                    test_dataset_file_name=test_name,
                    logic_dir=logic_dir,
                    backend=backend,
                    model=model,
                    base_url=base_url,
                )
                break
            except Exception as exc:
                last_error = exc
                current_request = (
                    current_request
                    + "\n\nYour last answer was incomplete or invalid. "
                    "Reply with data_dir path, train_dataset_file_name, "
                    "test_dataset_file_name, and logic_dir."
                )

        if result is None:
            raise RuntimeError(
                "Dataset config extraction failed after LM retries."
            ) from last_error
            
        return result
    
class DatasetSimulationConfigModule(dspy.Module):
    """Chain-of-Thought extractor for simulation dataset config fields."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self.extract = dspy.ChainOfThought(ParseDatasetSimulationConfigSignature)

    def forward(self, user_message: str) -> DatasetSimulationExtractionResult:
        current_request = user_message.strip()
        if not current_request:
            raise ValueError("user_message must be non-empty")
        
        result: DatasetSimulationExtractionResult | None = None
        last_error: Exception | None = None
        
        for _ in range(self._settings.dataset_extract_max_retries):
            try:
                prediction = self.extract(user_message=current_request)
                data_dir, client_dirs, train_name, test_name, logic_dir = _validate_simulation_extraction(
                    str(getattr(prediction, "data_dir", "") or ""),
                    str(getattr(prediction, "client_dirs", "") or ""),
                    str(getattr(prediction, "train_dataset_file_name", "") or ""),
                    str(getattr(prediction, "test_dataset_file_name", "") or ""),
                    str(getattr(prediction, "logic_dir", "") or ""),
                )
                lm = dspy.settings.lm
                backend = getattr(lm, "last_backend", None) or "unknown"
                model = getattr(lm, "last_model", None) or getattr(lm, "model", "unknown")
                base_url = getattr(lm, "last_base_url", None)
                
                result = DatasetSimulationExtractionResult(
                    data_dir=data_dir,
                    client_dirs=client_dirs,
                    train_dataset_file_name=train_name,
                    test_dataset_file_name=test_name,
                    logic_dir=logic_dir,
                    backend=backend,
                    model=model,
                    base_url=base_url,
                )
                break
            except Exception as exc:
                last_error = exc
                current_request = (
                    current_request
                    + "\n\nYour last answer was incomplete or invalid. "
                    "Reply with data_dir, client_dirs, train_dataset_file_name, "
                    "test_dataset_file_name, and logic_dir."
                )

        if result is None:
            raise RuntimeError(
                "Dataset config extraction failed after LM retries."
            ) from last_error
            
        return result