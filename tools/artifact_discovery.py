"""Generic artifact discovery for DSPy agents and other tooling."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


class ArtifactDiscovery:
    """Scan a directory and map artifact names to repo-relative file paths."""

    def __init__(
        self,
        target_dir: Path | str,
        *,
        repo_root: Path | str | None = None,
    ) -> None:
        self._target_dir = Path(target_dir)
        self._repo_root = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT

    @property
    def target_dir(self) -> Path:
        return self._target_dir

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def discover(self) -> dict[str, str]:
        """
        Return artifacts under target_dir, including nested subfolders
        (e.g. plugins/models/generated_architecture/customized_model_v2.py).

        Keys are filenames (e.g. FedAvg.py); values are paths relative to repo_root
        (e.g. plugins/aggregators/FedAvg.py).

        NOTE: if two files in different subfolders share the same filename,
        the later one (per rglob traversal order) overwrites the earlier one
        in the returned dict, since the key is the filename only.
        """
        if not self._target_dir.exists() or not self._target_dir.is_dir():
            return {}

        artifacts: dict[str, str] = {}
        for entry in self._target_dir.rglob("*.py"):
            if not entry.is_file():
                continue
            if entry.name == "__init__.py" or entry.name.startswith("."):
                continue
            artifacts[entry.name] = entry.relative_to(self._repo_root).as_posix()

        return dict(sorted(artifacts.items(), key=lambda item: item[0].casefold()))
