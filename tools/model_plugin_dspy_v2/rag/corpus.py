"""Step 1 — Corpus: load plugin .py files from disk (no vectors yet)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings


@dataclass(frozen=True)
class CorpusDocument:
    path: str
    text: str
    source_hash: str


def iter_plugin_documents(repo_root: Path, settings: Settings) -> list[CorpusDocument]:
    """Walk plugins/models. Skip generated_architecture (scratch output of this tool)."""
    root = repo_root.joinpath(*settings.rag_corpus_rel)
    if not root.is_dir():
        raise RuntimeError(f"RAG corpus directory not found: {root}")

    exclude_dirs = {name.lower() for name in settings.rag_exclude_dir_names}
    exclude_files = {name.lower() for name in settings.rag_exclude_filenames}
    documents: list[CorpusDocument] = []

    for path in sorted(root.rglob("*.py")):
        if any(part.lower() in exclude_dirs for part in path.parts):
            continue
        if path.name.lower() in exclude_files:
            continue
        rel = path.relative_to(repo_root).as_posix()
        documents.append(
            CorpusDocument(
                path=rel,
                text=path.read_text(encoding="utf-8"),
                source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )

    if not documents:
        raise RuntimeError(f"RAG corpus is empty under {root}")
    return documents
