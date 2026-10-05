"""Step 2 — Index (offline): chunk, embed, save; rebuild only when plugin files change."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings

from .chunking import chunk_documents
from .corpus import CorpusDocument, iter_plugin_documents
from .embedder import TfidfModel, fit_tfidf


@dataclass
class StoredIndex:
    fingerprint: str
    chunks: list[CorpusDocument]
    model: TfidfModel
    matrix: list[list[float]]


def _fingerprint(documents: list[CorpusDocument]) -> str:
    payload = "|".join(f"{doc.path}:{doc.source_hash}" for doc in documents)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_path(package_dir: Path, settings: Settings) -> Path:
    return package_dir / settings.rag_index_dir_name / "index.json"


def _search_text(chunk: CorpusDocument) -> str:
    # Why: a prompt like "like c_f_test1.py" must match the filename, not only the code body.
    return f"{chunk.path}\n{chunk.text}"


def _write_index(path: Path, stored: StoredIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": stored.fingerprint,
        "chunks": [
            {"path": c.path, "text": c.text, "source_hash": c.source_hash}
            for c in stored.chunks
        ],
        "vocab": stored.model.vocab,
        "idf": stored.model.idf,
        "matrix": stored.matrix,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_index(raw: dict, fingerprint: str) -> StoredIndex:
    chunks = [
        CorpusDocument(
            path=item["path"],
            text=item["text"],
            source_hash=item["source_hash"],
        )
        for item in raw["chunks"]
    ]
    model = TfidfModel(vocab=raw["vocab"], idf=raw["idf"])
    return StoredIndex(
        fingerprint=fingerprint,
        chunks=chunks,
        model=model,
        matrix=raw["matrix"],
    )


def build_index(
    repo_root: Path,
    package_dir: Path,
    settings: Settings,
    documents: list[CorpusDocument] | None = None,
) -> StoredIndex:
    """Chunk + TF-IDF + write rag_index/index.json. Pass documents to skip a second disk walk."""
    if documents is None:
        documents = iter_plugin_documents(repo_root, settings)
    chunks = chunk_documents(documents, settings.rag_max_chunk_chars)
    searchable = [_search_text(chunk) for chunk in chunks]
    model = fit_tfidf(searchable)
    stored = StoredIndex(
        fingerprint=_fingerprint(documents),
        chunks=chunks,
        model=model,
        matrix=[model.transform(text) for text in searchable],
    )
    _write_index(_index_path(package_dir, settings), stored)
    return stored


def load_or_build_index(
    repo_root: Path,
    package_dir: Path,
    settings: Settings,
) -> StoredIndex:
    """Load saved index, or rebuild if missing or plugin files changed (fingerprint)."""
    documents = iter_plugin_documents(repo_root, settings)
    fingerprint = _fingerprint(documents)
    path = _index_path(package_dir, settings)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("fingerprint") == fingerprint:
            return _read_index(raw, fingerprint)
    return build_index(repo_root, package_dir, settings, documents=documents)
