"""Step 3 — Retrieve (online): prompt → vector → top-k chunks → text for the LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Settings

from .embedder import cosine_similarity
from .index_store import StoredIndex, load_or_build_index

_PACKAGE_DIR = Path(__file__).resolve().parents[1]

# Tiny-corpus boosts so everyday words still pick the right plugin.
_PATH_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cnn.py", ("cnn", "conv2d", "conv", "image", "vision", "convolution")),
    ("mlp.py", ("mlp", "linear", "tabular", "fully connected", "dense")),
    ("resnet18.py", ("resnet", "resnet18")),
    ("efficientnet_b0.py", ("efficientnet", "efficientnet_b0")),
    ("mobilenetv3_small.py", ("mobilenet", "mobilenetv3")),
    ("tabnet.py", ("tabnet",)),
    ("ft_transformer.py", ("ft-transformer", "ft_transformer", "ft transformer")),
    ("tabtransformer.py", ("tabtransformer", "tab transformer")),
)


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    paths: tuple[str, ...]


def _load_contract() -> str:
    return (_PACKAGE_DIR / "plugin_contract.md").read_text(encoding="utf-8").strip()


def _hybrid_boost(path: str, query: str) -> float:
    lowered = query.lower()
    name = path.split("/")[-1].split("#")[0].lower()
    stem = name[:-3] if name.endswith(".py") else name
    boost = 0.0
    if name in lowered or stem in lowered:
        boost += 0.25
    for filename, keywords in _PATH_KEYWORDS:
        if name != filename:
            continue
        if any(key in lowered for key in keywords):
            boost += 0.15
    return boost


def _cnn_chunk_index(index: StoredIndex) -> int | None:
    for i, chunk in enumerate(index.chunks):
        if chunk.path.endswith("plugins/models/cnn.py") or chunk.path.endswith("cnn.py"):
            return i
    return None


def _top_k_indices(index: StoredIndex, query: str, top_k: int) -> list[int]:
    query_vec = index.model.transform(query)
    ranked: list[tuple[float, int]] = []
    for i, vector in enumerate(index.matrix):
        score = cosine_similarity(query_vec, vector)
        score += _hybrid_boost(index.chunks[i].path, query)
        ranked.append((score, i))
    ranked.sort(key=lambda item: item[0], reverse=True)

    picked: list[int] = []
    for score, i in ranked:
        if len(picked) >= top_k:
            break
        if score > 0.0:
            picked.append(i)

    # Why: all-zero scores still need a teacher file; prefer the built-in CNN plugin.
    if not picked:
        fallback = _cnn_chunk_index(index)
        picked.append(fallback if fallback is not None else 0)
    return picked[:top_k]


def _to_prompt(
    contract: str,
    index: StoredIndex,
    picked: list[int],
    max_chars: int,
) -> RetrievedContext:
    blocks = [f"### Plugin contract\n{contract}"]
    paths: list[str] = []
    for i in picked:
        chunk = index.chunks[i]
        body = chunk.text
        if len(body) > max_chars:
            body = body[:max_chars] + "\n# ... truncated ..."
        blocks.append(f"### {chunk.path}\n{body}")
        paths.append(chunk.path)
    return RetrievedContext(text="\n\n".join(blocks), paths=tuple(paths))


def retrieve_plugin_context(
    repo_root: Path,
    query: str,
    settings: Settings,
) -> RetrievedContext:
    """Public retrieve step. If RAG is off, return only the plugin contract."""
    contract = _load_contract()
    if not settings.rag_enabled:
        return RetrievedContext(text=contract, paths=())

    stored = load_or_build_index(repo_root, _PACKAGE_DIR, settings)
    picked = _top_k_indices(stored, query, settings.rag_top_k)
    return _to_prompt(contract, stored, picked, settings.rag_max_prompt_chars)
