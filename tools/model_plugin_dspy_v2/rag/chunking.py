"""Used only while indexing: one plugin file is one chunk unless the file is huge."""

from __future__ import annotations

from .corpus import CorpusDocument


def chunk_documents(
    documents: list[CorpusDocument],
    max_chars: int,
) -> list[CorpusDocument]:
    """Keep class Model in one piece when possible; split huge files on blank lines."""
    chunks: list[CorpusDocument] = []
    for doc in documents:
        if len(doc.text) <= max_chars:
            chunks.append(doc)
            continue
        pieces = _split_on_blank_lines(doc.text, max_chars)
        for index, piece in enumerate(pieces):
            chunks.append(
                CorpusDocument(
                    path=f"{doc.path}#part{index + 1}",
                    text=piece,
                    source_hash=doc.source_hash,
                )
            )
    return chunks


def _split_on_blank_lines(text: str, max_chars: int) -> list[str]:
    blocks = text.split("\n\n")
    pieces: list[str] = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        current = block if len(block) <= max_chars else block[:max_chars]
    if current:
        pieces.append(current)
    return pieces or [text[:max_chars]]
