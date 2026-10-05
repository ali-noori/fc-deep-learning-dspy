"""Local TF-IDF: same vectors for indexing files and for the user prompt (no cloud API)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN.findall(text)]


@dataclass
class TfidfModel:
    vocab: dict[str, int]
    idf: list[float]

    def transform(self, text: str) -> list[float]:
        counts: dict[int, int] = {}
        for token in tokenize(text):
            index = self.vocab.get(token)
            if index is None:
                continue
            counts[index] = counts.get(index, 0) + 1
        dim = len(self.vocab)
        vector = [0.0] * dim
        if not counts:
            return vector
        length = sum(counts.values())
        for index, count in counts.items():
            tf = count / length
            vector[index] = tf * self.idf[index]
        return vector


def fit_tfidf(texts: list[str]) -> TfidfModel:
    doc_tokens = [tokenize(text) for text in texts]
    vocab: dict[str, int] = {}
    for tokens in doc_tokens:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    n_docs = max(len(doc_tokens), 1)
    df = [0] * len(vocab)
    for tokens in doc_tokens:
        seen = set(tokens)
        for token in seen:
            df[vocab[token]] += 1
    idf = [math.log((1 + n_docs) / (1 + count)) + 1.0 for count in df]
    return TfidfModel(vocab=vocab, idf=idf)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
