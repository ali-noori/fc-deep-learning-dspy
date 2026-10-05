"""Local RAG for this tool only: search plugin .py files, then generate.

Read in this order (usual RAG, no extra framework):
  1. corpus.py      — load plugins/models/*.py
  2. index_store.py — chunk, TF-IDF, save rag_index/index.json
  3. retriever.py   — top-k for the user prompt (public: retrieve_plugin_context)
  4. architecture_generator.py — DSPy generate with retrieved_examples

chunking.py and embedder.py are helpers for step 2 (and cosine in step 3).
"""

from __future__ import annotations

from .retriever import RetrievedContext, retrieve_plugin_context

__all__ = ["RetrievedContext", "retrieve_plugin_context"]
