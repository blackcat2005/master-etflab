"""Embedding wrapper (SBERT by default, L2-normalized for cosine search)."""

from __future__ import annotations

import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norms

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
