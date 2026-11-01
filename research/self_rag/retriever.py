"""In-memory dense retriever (cosine search, L2-normalized)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Passage:
    id: int
    text: str


@dataclass
class Hit:
    passage: Passage
    score: float


class InMemoryRetriever:
    def __init__(self, embedding_model=None):
        # Lazy import để test không cần tải model khi chưa dùng.
        self._embedding_model = embedding_model
        self.passages: list[Passage] = []
        self._matrix: np.ndarray | None = None

    def _emb(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            class _ST:
                def __init__(self):
                    self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

                def embed(self, texts: list[str]) -> np.ndarray:
                    v = np.asarray(
                        self.m.encode(texts, convert_to_numpy=True, show_progress_bar=False),
                        dtype=np.float32,
                    )
                    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

            self._embedding_model = _ST()
        return self._embedding_model

    def add(self, texts: list[str]) -> None:
        start = len(self.passages)
        for i, t in enumerate(texts):
            self.passages.append(Passage(id=start + i, text=t))
        self._matrix = None  # invalidate cache

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            vecs = self._emb().embed([p.text for p in self.passages])
            self._matrix = np.asarray(vecs, dtype=np.float32)
        return self._matrix

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self.passages:
            return []
        mat = self._ensure_matrix()
        q = self._emb().embed([query])[0]
        scores = mat @ q
        k = min(top_k, len(self.passages))
        idx = np.argsort(-scores)[:k]
        return [Hit(passage=self.passages[i], score=float(scores[i])) for i in idx]
