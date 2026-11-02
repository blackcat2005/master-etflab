"""Retrieval over the collapsed tree (paper's best-performing strategy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tree import Node


@dataclass
class RetrievalResult:
    node: Node
    score: float


class CollapsedRetriever:
    """Flatten all nodes, rank by cosine similarity to the query."""

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        if nodes:
            mat = np.stack([n.embedding for n in nodes]).astype(np.float32)
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
            self.matrix = mat
        else:
            self.matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if not self.nodes:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self.matrix @ q
        k = min(top_k, len(self.nodes))
        idx = np.argsort(-scores)[:k]
        return [RetrievalResult(node=self.nodes[i], score=float(scores[i])) for i in idx]
