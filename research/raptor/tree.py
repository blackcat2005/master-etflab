"""Tree construction: recursive cluster + summarize until convergence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import cluster_embeddings
from .summarization import BaseSummarizer, ExtractiveSummarizer


@dataclass
class Node:
    id: int
    text: str
    embedding: np.ndarray
    level: int  # 0 = leaf chunk, >0 = summary
    children: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "level": self.level,
            "children": self.children,
        }

    @staticmethod
    def from_dict(d: dict) -> "Node":
        return Node(
            id=d["id"],
            text=d["text"],
            embedding=np.asarray(d["embedding"], dtype=np.float32),
            level=d["level"],
            children=d.get("children", []),
        )


class RaptorTreeBuilder:
    def __init__(
        self,
        summarizer: BaseSummarizer | None = None,
        max_levels: int = 5,
        min_cluster_size: int = 6,
    ):
        self.summarizer = summarizer or ExtractiveSummarizer()
        self.max_levels = max_levels
        self.min_cluster_size = min_cluster_size

    def build(
        self,
        leaf_texts: list[str],
        leaf_embeddings: np.ndarray,
        embed_fn=None,
    ) -> list[Node]:
        """Build the full tree. `embed_fn` embeds new summary texts."""
        nodes: list[Node] = []
        next_id = 0

        cur_texts = list(leaf_texts)
        cur_embs = np.asarray(leaf_embeddings, dtype=np.float32)
        cur_ids: list[int] = []
        for t, e in zip(cur_texts, cur_embs):
            nodes.append(Node(id=next_id, text=t, embedding=e, level=0))
            cur_ids.append(next_id)
            next_id += 1

        level = 0
        while len(cur_texts) > 1 and level < self.max_levels:
            if len(cur_texts) <= self.min_cluster_size:
                # One final summary parent if more than 1 node.
                labels = np.zeros(len(cur_texts), dtype=int)
            else:
                labels = cluster_embeddings(cur_embs)
            n_parents = int(labels.max()) + 1
            if n_parents <= 1 and len(cur_texts) <= self.min_cluster_size:
                # Collapse remaining into a single root, then stop.
                summary = self.summarizer.summarize(cur_texts, cur_embs)
                emb = self._embed([summary], cur_embs, embed_fn)
                nodes.append(Node(id=next_id, text=summary, embedding=emb[0], level=level + 1, children=list(cur_ids)))
                next_id += 1
                break
            if n_parents <= 1:
                break  # no further split possible

            next_texts, next_embs_list, next_ids = [], [], []
            for c in range(n_parents):
                idx = np.where(labels == c)[0]
                if len(idx) == 0:
                    continue
                c_texts = [cur_texts[i] for i in idx]
                c_embs = cur_embs[idx]
                summary = self.summarizer.summarize(c_texts, c_embs)
                emb = self._embed([summary], c_embs, embed_fn)
                parent_id = next_id
                next_id += 1
                child_ids = [cur_ids[i] for i in idx]
                nodes.append(Node(id=parent_id, text=summary, embedding=emb[0], level=level + 1, children=child_ids))
                next_texts.append(summary)
                next_embs_list.append(emb[0])
                next_ids.append(parent_id)

            cur_texts = next_texts
            cur_embs = np.stack(next_embs_list).astype(np.float32)
            cur_ids = next_ids
            level += 1

        return nodes

    @staticmethod
    def _embed(texts: list[str], fallback_embs: np.ndarray, embed_fn) -> np.ndarray:
        if embed_fn is not None:
            try:
                return np.asarray(embed_fn(texts), dtype=np.float32)
            except Exception:
                pass
        # Fallback: mean of children embeddings (keeps pipeline offline-safe).
        mean = fallback_embs.mean(axis=0, keepdims=True)
        mean = mean / (np.linalg.norm(mean, axis=1, keepdims=True) + 1e-9)
        return np.repeat(mean, len(texts), axis=0).astype(np.float32)
