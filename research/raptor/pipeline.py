"""End-to-end RAPTOR pipeline: fit (build tree) → retrieve → build_context."""

from __future__ import annotations

import json

import numpy as np

from .chunking import chunk_corpus
from .embedding import EmbeddingModel
from .retrieval import CollapsedRetriever, RetrievalResult
from .summarization import BaseSummarizer, ExtractiveSummarizer
from .tree import Node, RaptorTreeBuilder


class RaptorPipeline:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        summarizer: BaseSummarizer | None = None,
        chunk_size: int = 100,
        chunk_overlap: int = 0,
        max_levels: int = 5,
    ):
        self.embedding = embedding_model  # lazy init to avoid downloading in tests that don't need it
        self.summarizer = summarizer or ExtractiveSummarizer()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.builder = RaptorTreeBuilder(summarizer=self.summarizer, max_levels=max_levels)
        self.nodes: list[Node] = []
        self._retriever: CollapsedRetriever | None = None

    def _ensure_embedding(self) -> EmbeddingModel:
        if self.embedding is None:
            self.embedding = EmbeddingModel()
        return self.embedding

    # ---- Build ----
    def fit(self, docs: list[str]) -> list[Node]:
        chunks = chunk_corpus(docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            raise ValueError("No chunks produced — corpus is empty.")
        emb = self._ensure_embedding()
        leaf_embs = emb.embed(chunks)
        self.nodes = self.builder.build(chunks, leaf_embs, embed_fn=emb.embed)
        self._retriever = CollapsedRetriever(self.nodes)
        return self.nodes

    # ---- Retrieve ----
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.nodes or self._retriever is None:
            raise RuntimeError("Call fit() before retrieve().")
        q = self._ensure_embedding().embed_query(query)
        return self._retriever.search(q, top_k=top_k)

    def build_context(self, query: str, top_k: int = 5, max_chars: int = 4000) -> str:
        results = self.retrieve(query, top_k=top_k)
        parts, total = [], 0
        for r in results:
            tag = f"[L{r.node.level}|{r.score:.3f}]"
            piece = f"{tag} {r.node.text}"
            if total + len(piece) > max_chars:
                break
            parts.append(piece)
            total += len(piece)
        return "\n\n".join(parts)

    # ---- Persist ----
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([n.to_dict() for n in self.nodes], f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "RaptorPipeline":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pipe = cls()
        pipe.nodes = [Node.from_dict(d) for d in data]
        pipe._retriever = CollapsedRetriever(pipe.nodes)
        return pipe

    # ---- Introspection ----
    def level_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for n in self.nodes:
            counts[n.level] = counts.get(n.level, 0) + 1
        return dict(sorted(counts.items()))

    def all_nodes(self) -> list[Node]:
        return self.nodes
