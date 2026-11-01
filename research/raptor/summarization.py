"""Summarizers: turn a cluster of chunks into one parent-node summary."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, texts: list[str], embeddings: np.ndarray | None = None) -> str:
        ...


def _sentences(texts: list[str]) -> list[str]:
    out: list[str] = []
    for t in texts:
        out.extend([s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()])
    return out


class ExtractiveSummarizer(BaseSummarizer):
    """Offline baseline: pick sentences closest to the cluster centroid.

    Production should replace this with an abstractive LLM call.
    """

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def summarize(self, texts: list[str], embeddings: np.ndarray | None = None) -> str:
        if not texts:
            return ""
        if len(texts) == 1:
            sents = _sentences(texts)
            return " ".join(sents[: self.max_sentences]) or texts[0]
        if embeddings is None or len(embeddings) != len(texts):
            # Fallback: first sentences.
            return " ".join(_sentences(texts)[: self.max_sentences])
        centroid = embeddings.mean(axis=0)
        norms = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(centroid) + 1e-9) + 1e-9
        sims = (embeddings @ centroid) / norms
        order = np.argsort(-sims)[: self.max_sentences]
        picked = [texts[i] for i in sorted(order)]
        joined = " ".join(picked)
        sents = _sentences([joined])
        return " ".join(sents[: self.max_sentences * 2]) or joined


class OpenAISummarizer(BaseSummarizer):
    """Stub for production abstractive summarization.

    Subclass and override _call_llm, or pass any callable with the same
    signature, to plug in OpenAI / local LLM (GPT-4, Llama-3, Qwen...).
    Paper prompt: summarize cluster into a coherent ~150-word summary.
    """

    PROMPT = (
        "Summarize the following cluster of texts into a single coherent summary "
        "that preserves key facts, entities and relations (max ~150 words):\n\n{chunk}\n\nSummary:"
    )

    def __init__(self, call_llm=None, model: str = "gpt-4o-mini"):
        self.call_llm = call_llm
        self.model = model
        self.fallback = ExtractiveSummarizer()

    def _call_llm(self, prompt: str) -> str:
        if self.call_llm is not None:
            return self.call_llm(prompt)
        raise NotImplementedError("Provide `call_llm` (your LLM API function) to use OpenAISummarizer.")

    def summarize(self, texts: list[str], embeddings: np.ndarray | None = None) -> str:
        chunk = "\n---\n".join(texts)
        try:
            return self._call_llm(self.PROMPT.format(chunk=chunk)).strip()
        except Exception:
            return self.fallback.summarize(texts, embeddings)
