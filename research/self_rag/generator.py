"""Generators: sinh candidate segment cho mỗi passage.

- ExtractiveGenerator (offline): chọn câu trong passage giống query nhất.
- LlmGenerator (production stub): gọi LLM với (query, passage), fallback extractive.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


class BaseGenerator(ABC):
    @abstractmethod
    def draft(self, query: str, passage: str) -> str:
        ...


class ExtractiveGenerator(BaseGenerator):
    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def draft(self, query: str, passage: str) -> str:
        sents = _sentences(passage)
        if not sents:
            return passage[:300]
        try:
            from .critic import word_f1

            ranked = sorted(sents, key=lambda s: word_f1(s, query), reverse=True)
            top = ranked[: self.max_sentences]
            # Giữ thứ tự gốc cho mạch văn.
            order = sorted(top, key=sents.index)
            return " ".join(order)
        except Exception:
            return " ".join(sents[: self.max_sentences])


class LlmGenerator(BaseGenerator):
    PROMPT = (
        "Answer the question using ONLY the passage below. "
        "If the passage does not contain the answer, reply exactly: NOT_SUPPORTED.\n\n"
        "Question: {query}\nPassage: {passage}\nAnswer:"
    )

    def __init__(self, call_llm=None):
        self.call_llm = call_llm
        self.fallback = ExtractiveGenerator()

    def draft(self, query: str, passage: str) -> str:
        if self.call_llm is None:
            return self.fallback.draft(query, passage)
        try:
            out = self.call_llm(self.PROMPT.format(query=query, passage=passage)).strip()
            return out
        except Exception:
            return self.fallback.draft(query, passage)
