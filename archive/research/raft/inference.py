"""RAFT inference (offline): lọc distractor + trích quote verbatim + trả lời CoT.

Mô phỏng model đã qua RAFT-finetune:
  1. Score từng doc theo query (embedding cosine + lexical).
  2. Doc top-1 vượt threshold -> oracle, trích câu verbatim làm quote.
  3. Dưới threshold -> fallback parametric (như trường hợp (1-P)% lúc train).
  4. Trả lời theo format CoT + citation của paper.
Production: thay extractive span bằng LLM đã SFT với data từ data.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .reasoning import format_raft_answer


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", s.lower()))


def _word_f1(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    if inter == 0:
        return 0.0
    p, r = inter / len(wa), inter / len(wb)
    return 2 * p * r / (p + r)


@dataclass
class RAFTPrediction:
    answer: str
    quote: str
    used_doc_idx: int  # -1 = parametric fallback (không dùng doc nào)
    doc_scores: list[float]


class RAFTInference:
    def __init__(self, relevance_threshold: float = 0.30, embedding_model=None):
        self.relevance_threshold = relevance_threshold
        self._embedding_model = embedding_model

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

    def _score(self, query: str, doc: str) -> float:
        emb = self._emb().embed([query, doc])
        cos01 = (float(emb[0] @ emb[1]) + 1.0) / 2.0
        return 0.7 * cos01 + 0.3 * _word_f1(query, doc)

    @staticmethod
    def _best_sentence(doc: str, query: str) -> str:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.strip()) if s.strip()]
        if not sents:
            return doc[:300]
        return max(sents, key=lambda s: len(_words(s) & _words(query)))

    def predict(
        self,
        question: str,
        docs: list[str],
        parametric_answer: str = "I don't have enough evidence in the retrieved documents.",
    ) -> RAFTPrediction:
        if not docs:
            out = format_raft_answer("No documents retrieved, using parametric knowledge.", "", parametric_answer)
            return RAFTPrediction(answer=out, quote="", used_doc_idx=-1, doc_scores=[])
        scores = [self._score(question, d) for d in docs]
        best = int(np.argmax(scores))
        if scores[best] < self.relevance_threshold:
            out = format_raft_answer(
                "All retrieved documents look like distractors, so I rely on memorized domain knowledge.",
                "",
                parametric_answer,
            )
            # Xóa quote rỗng cho đúng format verify (fallback không claim citation).
            return RAFTPrediction(answer=out, quote="", used_doc_idx=-1, doc_scores=scores)
        quote = self._best_sentence(docs[best], question)
        reasoning = (
            f"Doc {best+1} is the most relevant (score {scores[best]:.2f}); "
            f"the other {len(docs)-1} document(s) are treated as distractors and ignored."
        )
        return RAFTPrediction(
            answer=format_raft_answer(reasoning, quote, quote),
            quote=quote,
            used_doc_idx=best,
            doc_scores=scores,
        )
