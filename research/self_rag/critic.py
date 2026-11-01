"""Offline critic: xấp xỉ Critic model của paper bằng embedding + lexical overlap.

Paper huấn luyện Critic (GPT-4 distill) để gán reflection tokens.
Ở đây dùng heuristic minh bạch, chạy offline, đủ để test pipeline:
  - Retrieve: rule đơn giản (chitchat ngắn -> No, còn lại Yes).
  - IsRel: cosine(query, passage).
  - IsSup: max(cosine(segment, passage), word-F1(segment, passage)).
  - IsUse: kết hợp IsSup + độ phủ query + độ dài hợp lý.
Production: thay bằng LLM gọi prompt như paper (xem LlmCritic stub bên dưới).
"""

from __future__ import annotations

import re

import numpy as np

from .reflection import IsRel, IsSup, IsUse, Retrieve, SegmentCritique

_CHITCHAT = re.compile(r"^(hi|hello|hey|chào|xin chào|cảm ơn|thanks|bye|tạm biệt)[\s!.?]*$", re.I)
_WH = re.compile(r"\b(what|which|who|whom|when|where|why|how|là gì|ai|ở đâu|khi nào|tại sao|như thế nào)\b", re.I)


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", s.lower()))


def word_f1(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    if inter == 0:
        return 0.0
    p, r = inter / len(wa), inter / len(wb)
    return 2 * p * r / (p + r)


class EmbeddingCritic:
    def __init__(
        self,
        rel_threshold: float = 0.25,
        sup_full: float = 0.55,
        sup_part: float = 0.30,
        embedding_model=None,
    ):
        self.rel_threshold = rel_threshold
        self.sup_full = sup_full
        self.sup_part = sup_part
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

    # ---- Retrieve ----
    def decide_retrieve(self, query: str) -> Retrieve:
        q = query.strip()
        if not q or _CHITCHAT.match(q):
            return Retrieve.NO
        if len(q.split()) <= 2 and not _WH.search(q):
            return Retrieve.NO
        return Retrieve.YES

    # ---- IsRel ----
    def score_relevance(self, query: str, passage: str) -> tuple[IsRel, float]:
        s = float(self._emb().embed([query])[0] @ self._emb().embed([passage])[0])
        s = (s + 1.0) / 2.0  # cosine [-1,1] -> [0,1] cho dễ đọc threshold
        return (IsRel.RELEVANT if s >= self.rel_threshold else IsRel.IRRELEVANT, s)

    # ---- IsSup + IsUse ----
    def critique(self, query: str, segment: str, passage: str, passage_id: int = -1) -> SegmentCritique:
        isrel, rel_s = self.score_relevance(query, passage)
        emb = self._emb().embed([segment, passage])
        cos = float(emb[0] @ emb[1])
        cos01 = (cos + 1.0) / 2.0
        f1 = word_f1(segment, passage)
        sup_s = max(cos01, f1)
        issup = IsSup.FULLY if sup_s >= self.sup_full else (IsSup.PARTIALLY if sup_s >= self.sup_part else IsSup.NO)

        # Utility: grounded + trả lời đúng trọng tâm query + độ dài hợp lý.
        q_cov = word_f1(segment, query)
        length_bonus = 1.0 if 5 <= len(segment.split()) <= 80 else 0.6
        use_s = 0.5 * sup_s + 0.3 * q_cov + 0.2 * length_bonus
        level = 1 if use_s < 0.25 else (2 if use_s < 0.4 else (3 if use_s < 0.55 else (4 if use_s < 0.7 else 5)))
        return SegmentCritique(
            passage_id=passage_id,
            isrel=isrel,
            isrel_score=rel_s,
            issup=issup,
            issup_score=sup_s,
            isuse=IsUse(level),
            isuse_score=use_s,
        )


class LlmCritic(EmbeddingCritic):
    """Stub cho production: gọi LLM với prompt kiểu paper, fallback heuristic khi lỗi.

    Paper dùng GPT-4 để gán Retrieve/IsRel/IsSup/IsUse rồi distill vào Critic.
    Truyền `call_llm(prompt) -> str` của bạn vào để dùng.
    """

    def __init__(self, call_llm=None, **kwargs):
        super().__init__(**kwargs)
        self.call_llm = call_llm
