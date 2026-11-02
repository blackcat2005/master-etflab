"""Self-RAG inference pipeline (paper Section 3.1 + 3.3, bản offline).

Luồng:
  1. decide Retrieve (Yes/No). No -> trả lời parametric trực tiếp.
  2. Yes -> retrieve top-K, lọc IsRel.
  3. Với mỗi passage còn lại: draft 1 candidate segment (song song).
  4. Critique từng candidate (IsSup/IsUse), rank theo SegmentCritique.total,
     áp hard constraint (bỏ No-support) hoặc soft re-rank.
  5. Lặp tối đa max_segments, mỗi segment dùng passage chưa dùng (long-form).
  6. Ghép answer + citations [i].
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .critic import EmbeddingCritic
from .generator import BaseGenerator, ExtractiveGenerator
from .reflection import IsSup, Retrieve, SegmentCritique
from .retriever import InMemoryRetriever


@dataclass
class CitedSegment:
    text: str
    passage_id: int
    citation: str
    critique: SegmentCritique


@dataclass
class SelfRAGResult:
    answer: str
    retrieve: Retrieve
    segments: list[CitedSegment] = field(default_factory=list)
    skipped_irrelevant: int = 0

    def trace(self) -> str:
        lines = [f"Retrieve={self.retrieve.value}", f"segments={len(self.segments)}"]
        for s in self.segments:
            c = s.critique
            lines.append(
                f"  {s.citation} p{c.passage_id} "
                f"rel={c.isrel.value}({c.isrel_score:.2f}) "
                f"sup={c.issup.value}({c.issup_score:.2f}) "
                f"use={c.isuse.value} total={c.total:.2f} :: {s.text[:100]}"
            )
        return "\n".join(lines)


class SelfRAGPipeline:
    def __init__(
        self,
        retriever: InMemoryRetriever | None = None,
        critic: EmbeddingCritic | None = None,
        generator: BaseGenerator | None = None,
        parametric_answer: str = "I don't have enough evidence in the corpus to answer.",
    ):
        self.retriever = retriever or InMemoryRetriever()
        self.critic = critic or EmbeddingCritic()
        self.generator = generator or ExtractiveGenerator()
        self.parametric_answer = parametric_answer

    # ---- Build ----
    def fit(self, passages: list[str]) -> "SelfRAGPipeline":
        self.retriever.add(passages)
        return self

    # ---- Inference ----
    def answer(
        self,
        query: str,
        top_k: int = 5,
        max_segments: int = 2,
        enforce_supported: bool = True,
    ) -> SelfRAGResult:
        decision = self.critic.decide_retrieve(query)
        if decision == Retrieve.NO:
            return SelfRAGResult(answer=self.parametric_answer, retrieve=decision)

        hits = self.retriever.search(query, top_k=top_k)
        # Filter IsRel (paper: đánh giá relevance từng passage).
        relevant = []
        skipped = 0
        for h in hits:
            label, _ = self.critic.score_relevance(query, h.passage.text)
            if label.value == "Relevant":
                relevant.append(h)
            else:
                skipped += 1
        if not relevant:
            return SelfRAGResult(answer=self.parametric_answer, retrieve=decision, skipped_irrelevant=skipped)

        used: set[int] = set()
        segments: list[CitedSegment] = []
        for _ in range(max(1, max_segments)):
            cands: list[tuple[float, CitedSegment]] = []
            for h in relevant:
                if h.passage.id in used:
                    continue
                draft = self.generator.draft(query, h.passage.text)
                if not draft or draft.strip() == "NOT_SUPPORTED":
                    continue
                crit = self.critic.critique(query, draft, h.passage.text, passage_id=h.passage.id)
                if enforce_supported and crit.issup == IsSup.NO:
                    continue
                cands.append((crit.total, CitedSegment(text=draft, passage_id=h.passage.id, citation="", critique=crit)))
            if not cands:
                break
            cands.sort(key=lambda x: -x[0])
            best = cands[0][1]
            used.add(best.passage_id)
            segments.append(best)

        if not segments:
            return SelfRAGResult(answer=self.parametric_answer, retrieve=decision, skipped_irrelevant=skipped)

        # Đánh số citation theo thứ tự chọn.
        for i, s in enumerate(segments, start=1):
            s.citation = f"[{i}]"
        answer = " ".join(f"{s.text} {s.citation}" for s in segments)
        return SelfRAGResult(answer=answer, retrieve=decision, segments=segments, skipped_irrelevant=skipped)
