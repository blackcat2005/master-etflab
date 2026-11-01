"""RAFT training-data builder: Q + oracle + distractors -> CoT answer.

Công thức paper:
  - P% data:  Q + D* (oracle) + D1..Dk (distractors) -> A* (CoT + citation)
  - (1-P)%:   Q + D1..Dk (chỉ distractors, không oracle) -> A* (ép memorize)
Distractors = mix hard negatives (giống query nhưng không chứa đáp án)
              + random docs, mô phỏng retriever nhiễu thực tế.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from .reasoning import format_raft_answer


@dataclass
class QAItem:
    question: str
    answer: str  # gold answer ngắn
    oracle_doc: str  # doc chứa đáp án


@dataclass
class RAFTExample:
    question: str
    context_docs: list[str]
    has_oracle: bool
    answer: str  # CoT + citation (chuẩn training target)
    oracle_doc: str = ""


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", s.lower()))


def _jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def pick_distractors(
    question: str,
    oracle_doc: str,
    corpus: list[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Mix hard negatives (top-similar, không phải oracle) + random."""
    cands = [d for d in corpus if d != oracle_doc]
    if not cands:
        return []
    ranked = sorted(cands, key=lambda d: _jaccard(question, d), reverse=True)
    n_hard = min(len(ranked), (n + 1) // 2)
    hard = ranked[:n_hard]
    rest = [d for d in cands if d not in hard]
    rng.shuffle(rest)
    out = hard + rest[: max(0, n - len(hard))]
    rng.shuffle(out)
    return out[:n]


def _make_cot_answer(question: str, oracle_doc: str, gold: str, has_oracle: bool) -> str:
    # Quote = câu trong oracle chứa nhiều từ chung với gold nhất (verbatim).
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", oracle_doc.strip()) if s.strip()]
    quote = max(sents, key=lambda s: len(_words(s) & _words(gold)), default=oracle_doc[:300])
    if has_oracle:
        reasoning = "I examine each document and identify the one containing the answer, ignoring distractors."
    else:
        reasoning = "None of the documents contain the answer, so I rely on memorized domain knowledge."
    return format_raft_answer(reasoning, quote, gold)


def build_raft_examples(
    qa_items: list[QAItem],
    corpus: list[str],
    n_distractors: int = 3,
    p_with_oracle: float = 0.8,
    seed: int = 42,
) -> list[RAFTExample]:
    rng = random.Random(seed)
    examples: list[RAFTExample] = []
    for item in qa_items:
        has_oracle = rng.random() < p_with_oracle
        distractors = pick_distractors(item.question, item.oracle_doc, corpus, n_distractors, rng)
        context = ([item.oracle_doc] + distractors) if has_oracle else distractors
        rng.shuffle(context)
        answer = _make_cot_answer(item.question, item.oracle_doc, item.answer, has_oracle)
        examples.append(
            RAFTExample(
                question=item.question,
                context_docs=context,
                has_oracle=has_oracle,
                answer=answer,
                oracle_doc=item.oracle_doc,
            )
        )
    return examples


def to_jsonl(examples: list[RAFTExample]) -> list[dict]:
    """Format SFT chuẩn: prompt open-book + completion CoT."""
    rows = []
    for ex in examples:
        docs = "\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(ex.context_docs))
        rows.append(
            {
                "prompt": f"Question: {ex.question}\nDocuments:\n{docs}\nAnswer with reasoning and verbatim quotes.",
                "completion": ex.answer,
                "has_oracle": ex.has_oracle,
            }
        )
    return rows
