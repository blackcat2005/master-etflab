"""Metrics cho RAFT: citation fidelity + answer accuracy (offline, không cần LLM judge)."""

from __future__ import annotations

from .reasoning import extract_quotes, verify_quotes


def citation_fidelity(pred_answer: str, context_docs: list[str]) -> float:
    """1.0 nếu mọi quote verbatim trong context, else 0.0 (fallback không quote = 0.0)."""
    return 1.0 if verify_quotes(pred_answer, context_docs) else 0.0


def answer_contains(pred_answer: str, gold: str) -> float:
    return 1.0 if gold.lower().strip() in pred_answer.lower() else 0.0


def evaluate_dataset(items: list[dict]) -> dict:
    """items: [{pred_answer, context_docs, gold}] -> mean metrics."""
    if not items:
        return {"n": 0, "citation_fidelity": 0.0, "answer_accuracy": 0.0}
    cf = sum(citation_fidelity(i["pred_answer"], i["context_docs"]) for i in items) / len(items)
    acc = sum(answer_contains(i["pred_answer"], i["gold"]) for i in items) / len(items)
    return {"n": len(items), "citation_fidelity": round(cf, 3), "answer_accuracy": round(acc, 3)}
