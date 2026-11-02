"""RAFT answer format: Chain-of-Thought + verbatim citation.

Paper format (blog/paper):
  ## Reasoning
  ... explain which document helps ...
  ##begin_quote##
  <exact span copied from the oracle document>
  ##end_quote##
  ## Answer
  <final answer>

Quy tắc quan trọng: quote phải là substring verbatim của context,
đây chính là cơ chế ép model "nhìn" tài liệu thay vì hallucinate.
"""

from __future__ import annotations

import re

BEGIN = "##begin_quote##"
END = "##end_quote##"


def format_raft_answer(reasoning: str, quote: str, answer: str) -> str:
    return (
        f"## Reasoning\n{reasoning.strip()}\n"
        f"{BEGIN}\n{quote.strip()}\n{END}\n"
        f"## Answer\n{answer.strip()}"
    )


def extract_quotes(text: str) -> list[str]:
    return [m.strip() for m in re.findall(rf"{re.escape(BEGIN)}(.*?){re.escape(END)}", text, re.S)]


def verify_quotes(answer: str, context_docs: list[str]) -> bool:
    """Mọi quote phải xuất hiện verbatim trong ít nhất 1 context doc."""
    quotes = extract_quotes(answer)
    if not quotes:
        return False
    blob = "\n".join(context_docs)
    return all(q and q in blob for q in quotes)
