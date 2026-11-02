"""Reflection-token schema (paper Section 3.2).

Paper trains the LM to emit 4 token groups interleaved with text:
  Retrieve: Yes / No / Continue  — có cần retrieve cho segment tiếp theo?
  IsRel:    Relevant / Irrelevant — passage có liên quan query?
  IsSup:    Fully / Partially / No support — câu trả lời có được passage chống lưng?
  IsUse:    1..5 — độ hữu ích tổng thể của segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Retrieve(str, Enum):
    YES = "Yes"
    NO = "No"
    CONTINUE = "Continue"  # dùng tiếp passage trước đó, không retrieve mới


class IsRel(str, Enum):
    RELEVANT = "Relevant"
    IRRELEVANT = "Irrelevant"


class IsSup(str, Enum):
    FULLY = "Fully supported"
    PARTIALLY = "Partially supported"
    NO = "No support"


class IsUse(int, Enum):
    USELESS = 1
    WEAK = 2
    OK = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass
class SegmentCritique:
    """Kết quả critique cho 1 cặp (segment, passage)."""

    passage_id: int
    isrel: IsRel
    isrel_score: float
    issup: IsSup
    issup_score: float
    isuse: IsUse
    isuse_score: float

    @property
    def total(self) -> float:
        # Trọng số theo paper: relevance + groundedness + utility.
        return 1.0 * self.isrel_score + 1.0 * self.issup_score + 0.5 * self.isuse_score
