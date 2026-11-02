#!/usr/bin/env python
"""Metric content-based dùng chung cho các script đánh giá.

Tách khỏi ``eval_chunking.py`` (đã chuyển sang archive) để phần đánh giá
``unstructured`` không phụ thuộc parser hình học. Hành vi giữ nguyên:

  * ``tok``     — token hoá đơn giản (bỏ dấu, chữ thường) dùng cho BM25.
  * ``numbers_in`` — trích số theo quy tắc Việt Nam, nhận số âm kế toán
    ``(36,2)`` và bỏ đơn vị ``('000 cổ phiếu)``.
  * ``matches`` — chunk chứa cả entity và value đã chuẩn hoá hay không.
"""

from __future__ import annotations

import re

from src.vnnum import parse_number, strip_accents

__all__ = ["matches", "numbers_in", "strip_accents", "tok"]

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Số Việt Nam: "21.094", "0,87", "-41,6", "3.274.147"; "(36,2)" là số âm.
_NUM_SPAN_RE = re.compile(r"-?\(?\d+(?:[.,]\d+)*\)?%?")

# "('000 cổ phiếu)" / "(Triệu cổ phiếu)" là đơn vị, không phải giá trị 0.
_UNIT_ZERO_RE = re.compile(r"['’]\s*0{3}")


def tok(text: str) -> list[str]:
    """Token hoá BM25: bỏ dấu, chữ thường, tách theo từ."""
    folded = strip_accents(text).lower().replace("đ", "d")
    return _WORD_RE.findall(folded)


def numbers_in(text: str) -> set[float]:
    """Trích mọi số theo quy tắc Việt Nam (dấu chấm=nghìn, phẩy=thập phân)."""
    text = _UNIT_ZERO_RE.sub(" ", text)
    out: set[float] = set()
    for m in _NUM_SPAN_RE.finditer(text):
        token = m.group().rstrip("%")
        for dot_as_thousands in (True, False):
            try:
                num = parse_number(token, dot_as_thousands=dot_as_thousands)
            except (ValueError, OverflowError):
                continue
            if num is not None:
                out.add(round(num.value, 6))
    return out


def matches(chunk, entity: str, value_norm: float | None) -> bool:
    """Chunk đúng nếu chứa cả mã cổ phiếu và giá trị đã chuẩn hoá."""
    text = strip_accents(chunk.text).lower()
    if entity and strip_accents(entity).lower() not in text:
        return False
    if value_norm is None:
        return True
    return round(float(value_norm), 6) in numbers_in(text)
