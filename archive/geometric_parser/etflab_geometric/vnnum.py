"""Chuan hoa so lieu theo quy uoc so tieng Viet Nam.

Quy uoc tai chinh Viet Nam: dau cham (.) la dau phan tach nghin, dau phay (,) la
dau thap phan. Day la nguon sai so 1000 lan pho bien nhat khi parse PDF bao cao
tai chinh, vi `float("2.976,9")` -> ValueError con `float("2976.9")` moi dung.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["Num", "is_number", "normalize_text", "parse_number", "strip_accents"]

_DATE_PATTERNS = (
    re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"),
    re.compile(r"^\d{1,2}[/-]\d{4}$"),
    re.compile(r"^[Qq]\d(?:[/-]\d{1,2})?$"),
    re.compile(r"^\d{1,2}/\d{1,2}$"),
    re.compile(r"^H\d(?:[/-]\d{1,2})?$"),
)

_UNIT_PATTERNS = (
    (re.compile(r"^\((?P<scale>nghìn|tr|tỷ|ngàn|triệu)\s+(cổ phiếu|đồng|USD|VND|cp)\)$", re.IGNORECASE), None),
)

_SCALE_WORDS = {
    "nghìn": 1e3,
    "ngàn": 1e3,
    "tr": 1e3,
    "tỷ": 1e9,
    "triệu": 1e6,
}

_TRAILING_JUNK = ".,;:!?)]}»”’"

_LEADING_SIGNS = "+-−–—~≈<>^*"

# Nhan dien chuoi CHI chua ky tu so (truoc khi tach dau % va dau ngoac).
_NUMERIC_SHAPE = re.compile(r"^[+\-−–—~≈<>^*]?\(?\d[\d.,]*\)?%?$")


@dataclass(frozen=True, slots=True)
class Num:
    value: float
    raw: str
    unit: str | None = None
    approx: bool = False
    negative_style: str | None = None
    scale_word: str | None = None

    @property
    def is_percent(self) -> bool:
        return self.unit == "percent"


def strip_accents(text: str) -> str:
    """Bo dau tieng Viet, ha thuong con nguyen dau."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)


def normalize_text(text: str) -> str:
    """Chuan hoa khoang trang: gom run whitespace, bo khoang trang thua."""
    return re.sub(r"\s+", " ", text).strip()


# Tu don vi phai duoc khop tren dang da bo dau, con gia tri tra ve giu dau.
_SCALE_LOOKUP: dict[str, str] = {
    strip_accents(word).lower(): word for word in _SCALE_WORDS
}


def _is_date(token: str) -> bool:
    return any(p.match(token) for p in _DATE_PATTERNS)


def parse_number(token: str, *, dot_as_thousands: bool = True) -> Num | None:
    """Parse mot token thanh so kieu Viet Nam. Tra None neu khong phai so.

    Xu ly duoc: ``2.976,9`` -> 2976.9; ``12.930.183`` -> 12930183;
    ``(69.559)`` -> -69559; ``0,87%`` -> 0.87 (unit=percent);
    ``-0,48%`` -> -0.48; ``(~1,8`` -> -1.8 (approx=True);
    ``30/06/2026`` -> None (ngay thang).

    ``scale_word`` chi ghi nhan, KHONG nhan he so: don vi da nam o header
    (``(Nghìn cổ phiếu)`` -> ``Table.unit_notes``), nhan o day se tinh
    dop 1000 lan.
    """
    raw = token.strip()
    if not raw:
        return None

    work = raw
    sign = 1.0
    while work and work[0] in _LEADING_SIGNS:
        if work[0] == "+":
            pass
        elif work[0] in "-−–—":
            sign = -1.0
        work = work[1:]
    while work and work[-1] in _TRAILING_JUNK:
        work = work[:-1]
    if not work:
        return None

    negative_style: str | None = "sign" if sign < 0 else None
    paren_open = work.startswith("(")
    paren_close = work.endswith(")")
    if paren_open and paren_close:
        negative_style = "paren"
        work = work[1:-1]
    elif paren_open:
        negative_style = "paren"
        work = work[1:]

    unit: str | None = None
    if work.endswith("%"):
        unit = "percent"
        work = work[:-1]
    work = work.strip()

    scale_word: str | None = None
    norm = strip_accents(work).lower()
    # Sap xep tu dai den ngan: "trieu" phai khop "triệu" chu khong phai "tr",
    # neu khong thi he so sai 1000 lan.
    for scale in sorted(_SCALE_LOOKUP, key=len, reverse=True):
        if norm.startswith(scale):
            scale_word = _SCALE_LOOKUP[scale]
            work = work[len(scale):].strip()
            break
        if norm.endswith(" " + scale):
            scale_word = _SCALE_LOOKUP[scale]
            work = work[: -len(scale)].strip()
            break

    if not work or not _NUMERIC_SHAPE.match(work):
        return None
    if not any(ch.isdigit() for ch in work):
        return None
    if _is_date(work):
        return None

    cleaned = work.replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"[^0-9.,-]", "", cleaned)

    if "," in cleaned:
        integer_part, _, fraction_part = cleaned.rpartition(",")
        if dot_as_thousands:
            integer_part = integer_part.replace(".", "")
        digits_fraction = re.sub(r"\D", "", fraction_part)
        if len(digits_fraction) == 3 and "." in fraction_part:
            integer_part += digits_fraction
            digits_fraction = ""
        cleaned = f"{integer_part}.{digits_fraction}" if digits_fraction else integer_part
    elif dot_as_thousands:
        cleaned = cleaned.replace(".", "")
    elif cleaned.count(".") > 1:
        head, _, tail = cleaned.rpartition(".")
        cleaned = f"{head.replace('.', '')}.{tail}"

    if not cleaned or cleaned in {"-", ".", "-."}:
        return None

    try:
        value = float(cleaned)
    except ValueError:
        return None

    if negative_style == "paren" or sign < 0:
        value = -abs(value)

    approx = any(ch in raw for ch in "~≈<>^*")

    return Num(
        value=value,
        raw=raw,
        unit=unit,
        approx=approx,
        negative_style=negative_style,
        scale_word=scale_word,
    )


def is_number(token: str) -> bool:
    return parse_number(token) is not None
