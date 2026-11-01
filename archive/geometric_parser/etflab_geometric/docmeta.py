"""Phase 1 — metadata cap TAI LIEU: loai, chi so, ky, nam.

Vay do: chunk chi chua caption + du lieu dong. Ten file la noi duy nhat biet
tai lieu nao la ky Q2, nen cau hoi "ty trong moi cua BSR Q2" khong co token
``q2`` nao de so sanh — BM25 chi con khop ``bsr``/``ty trong moi`` va tra ve
ban ky Q3. Day metadata vao ``sparse_text`` de moi chunk mang nhan dinh
dung cua no, va cho phep loc cung the.

Ten file trong corpus SSI co mau rong nhat: ``<so> - <ten>...<chi so>_<ky>_<nam>``
vi du ``Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf``. Khi ten file khong co
ky (ban tin, dieu le) thi fallback sang trang dau.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .vnnum import strip_accents

__all__ = ["DocMeta", "parse_doc_meta"]


def _fold(text: str) -> str:
    """Bo dau de regex khop duoc tieng Viet ("quý", "Kỳ", "Điều")."""
    return strip_accents(str(text)).lower().replace("đ", "d")


# "kỳ Q2/2026", "quý II/2026", "Kỳ Q1 năm 2026" — chay tren text DA BO DAU.
_PERIOD_RES = (
    re.compile(r"\bq\s*([1-4])\s*[/\-.]?\s*(\d{4})\b"),
    re.compile(r"\bquy\s*(i{1,3}v?|iv)\s*[/\-.]?\s*(\d{4})\b"),
    re.compile(r"\bquy\s*([1-4])\s*(?:nam|[/\-.])\s*(\d{4})\b"),
    re.compile(r"\bk[yy]\s*q?\s*([1-4])\s*(?:nam|[/\-.])\s*(\d{4})\b"),
)
# Ten file: "VN30_Q2_2026", "VNFINLEAD_Q1_2026"
_FILE_PERIOD_RE = re.compile(r"(?<![a-z0-9])q([1-4])[_\- ]?(\d{4})(?![0-9])")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_DMY_RE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](20\d{2})\b")
_DATE_PREFIX_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})\b")

# Thu tu la quan trong: kieu do dai truoc vi "VN30" chua "VNDIAMOND".
_INDEX_RES = (
    re.compile(r"vn\s*fin\s*lead"),
    re.compile(r"vn\s*diamond"),
    re.compile(r"\bvn\s*30\b"),
)
_TYPE_RES = (
    ("review", re.compile(r"thay\s*doi\s*thanh\s*phan")),
    ("forecast", re.compile(r"du\s*bao\s*thay\s*doi")),
    ("charter", re.compile(r"dieu\s*le|quy\s*dinh")),
)
_TYPE_LABELS = {
    # Chu "chỉ số" duoc them sau khi noi voi ``index`` — ghi san se ra
    # "chỉ số chỉ số VN30".
    "review": "Bản tin thay đổi thành phần",
    "forecast": "Bản tin dự báo thay đổi danh mục",
    "charter": "Điều lệ quỹ",
}
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 4}


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Nhan dinh cap tai lieu. Moi truong deu co the rong."""

    source_file: str
    doc_type: str = "unknown"
    index: str = ""
    quarter: str = ""  # "Q2"
    year: str = ""  # "2026"
    date: str = ""  # "2026-05-08"

    @property
    def period(self) -> str:
        return f"{self.quarter}/{self.year}" if self.quarter and self.year else ""

    @property
    def type_label(self) -> str:
        return _TYPE_LABELS.get(self.doc_type, "Tài liệu")

    @property
    def label(self) -> str:
        """Nhan doc duoc, dung lam tien to cho ``sparse_text``.

        Giu nguyen dang ``Q2/2026`` de token hoa ra ``q2`` + ``2026`` — kieu
        viet "quý 2" chi co token ``quy``/``2`` va khong khop duoc.
        """
        parts = [self.type_label]
        if self.index:
            parts.append(f"chỉ số {self.index}")
        if self.period:
            parts.append(f"kỳ {self.period}")
        elif self.year:
            parts.append(f"năm {self.year}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "index": self.index,
            "quarter": self.quarter,
            "year": self.year,
            "period": self.period,
            "date": self.date,
            "label": self.label,
        }


def _doc_type(stem: str) -> str:
    for name, rx in _TYPE_RES:
        if rx.search(stem):
            return name
    return "unknown"


def _index_name(flat: str) -> str:
    for rx in _INDEX_RES:
        m = rx.search(flat)
        if m:
            return re.sub(r"\s+", "", m.group()).upper()
    return ""


def _period_from_text(text: str) -> tuple[str, str]:
    flat = _fold(text)
    for rx in _PERIOD_RES:
        m = rx.search(flat)
        if not m:
            continue
        raw_q, year = m.group(1), m.group(2)
        q = _ROMAN.get(raw_q.upper(), None)
        if q is None:
            try:
                q = int(raw_q)
            except ValueError:
                continue
        if 1 <= q <= 4:
            return f"Q{q}", year
    return "", ""


def parse_doc_meta(source_file: str, hint_text: str = "") -> DocMeta:
    """Nhan dinh tai lieu tu ten file, fallback sang ``hint_text`` (trang dau).

    ``hint_text`` chi dung khi ten file khong co ky — nhieu ban tin SSI viet
    "ky Q2/2026" ngay tren trang dau nhung ten file la ngay thang
    ("Du_bao_..._VNDiamond_Q2_2026" co, nhung "BCB VNFIN LEAD TIENG VIET" thi
    khong).
    """
    stem = re.sub(r"\.pdf$", "", str(source_file), flags=re.IGNORECASE)
    flat = re.sub(r"\s+", " ", _fold(stem.replace("_", " ")))

    quarter = year = ""
    m = _FILE_PERIOD_RE.search(flat)
    if m:
        quarter, year = f"Q{m.group(1)}", m.group(2)
    if not quarter and hint_text:
        quarter, year = _period_from_text(hint_text)
    if not year:
        ym = _YEAR_RE.search(flat)
        if ym:
            year = ym.group(1)

    date = ""
    dm = _DMY_RE.search(flat)
    if dm:
        date = f"{dm.group(3)}-{int(dm.group(2)):02d}-{int(dm.group(1)):02d}"
    else:
        pm = _DATE_PREFIX_RE.match(flat.strip())
        if pm:
            date = f"{pm.group(1)}-{pm.group(2)}-{pm.group(3)}"

    return DocMeta(
        source_file=str(source_file),
        doc_type=_doc_type(flat),
        index=_index_name(flat),
        quarter=quarter,
        year=year,
        date=date,
    )