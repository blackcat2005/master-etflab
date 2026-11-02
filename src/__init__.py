"""etflab — chunking + retrieval cho báo cáo tài chính ETF Việt Nam.

Pipeline hiện tại dùng ``unstructured`` (hi_res + Table Transformer) làm
parser; mã tách bảng bằng hình học cũ nằm ở ``archive/geometric_parser/`` và
không còn được export ở đây.
"""

from .chunk import Chunk
from .chunking_methods import (
    chipper_chunks,
    dual_granularity_chunks,
    row_json_chunks,
)
from .docmeta import DocMeta, parse_doc_meta
from .retrieval import DocFilter, infer_doc_filter
from .uelements import UElement, UTable
from .vnnum import Num, normalize_text, parse_number, strip_accents

__all__ = [
    "Chunk",
    "DocFilter",
    "DocMeta",
    "Num",
    "UElement",
    "UTable",
    "chipper_chunks",
    "dual_granularity_chunks",
    "infer_doc_filter",
    "normalize_text",
    "parse_doc_meta",
    "parse_number",
    "row_json_chunks",
    "strip_accents",
]
