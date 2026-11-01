"""Bản lưu trữ của parser bảng bằng HÌNH HỌC (PyMuPDF + rule).

Mã này từng là pipeline chính của ``etflab`` (Phase 0/1): dò bảng không viền
bằng toạ độ chữ, dựng cột theo "thung lũng" x-coordinate, ghép header nhiều
tầng. Nó **không còn nằm trong pipeline đánh giá** vì không cover hết layout
và không tổng quát hoá được; xem ``archive/geometric_parser/README.md``.

Gói này tự chứa (copy ``vnnum``/``docmeta``) để có thể chạy lại kết quả cũ
trong môi trường có PyMuPDF, mà không ảnh hưởng package ``etflab`` hiện tại.
"""

from .chunking import (
    Chunk,
    ChunkConfig,
    build_chunks,
    header_paths,
    row_chunks,
    table_card,
)
from .docmeta import DocMeta, parse_doc_meta
from .geometry import (
    Line,
    Word,
    build_lines,
    column_valleys,
    group_cells,
    words_from_page,
)
from .pipeline import DocumentResult, IngestConfig, process_pdf
from .tables import Table, TableConfig, TableExtractor, TickerExtractor
from .vnnum import Num, normalize_text, parse_number, strip_accents

__all__ = [
    "Chunk",
    "ChunkConfig",
    "DocMeta",
    "DocumentResult",
    "IngestConfig",
    "Line",
    "Num",
    "Table",
    "TableConfig",
    "TableExtractor",
    "TickerExtractor",
    "Word",
    "build_chunks",
    "build_lines",
    "column_valleys",
    "group_cells",
    "header_paths",
    "normalize_text",
    "parse_doc_meta",
    "parse_number",
    "process_pdf",
    "row_chunks",
    "strip_accents",
    "table_card",
    "words_from_page",
]
