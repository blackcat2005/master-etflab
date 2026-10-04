"""etflab — ingestion + chunking cho hybrid RAG phan tich quy ETF."""

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
from .retrieval import DocFilter, infer_doc_filter
from .tables import Table, TableConfig, TableExtractor, TickerExtractor
from .vnnum import Num, normalize_text, parse_number, strip_accents

__all__ = [
    "Chunk",
    "ChunkConfig",
    "DocFilter",
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
    "infer_doc_filter",
    "normalize_text",
    "parse_doc_meta",
    "parse_number",
    "process_pdf",
    "row_chunks",
    "strip_accents",
    "table_card",
    "words_from_page",
]
