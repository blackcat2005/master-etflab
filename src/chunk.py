"""Đơn vị truy hồi dùng chung (``Chunk``) — không phụ thuộc parser.

``Chunk`` được cả ``chunking_methods`` (unstructured) và ``retrieval`` dùng.
Các hàm chunking theo bảng hình học đã được chuyển sang
``archive/geometric_parser/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .docmeta import DocMeta

__all__ = ["Chunk"]


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    kind: str
    text: str
    sparse_text: str
    source_file: str
    page: int
    parent_id: str | None = None
    table_id: str | None = None
    row_index: int | None = None
    label: str = ""
    cells: list[dict[str, Any]] = field(default_factory=list)
    doc: DocMeta = field(default_factory=lambda: DocMeta(source_file=""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "kind": self.kind,
            "text": self.text,
            "sparse_text": self.sparse_text,
            "source_file": self.source_file,
            "page": self.page,
            "parent_id": self.parent_id,
            "table_id": self.table_id,
            "row_index": self.row_index,
            "label": self.label,
            "cells": self.cells,
            "doc": self.doc.to_dict(),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "page": self.page,
            "kind": self.kind,
            "table_id": self.table_id,
            "row_index": self.row_index,
            "parent_id": self.parent_id,
            "doc": self.doc.to_dict(),
        }
