"""Phase 1 — bien JSON thanh bang thanh don vi truy hoi (chunking).

Thiet ke theo "RAG Bang Bieu Tai Chinh.md": moi dong du lieu la mot chunk
rieng (row-wise) de tra loi "gia tri cua BSR la bao nhieu", con ca bang la
chunk cha (parent) de mo ta nganh chung va lam context khi sinh cau tra loi.

Moi chunk co hai ban text:
  * ``text``       — ban day du, dung cho embedding: chua header path day du,
                     so da chuan hoa, don vi.
  * ``sparse_text`` — ban rut gon, dung cho BM25: nhan cot gon + gia tri tho.

Vi do vector search va keyword search nhin vao cung mot chunk nhung hai
dien giai khac nhau — thuoc ve "dual granularity".
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .docmeta import DocMeta, parse_doc_meta
from .tables import Table
from .vnnum import Num, is_number, normalize_text, parse_number

__all__ = [
    "Chunk",
    "ChunkConfig",
    "build_chunks",
    "header_paths",
    "prose_chunks",
    "row_chunks",
    "table_card",
]

_UNIT_TAIL_RE = re.compile(r"\s*\(([^()]*)\)\s*$")
_LABEL_CLEAN_RE = re.compile(r"\s{2,}")


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    max_prose_chars: int = 900
    overlap_chars: int = 120
    min_row_values: int = 1
    include_table_card: bool = True
    label_max_cols: int = 2


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


def header_paths(table: Table) -> list[str]:
    """Nhan cot de doc, lay tu ``columns`` da flatten san.

    ``columns`` da duoc ``_names_from_levels`` noi nhieu tang roi nen gon va
    dung; ``header_levels`` chi la du phong cho bang khong co ten cot.
    """
    out = [_clean_label(c) for c in table.columns]
    if len(out) == table.n_cols and out:
        return out
    if not table.header_levels:
        return out or [f"Cot {i}" for i in range(table.n_cols)]
    rebuilt: list[str] = []
    for col in range(table.n_cols):
        parts: list[str] = []
        for level in table.header_levels:
            if col >= len(level):
                continue
            piece = (level[col] or "").strip()
            if piece and piece not in parts:
                parts.append(piece)
        rebuilt.append(_clean_label(" ".join(parts) if parts else f"Cot {col}"))
    return rebuilt


def short_caption(table: Table, max_chars: int = 110) -> str:
    """Caption ngan cho chunk dong — tranh lap lai cau van ban dai o moi dong."""
    text = re.sub(r"^\s*(Bảng|Table)\s*\d*\s*[:.\-–]?\s*", "", table.caption or "")
    text = normalize_text(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    dot = head.rfind(". ")
    if dot > max_chars // 2:
        return head[: dot + 1].strip()
    return head[: head.rfind(" ")].rstrip(",;:") + "…"


def _clean_label(label: str) -> str:
    """Bo don vi tail va chuan hoa khoang trang; 'col_3' -> 'Cot 3'."""
    label = _LABEL_CLEAN_RE.sub(" ", label).strip()
    tail = _UNIT_TAIL_RE.search(label)
    if tail:
        label = label[: tail.start()].strip()
    m = re.fullmatch(r"col_(\d+)", label)
    if m:
        label = f"Cot {m.group(1)}"
    return label or "Cot"


def _display_units(notes: Sequence[str]) -> list[str]:
    """Hien thi don vi cho nguoi doc; 'percent' la token noi bo, ra '%'."""
    out: list[str] = []
    for note in notes:
        text = "%" if note == "percent" else note
        if text not in out:
            out.append(text)
    return out


def _cell_unit(num: Num | None) -> str:
    """Don vi do CHINH gia tri mang theo.

    Khong gan ``table.unit_notes`` vao tung o: don vi cap bang thuong chi ap
    dung cho mot nhom cot (VD "Nghin co phieu"), con cot khac nhu "So phien GD"
    la don vi khac — gan vao se sinh cau tra loi sai.
    """
    if num is None:
        return ""
    return "%" if num.unit == "percent" else ""


def _cell_value(raw: str | None) -> Num | None:
    if raw is None:
        return None
    token = raw.strip()
    if not token or not is_number(token):
        return None
    return parse_number(token)


def row_chunks(table: Table, cfg: ChunkConfig | None = None) -> list[Chunk]:
    """Moi dong du lieu -> mot chunk, giu null (o rong) de khong bi nham sai."""
    cfg = cfg or ChunkConfig()
    labels = header_paths(table)
    label_cols = _label_columns(table, cfg)
    caption = short_caption(table)
    prefix = f"{caption} | " if caption else ""
    out: list[Chunk] = []
    for idx, row in enumerate(table.rows):
        cells: list[dict[str, Any]] = []
        dense: list[str] = []
        sparse: list[str] = []
        n_values = 0
        for col in range(table.n_cols):
            raw = row.get(str(col))
            label = labels[col] if col < len(labels) else f"Cot {col}"
            if raw is None or str(raw).strip() == "":
                cells.append({"col": col, "label": label, "raw": None,
                              "value": None, "unit": "", "missing": True})
                continue
            value = str(raw).strip()
            num = _cell_value(value)
            if num is not None:
                if col != table.row_index_col:
                    n_values += 1
                unit = _cell_unit(num)
                shown = value if unit and unit in value else f"{value} {unit}".strip()
                dense.append(f"{label}: {shown}")
                sparse.append(f"{label} {value}")
                cells.append({"col": col, "label": label, "raw": value,
                              "value": num.value, "unit": unit,
                              "missing": False})
            else:
                dense.append(f"{label}: {value}")
                sparse.append(f"{label} {value}")
                cells.append({"col": col, "label": label, "raw": value,
                              "value": None, "unit": "", "missing": False})
        if n_values < cfg.min_row_values:
            continue
        label = _row_label(cells, label_cols)
        detail_sparse = [s for i, s in enumerate(sparse) if i not in set(label_cols)]
        units = _display_units(table.unit_notes)
        unit_note = f" | Don vi: {'; '.join(units)}" if units else ""
        out.append(
            Chunk(
                chunk_id=f"{table.table_id}#r{idx}",
                kind="table_row",
                text=f"{prefix}{'; '.join(dense)}{unit_note}",
                sparse_text=f"{label} {' | '.join(detail_sparse)}".strip(),
                source_file=table.source_file,
                page=table.page,
                parent_id=table.table_id,
                table_id=table.table_id,
                row_index=idx,
                label=label,
                cells=cells,
            )
        )
    return out


def _label_columns(table: Table, cfg: ChunkConfig) -> list[int]:
    """Cac cot dinh danh o dau bang.

    KHONG duoc lay "cot dau tien" theo vi tri: nhieu bang khong co cot STT, ma
    bat dau bang cot 'Ma CK' roi thang den cot so luong. Neu dem cung ma cot so
    dau tien se bi loai khoi ban rut gon va gia tri mat khoi chi muc BM25.
    Do la cot chu thi moi la nhan dong.
    """
    limit = min(cfg.label_max_cols + 2, table.n_cols)
    cols: list[int] = []
    for i in range(limit):
        if i == table.row_index_col:
            continue
        present = [r.get(str(i)) for r in table.rows]
        present = [v for v in present if v is not None and str(v).strip()]
        if not present:
            cols.append(i)
            continue
        numeric = sum(1 for v in present if _cell_value(str(v)) is not None)
        if numeric * 2 >= len(present):
            break
        cols.append(i)
    return cols


def _row_label(cells: list[dict[str, Any]], label_cols: Sequence[int]) -> str:
    """Nhan dong: uu tien cot chu (ma CK, ten) o dau bang."""
    lookup = {c["col"]: c for c in cells}
    for col in label_cols:
        cell = lookup.get(col)
        if cell and cell["raw"]:
            return f"{cell['label']} = {cell['raw']}"
    return ""


def table_card(table: Table) -> Chunk:
    """Chunk cha: mo ta bang, dung lam context va cho cau hoi o muc tong the."""
    labels = header_paths(table)
    units = ", ".join(_display_units(table.unit_notes))
    head = (table.caption or "").rstrip(" .:;,–-") or (
        f"Bang {table.n_rows} dong x {table.n_cols} cot"
    )
    text = (
        f"{head}. Cot: {'; '.join(labels)}. "
        f"So dong: {table.n_rows}. So cot: {table.n_cols}."
        + (f" Don vi: {units}." if units else "")
    )
    return Chunk(
        chunk_id=f"{table.table_id}#card",
        kind="table_card",
        text=text,
        sparse_text=f"{head} {' '.join(labels)}",
        source_file=table.source_file,
        page=table.page,
        parent_id=None,
        table_id=table.table_id,
        label=head,
    )


def prose_chunks(
    pages: Iterable[Any], source_file: str, cfg: ChunkConfig | None = None
) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    out: list[Chunk] = []
    for page in pages:
        for bi, block in enumerate(page.blocks):
            text = (block.get("text") or "").strip()
            if not text:
                continue
            for si, piece in enumerate(_split_text(text, cfg)):
                out.append(
                    Chunk(
                        chunk_id=f"{source_file}#{page.page}p{bi}s{si}",
                        kind="prose",
                        text=piece,
                        sparse_text=piece,
                        source_file=source_file,
                        page=page.page,
                    )
                )
    return out


def _split_text(text: str, cfg: ChunkConfig) -> list[str]:
    if len(text) <= cfg.max_prose_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    buf = ""
    for sentence in sentences:
        if buf and len(buf) + 1 + len(sentence) > cfg.max_prose_chars:
            pieces.append(buf)
            tail = buf[-cfg.overlap_chars :]
            buf = f"{tail} {sentence}".strip()
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def build_chunks(result: Any, cfg: ChunkConfig | None = None) -> list[Chunk]:
    """Bien DocumentResult thanh danh sach chunk co parent-child.

    Moi chunk nhan ``doc`` (loai/chi so/ky) va tien to ``sparse_text`` bang nhan
    do. Ly do: cau hoi "ty trong moi cua BSR **Q2**" phai phan biet duoc ban
    Q2 voi Q3. Ten file la noi duy nhat biet ky, nen neu khong dua vao chunk
    thi BM25 chi con khop ``bsr``/``ty trong moi`` va tra ve nham ky khac.
    """
    cfg = cfg or ChunkConfig()
    doc = _doc_meta_for(result)
    chunks: list[Chunk] = []
    for table in result.tables:
        if cfg.include_table_card:
            chunks.append(table_card(table))
        chunks.extend(row_chunks(table, cfg))
    chunks.extend(prose_chunks(result.pages_text, result.source_file, cfg))

    prefix = f"{doc.label} " if doc.label else ""
    for chunk in chunks:
        chunk.doc = doc
        if prefix:
            chunk.sparse_text = prefix + chunk.sparse_text
    return chunks


def _doc_meta_for(result: Any) -> DocMeta:
    """Doc meta tu ten file; trang dau chi dung lam du phong khi ten file im lang."""
    hint = ""
    for page in getattr(result, "pages_text", []) or []:
        hint = " ".join(
            str(block.get("text") or "") for block in getattr(page, "blocks", [])
        )
        if hint.strip():
            break
    return parse_doc_meta(result.source_file, hint[:4000])