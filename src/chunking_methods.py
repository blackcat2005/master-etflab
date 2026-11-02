"""Ba cach phan manh tren CUNG mot dau vao ``unstructured``.

Vi sao tach khoi ``chunking.py``: ``chunking.py`` gan chat voi ``Table`` do
``src`` tu tach (co ``unit_notes``, ``header_levels``, ``row_index_col``).
Ban thi nghiem nay co y khong dung parser hinh hoc, nen dau vao la
``UElement`` — chuoi phan tu theo thu tu doc do ``unstructured`` tra ve.

Ba chien luoc:

* ``chipper``  — dung 4 quy tac cua paper arXiv:2402.05131: gop phan tu cho
  du 2.048 ky tu, khong cat vo phan tu, ``Title`` mo chunk moi, ``Table``
  giu nguyen va mo chunk rieng. Them bien the ``chipper_prefix`` dung
  metadata "Prefix & Table Description" (2 cau dau + caption bang) — bien
  the tot nhat trong so cac bien the tat dinh cua paper.
* ``dual``     — dual-granularity (Peng et al.): khoi LON theo doan (gop
  toi gamma=1.024 ky tu, Title/Table la ranh gioi), khoi NHO la cau (gop
  hai cau lien tiep neu cosine >= theta=0.9) hoac tung dong bang. Truy hoi
  tren khoi nho, xep hang lai bang trung binh diem nho + diem khoi lon.
* ``row_json`` — moi dong bang thanh mot JSON tu chua (file, trang, caption,
  {ten cot: gia tri}); khong can embed ca bang.

Cam ket chung: khong dung toa do, khong dung parser hinh hoc; chi tieu thu
chuoi phan tu ma ``unstructured`` da tra ve.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .chunk import Chunk
from .docmeta import DocMeta, parse_doc_meta
from .uelements import UElement, UTable, skip_kind
from .vnnum import parse_number

__all__ = [
    "MethodConfig",
    "chipper_chunks",
    "dual_granularity_chunks",
    "row_json_chunks",
    "table_row_json",
]


@dataclass(frozen=True, slots=True)
class MethodConfig:
    chipper_target: int = 2048      # paper: 2.048 ky tu
    large_target: int = 1024        # paper: gamma = 1024
    small_theta: float = 0.9        # paper: theta = 0.9
    min_row_values: int = 1
    include_doc_label: bool = True


_Embed = Callable[[Sequence[str]], list[list[float]]]


# ---------------------------------------------------------------------------
# Tien ich chung.
# ---------------------------------------------------------------------------
def _doc_meta(doc: dict) -> DocMeta:
    hint = " ".join(
        el.text for el in doc.get("elements", [])[:20] if el.kind != "table"
    )[:4000]
    return parse_doc_meta(doc["source_file"], hint)


def _label_prefix(meta: DocMeta, cfg: MethodConfig) -> str:
    return f"{meta.label} " if cfg.include_doc_label and meta.label else ""


def _new_chunk(doc: dict, meta: DocMeta, cfg: MethodConfig, *, kind: str,
               text: str, page: int, chunk_id: str,
               parent_id: str | None = None,
               table_id: str | None = None,
               row_index: int | None = None,
               label: str = "") -> Chunk:
    prefix = _label_prefix(meta, cfg)
    return Chunk(
        chunk_id=chunk_id,
        kind=kind,
        text=text,
        sparse_text=(prefix + text).strip(),
        source_file=doc["source_file"],
        page=page,
        parent_id=parent_id,
        table_id=table_id,
        row_index=row_index,
        label=label,
        doc=meta,
    )


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text)]
    return [p for p in parts if p]


def _first_sentences(text: str, n: int = 2) -> str:
    sents = _sentences(text)
    return " ".join(sents[:n]) if sents else text.strip()


def _has_number(cells: Sequence[str]) -> bool:
    for cell in cells:
        token = str(cell).strip()
        if token and parse_number(token) is not None:
            return True
    return False


def table_row_json(table: UTable, row: Sequence[str], meta: DocMeta) -> str:
    """Mot dong -> JSON tu chua; khoa la ten cot, gia tri giu nguyen dang."""
    payload = {
        "file": meta.label or table.table_id,
        "page": table.page,
        "table": table.caption or table.table_id,
        "row": table.row_dict(list(row)),
    }
    return json.dumps(payload, ensure_ascii=False)


def _iter_tables(elements: Sequence[UElement]):
    for el in elements:
        if el.is_table and el.table is not None and el.table.rows:
            yield el.table


# ---------------------------------------------------------------------------
# 1. Chipper — 4 quy tac cua paper.
# ---------------------------------------------------------------------------
def chipper_chunks(doc: dict, elements: Sequence[UElement],
                   cfg: MethodConfig | None = None, *,
                   add_prefix: bool = False) -> list[Chunk]:
    """Chunk theo phan tu, dung nguyen van 4 quy tac cua paper.

    ``add_prefix`` bat bien the metadata "Prefix & Table Description":
    tien to 2 cau dau (van ban) hoac caption (bang). Khong dung LLM.
    """
    cfg = cfg or MethodConfig()
    meta = _doc_meta(doc)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    buf_page = 1

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf, buf_len = [], 0
        if not text:
            return
        cid = f"{doc['source_file']}#chipper_{len(chunks)}"
        body = f"{_first_sentences(text)}\n{text}" if add_prefix else text
        chunks.append(_new_chunk(doc, meta, cfg, kind="chipper",
                                 text=body, page=buf_page, chunk_id=cid))

    for el in elements:
        if skip_kind(el.kind):
            continue
        if el.kind == "table" and el.table is not None:
            # Quy tac 4: Table -> mo chunk moi, giu nguyen ca bang.
            flush()
            text = el.table.to_text()
            if add_prefix:
                caption = el.table.caption or f"Bảng trang {el.page}"
                text = f"{caption}\n{text}"
            chunks.append(_new_chunk(
                doc, meta, cfg, kind="chipper_table", text=text,
                page=el.page, chunk_id=f"{doc['source_file']}#chipperT_{len(chunks)}",
                table_id=el.table.table_id,
            ))
            continue

        # Quy tac 3: Title -> bat dau chunk moi (tieu de nam trong chunk moi).
        if el.kind == "title":
            flush()
            buf = [el.text]
            buf_len = len(el.text)
            buf_page = el.page
            continue

        # Quy tac 1+2: gop toi khi dat 2.048 ky tu, khong bao gio cat phan tu.
        if buf and buf_len + len(el.text) > cfg.chipper_target:
            flush()
        if not buf:
            buf_page = el.page
        buf.append(el.text)
        buf_len += len(el.text)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# 2. Dual granularity — khoi lon (doan) + khoi nho (cau/dong).
# ---------------------------------------------------------------------------
def dual_granularity_chunks(
    doc: dict,
    elements: Sequence[UElement],
    cfg: MethodConfig | None = None,
    embed: _Embed | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Tra ve ``(small_chunks, large_chunks)``.

    ``large_chunks`` la ngu canh dua cho buoc sinh (small2big); ``parent_id``
    cua tung small tro ve large tuong ung. ``embed`` la ham embed cau
    (da chuan hoa L2) de gop cau theo cosine >= theta; neu ``None`` thi
    khong gop (moi cau la mot khoi nho) — van giu nguyen cau truc 2 tang.
    """
    cfg = cfg or MethodConfig()
    meta = _doc_meta(doc)

    # --- Tang 1: khoi lon -------------------------------------------------
    large_specs: list[dict] = []
    buf: list[str] = []
    buf_len = 0
    buf_page = 1
    cur_title = ""

    def flush_large() -> None:
        nonlocal buf, buf_len, cur_title
        text = "\n".join(buf).strip()
        title = cur_title
        buf, buf_len, cur_title = [], 0, ""
        if text:
            large_specs.append({"kind": "prose", "text": text, "page": buf_page,
                                "title": title, "table": None})

    for el in elements:
        if skip_kind(el.kind):
            continue
        if el.kind == "title":
            flush_large()
            buf = [el.text]
            buf_len = len(el.text)
            buf_page = el.page
            cur_title = el.text
            continue
        if el.kind == "table" and el.table is not None:
            flush_large()
            large_specs.append({"kind": "table", "text": el.table.to_text(),
                                "page": el.page, "title": el.table.caption,
                                "table": el.table})
            continue
        if buf and buf_len + len(el.text) > cfg.large_target:
            flush_large()
            buf_page = el.page
        if not buf:
            buf_page = el.page
        buf.append(el.text)
        buf_len += len(el.text)
    flush_large()

    large_chunks: list[Chunk] = []
    small_chunks: list[Chunk] = []
    for li, spec in enumerate(large_specs):
        lid = f"{doc['source_file']}#dualL_{li}"
        large_chunks.append(_new_chunk(
            doc, meta, cfg, kind="dual_large", text=spec["text"],
            page=spec["page"], chunk_id=lid,
            label=spec.get("title") or "",
        ))
        if spec["kind"] == "table" and spec["table"] is not None:
            table: UTable = spec["table"]
            for ri, row in enumerate(table.rows):
                if not _has_number(row):
                    continue
                small_chunks.append(_new_chunk(
                    doc, meta, cfg, kind="dual_row",
                    text=table_row_json(table, row, meta),
                    page=table.page, chunk_id=f"{lid}#s{ri}",
                    parent_id=lid, table_id=table.table_id, row_index=ri,
                    label=table.row_dict(list(row)).get(table.columns[0], "")
                    if table.columns else "",
                ))
            continue

        # Khoi van ban: tach cau, gop 2 cau lien tiep neu cosine >= theta.
        sents = _sentences(spec["text"])
        if not sents:
            continue
        if embed is not None and len(sents) > 1:
            vecs = embed(sents)
            groups: list[list[str]] = [[sents[0]]]
            for i in range(1, len(sents)):
                if _cos(vecs[i - 1], vecs[i]) >= cfg.small_theta:
                    groups[-1].append(sents[i])
                else:
                    groups.append([sents[i]])
        else:
            groups = [[s] for s in sents]
        for gi, group in enumerate(groups):
            small_chunks.append(_new_chunk(
                doc, meta, cfg, kind="dual_small", text=" ".join(group),
                page=spec["page"], chunk_id=f"{lid}#s{gi}", parent_id=lid,
            ))

    return small_chunks, large_chunks


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 3. Row JSON — moi dong la mot chunk tu chua.
# ---------------------------------------------------------------------------
def row_json_chunks(doc: dict, elements: Sequence[UElement],
                    cfg: MethodConfig | None = None) -> list[Chunk]:
    cfg = cfg or MethodConfig()
    meta = _doc_meta(doc)
    out: list[Chunk] = []
    for table in _iter_tables(elements):
        for ri, row in enumerate(table.rows):
            if not _has_number(row):
                continue
            out.append(_new_chunk(
                doc, meta, cfg, kind="row_json",
                text=table_row_json(table, row, meta),
                page=table.page,
                chunk_id=f"{table.table_id}#jr{ri}",
                table_id=table.table_id, row_index=ri,
            ))
    return out
