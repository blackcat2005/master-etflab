"""Adapter cho output cua ``unstructured`` (strategy=``hi_res``).

Vi sao co file nay: phuong phap tach bang hinh hoc cua ``src`` chi dung
cho mot ho tai lieu. Ban thi nghiem nay thay no bang ``unstructured`` —
layout model (YOLOX) + Table Transformer — de tra loi cau hoi "neu dung mot
parser pho thong thi chunking con giu duoc loi the khong".

Diem quan trong: **khong dung hinh hoc o day**. Bang duoc lay tu
``metadata.text_as_html`` do Table Transformer sinh ra; moi ``<tr>`` la mot
dong, ``colspan``/``rowspan`` duoc trai thanh luoi chu nhat. Nho vay cung
mot dau vao cho ca ba chien luoc chunking (Chipper, dual-granularity,
row-JSON), va ket qua chi con khac nhau o *cach phan manh*.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

__all__ = [
    "UElement",
    "UTable",
    "elements_from_cache",
    "load_unstructured_doc",
    "parse_table_html",
]


# ---------------------------------------------------------------------------
# Mo hinh du lieu.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class UTable:
    """Bang da chuan hoa tu HTML cua unstructured."""

    table_id: str
    page: int
    caption: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    html: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def to_text(self) -> str:
        """Ban van ban de doc: caption, header, tung dong ngan bang ' | '."""
        parts: list[str] = []
        if self.caption:
            parts.append(self.caption)
        parts.append(" | ".join(self.columns))
        for row in self.rows:
            parts.append(" | ".join(row))
        return "\n".join(parts)

    def row_dict(self, row: list[str]) -> dict[str, str]:
        return {
            col: (row[i] if i < len(row) else "")
            for i, col in enumerate(self.columns)
        }


@dataclass(slots=True)
class UElement:
    """Mot phan tu theo thu tu doc cua unstructured."""

    kind: str  # title | narrative | list | table | caption | header | footer | ...
    text: str
    page: int
    table: UTable | None = None
    bbox: tuple[float, float, float, float] | None = None

    @property
    def is_table(self) -> bool:
        return self.kind == "table" and self.table is not None


# ---------------------------------------------------------------------------
# Parse HTML cua Table Transformer.
# ---------------------------------------------------------------------------
class _TableHTMLParser(HTMLParser):
    """Doc ``<table>`` thanh luoi, trai ``rowspan``/``colspan``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self._row: list[tuple[str, int, int, bool]] | None = None
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self._is_header = False
        self._in_thead = 0
        self.header_rows = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        if tag == "thead":
            self._in_thead += 1
            return
        if tag == "tr":
            self._row = []
            return
        if tag in ("td", "th"):
            if self._row is None:
                self._row = []
            self._cell = []
            self._is_header = tag == "th"
            self._span = (
                _int_attr(attrs_d.get("colspan"), 1),
                _int_attr(attrs_d.get("rowspan"), 1),
            )
            return
        if tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self._in_thead = max(0, self._in_thead - 1)
            return
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            assert self._row is not None
            self._row.append((text, self._span[0], self._span[1], self._is_header))
            self._cell = None
            return
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            if self._row and all(c[3] for c in self._row) or self._in_thead:
                self.header_rows += 1
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _int_attr(value: str | None, default: int) -> int:
    try:
        return max(1, int(str(value)))
    except (TypeError, ValueError):
        return default


def _expand(rows: list[list[tuple[str, int, int, bool]]]) -> list[list[str]]:
    """Trai colspan/rowspan thanh luoi chu nhat; o bi phu bo trong."""
    grid: list[list[str | None]] = []
    for r, row in enumerate(rows):
        while len(grid) <= r:
            grid.append([])
        c = 0
        for text, colspan, rowspan, _ in row:
            while c < len(grid[r]) and grid[r][c] is not None:
                c += 1
            for dr in range(rowspan):
                rr = r + dr
                while len(grid) <= rr:
                    grid.append([])
                for dc in range(colspan):
                    cc = c + dc
                    while len(grid[rr]) <= cc:
                        grid[rr].append(None)
                    # Chi dat text o o goc; o bi trai de trong (khac None)
                    # de buoc "tim o trong" biet o da bi chiem.
                    if dr == 0 and dc == 0:
                        grid[rr][cc] = text
                    elif grid[rr][cc] is None:
                        grid[rr][cc] = ""
            c += colspan
    width = max((len(r) for r in grid), default=0)
    out: list[list[str]] = []
    for row in grid:
        out.append([(cell or "") for cell in row] + [""] * (width - len(row)))
    return out


def _header_depth(grid: list[list[str]], n_header: int) -> int:
    """So dong header: uu tien <thead>/<th>, neu khong co thi doan theo so."""
    if n_header > 0:
        return min(n_header, len(grid))
    for i, row in enumerate(grid):
        filled = [c for c in row if c.strip()]
        if not filled:
            continue
        numeric = sum(1 for c in filled if _looks_numeric(c))
        if numeric / len(filled) < 0.5:
            return i + 1
        return 0
    return 0


_NUM_RE = re.compile(r"^[(\-−–—]?[\d.,]+\)?%?$")


def _looks_numeric(text: str) -> bool:
    t = text.strip().replace(" ", "")
    return bool(t) and bool(_NUM_RE.match(t))


def _merge_header(grid: list[list[str]], depth: int) -> list[str]:
    """Gop header nhieu tang thanh mot nhan: "Năm / 2024"."""
    if depth <= 0:
        return [f"Cot {i + 1}" for i in range(len(grid[0]))] if grid else []
    width = len(grid[0])
    # Forward-fill trong tung dong header: o bi colspan de trong o cac cot
    # bi phu, nen nhan nhom ("Tỷ trọng") phai duoc keo sang phai.
    filled_grid: list[list[str]] = []
    for r in range(min(depth, len(grid))):
        row = [(grid[r][c].strip() if c < len(grid[r]) else "") for c in range(width)]
        last = ""
        for c in range(width):
            if row[c]:
                last = row[c]
            elif last:
                row[c] = last
        filled_grid.append(row)

    labels: list[str] = []
    for c in range(width):
        parts: list[str] = []
        for row in filled_grid:
            part = row[c]
            if part and part not in parts:
                parts.append(part)
        labels.append(" / ".join(parts) if parts else f"Cot {c + 1}")
    seen: dict[str, int] = {}
    for i, label in enumerate(labels):
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            labels[i] = f"{label} ({seen[label]})"
    return labels


def parse_table_html(html: str, *, page: int = 0, caption: str = "",
                     table_id: str = "t0") -> UTable:
    """HTML -> UTable. Rong/khong co <tr> -> bang 0 dong (caller tu xu ly)."""
    parser = _TableHTMLParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:  # noqa: BLE001 — HTML hong: tra bang rong thay vi no.
        return UTable(table_id=table_id, page=page, caption=caption, html=html)

    grid = _expand(parser.rows)
    grid = [row for row in grid if any(c.strip() for c in row)]
    if not grid:
        return UTable(table_id=table_id, page=page, caption=caption, html=html)

    depth = _header_depth(grid, parser.header_rows)
    depth = max(depth, 1)  # luon co it nhat 1 dong header
    columns = _merge_header(grid, depth)
    rows = grid[depth:]
    # Cat cot trong o cuoi (Table Transformer hay them cot rong).
    if rows:
        while len(columns) > 1 and all(row and not row[-1].strip() for row in rows):
            columns.pop()
            rows = [row[:-1] for row in rows]
    return UTable(
        table_id=table_id,
        page=page,
        caption=caption,
        columns=columns,
        rows=rows,
        html=html,
    )


# ---------------------------------------------------------------------------
# Doc cache JSON do scripts/ingest/parse_unstructured.py ghi ra.
# ---------------------------------------------------------------------------
_KIND_MAP = {
    "Title": "title",
    "Section-header": "title",
    "Header": "header",
    "Footer": "footer",
    "PageBreak": "pagebreak",
    "NarrativeText": "narrative",
    "Text": "narrative",
    "UncategorizedText": "narrative",
    "ListItem": "list",
    "Table": "table",
    "FigureCaption": "caption",
    "Image": "image",
    "Formula": "formula",
    "Address": "address",
    "EmailAddress": "address",
}

_SKIP_KINDS = {"header", "footer", "pagebreak", "image", "formula"}

_CAPTION_RE = re.compile(
    r"^\s*(b\u1ea3ng|table|bi\u1ec3u|h\u00ecnh)\s*(\d+[.:\-)]|\b)", re.IGNORECASE
)


def elements_from_cache(payload: dict[str, Any]) -> list[UElement]:
    """JSON cache -> danh sach UElement (HTML duoc parse tai day)."""
    out: list[UElement] = []
    for el in payload.get("elements", []):
        kind = el["kind"]
        page = int(el.get("page", 1))
        if kind == "table":
            html = el.get("html") or ""
            table = parse_table_html(
                html, page=page, caption="",
                table_id=el.get("table_id", f"t{len(out)}"),
            )
            if not table.rows:
                continue
            out.append(UElement(kind="table", text=table.to_text(),
                                page=page, table=table))
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        out.append(UElement(kind=kind, text=text, page=page))
    _attach_captions(out)
    return out


def _attach_captions(elements: list[UElement]) -> None:
    """Gan caption cho bang theo thu tu doc (khong dung toa do).

    Table Transformer khong tra ve caption. Quy uoc: lay phan tu gan nhat
    phia truoc tren cung trang neu no la title/caption hoac bat dau bang
    "Bảng/Table/Biểu". Neu khong co thi de trong.
    """
    for i, el in enumerate(elements):
        if not el.is_table or el.table is None or el.table.caption:
            continue
        for j in range(i - 1, max(-1, i - 4), -1):
            prev = elements[j]
            if prev.page != el.page:
                break
            if prev.kind in ("title", "caption") or _CAPTION_RE.match(prev.text):
                el.table.caption = prev.text.strip()
                break


def load_unstructured_doc(path: str | Path) -> dict[str, Any]:
    """Doc 1 file cache va tra ve dict tien dung cho eval."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    elements = elements_from_cache(payload)
    return {
        "source_file": payload["source_file"],
        "elements": elements,
        "n_pages": payload.get("n_pages", 0),
        "parser": payload.get("parser", "unstructured"),
    }


def skip_kind(kind: str) -> bool:
    return kind in _SKIP_KINDS
