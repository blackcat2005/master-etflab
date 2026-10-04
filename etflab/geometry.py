"""Khoi phuc truc toa do: dong (line), o (cell), bien cot, dai trong doc.

Tieu chi cua phuong phap nay: ``pymupdf.find_tables()`` chi lam viec voi bang
co duong vien. Bao cao cua CTCK Viet Nam dung bang khong vien (borderless) —
chi duoc bao ve bang bang cach sap xep theo truc x. Toi day khoi phuc truc tiep
tu toa do tu.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .vnnum import Num, normalize_text, parse_number

__all__ = [
    "Cell",
    "Column",
    "Line",
    "Word",
    "assign_to_columns",
    "build_lines",
    "column_valleys",
    "group_cells",
    "median_char_width",
    "tighten_columns",
    "vertical_bands",
    "words_from_page",
    "words_in_slab",
]


@dataclass(slots=True)
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(slots=True)
class Cell:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    num: Num | None = None

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(slots=True)
class Line:
    y0: float
    y1: float
    words: list[Word] = field(default_factory=list)
    cells: list[Cell] = field(default_factory=list)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    @property
    def text(self) -> str:
        return normalize_text(" ".join(w.text for w in self.words))

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def numeric_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.num is not None]

    @property
    def numeric_density(self) -> float:
        if not self.words:
            return 0.0
        return len(self.numeric_cells) / len(self.words)

    @property
    def x_span(self) -> tuple[float, float]:
        if not self.words:
            return (0.0, 0.0)
        return (min(w.x0 for w in self.words), max(w.x1 for w in self.words))

    @property
    def has_text(self) -> bool:
        return bool(normalize_text(self.text))


@dataclass(frozen=True, slots=True)
class Column:
    index: int
    x0: float
    x1: float
    cx: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def words_from_page(page) -> list[Word]:
    out = [
        Word(x0=float(r[0]), y0=float(r[1]), x1=float(r[2]), y1=float(r[3]), text=str(r[4]))
        for r in page.get_text("words")
        if str(r[4]).strip()
    ]
    out.sort(key=lambda w: (w.cy, w.x0))
    return out


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def median_char_width(words: list[Word]) -> float:
    """Do rong trung binh cua MOT ky tu, dung lam co so cho cac nguong khoang cach."""
    samples = [
        (w.x1 - w.x0) / len(w.text)
        for w in words
        if len(w.text) >= 3 and w.height > 0 and w.x1 > w.x0
    ]
    return _median(samples) or 3.0


def build_lines(words: list[Word], y_tol: float = 3.0) -> list[Line]:
    """Gom tu thanh dong theo toa do y. Tu da sap xo theo (cy, x0)."""
    if not words:
        return []
    groups: list[list[Word]] = [[words[0]]]
    for prev, cur in itertools.pairwise(words):
        if abs(cur.cy - prev.cy) <= y_tol:
            groups[-1].append(cur)
        else:
            groups.append([cur])
    lines = [
        Line(y0=min(w.y0 for w in g), y1=max(w.y1 for w in g), words=g) for g in groups
    ]
    for line in lines:
        line.words.sort(key=lambda w: w.x0)
    lines.sort(key=lambda ln: ln.y0)
    return lines


def group_cells(line: Line, char_w: float) -> list[Cell]:
    """Tach dong thanh o dua tren khoang cach ngang giua cac tu."""
    if not line.words:
        line.cells = []
        return []
    gap_tol = max(4.0, char_w * 1.6)
    buckets: list[list[Word]] = [[line.words[0]]]
    for prev, cur in zip(line.words, line.words[1:]):
        if cur.x0 - prev.x1 > gap_tol:
            buckets.append([cur])
        else:
            buckets[-1].append(cur)

    cells = [
        Cell(
            x0=min(w.x0 for w in b),
            y0=min(w.y0 for w in b),
            x1=max(w.x1 for w in b),
            y1=max(w.y1 for w in b),
            text=normalize_text(" ".join(w.text for w in b)),
        )
        for b in buckets
    ]
    for cell in cells:
        cell.num = parse_number(cell.text)
    line.cells = cells
    return cells


def words_in_slab(words: list[Word], lo: float, hi: float) -> list[Word]:
    return [w for w in words if w.x1 > lo and w.x0 < hi]


def _empty_runs(
    lines: list[Line],
    lo: float,
    hi: float,
    *,
    char_w: float,
    min_gap: float,
    allow_sparse: int,
) -> list[tuple[float, float]]:
    """Cac dai trong doc ma khong dong nao phu de, tren toan bo danh sach dong."""
    bins = max(1, int(hi - lo))
    if bins <= 0:
        return []
    row_marks: list[list[bool]] = []
    for line in lines:
        marks = bytearray(bins)
        for w in line.words:
            a = max(0, int(w.x0 - lo))
            b = min(bins - 1, int(w.x1 - lo))
            for i in range(a, b + 1):
                marks[i] = 1
        row_marks.append(marks)

    gaps = bytearray(bins)
    for i in range(bins):
        gaps[i] = sum(1 for marks in row_marks if marks[i])
        if gaps[i] <= allow_sparse:
            gaps[i] = 0

    runs: list[tuple[float, float]] = []
    start: int | None = None
    for i in range(bins):
        if gaps[i] == 0:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_gap:
                    runs.append((lo + start, lo + i))
                start = None
    if start is not None and bins - start >= min_gap:
        runs.append((lo + start, hi))
    del char_w
    return runs


def column_valleys(lines: list[Line], char_w: float, *, allow_sparse: int = 0) -> list[Column]:
    """Bien tach cot tu khoang trong doc chung giua cac dong du lieu."""
    spans = [ln.x_span for ln in lines if ln.words]
    if not spans:
        return []
    lo = min(s[0] for s in spans)
    hi = max(s[1] for s in spans)
    if hi <= lo:
        return []

    min_gap = max(7.0, char_w * 2.0)
    cuts = _empty_runs(lines, lo, hi, char_w=char_w, min_gap=min_gap, allow_sparse=allow_sparse)

    edges: list[tuple[float, float]] = []
    cursor = lo
    for a, b in cuts:
        # Moi khe rong >= min_gap la mot ranh giua cot. Khong bo qua khe hep:
        # con sot no se lam cot sau bi nuot vao cot truoc.
        if a > cursor:
            edges.append((cursor, a))
            cursor = b
    if hi > cursor:
        edges.append((cursor, hi))

    return [Column(i, a, b, (a + b) / 2.0) for i, (a, b) in enumerate(edges) if b > a]


def vertical_bands(
    lines: list[Line], char_w: float, *, min_width: float | None = None, allow_sparse: int = 0
) -> list[tuple[float, float]]:
    """Dai trong doc rong chan doc giua hai bang dat canh nhau."""
    spans = [ln.x_span for ln in lines if ln.words]
    if not spans:
        return []
    lo = min(s[0] for s in spans)
    hi = max(s[1] for s in spans)
    if hi <= lo:
        return []
    width = min_width if min_width is not None else max(14.0, char_w * 4.0)
    return _empty_runs(lines, lo, hi, char_w=char_w, min_gap=width, allow_sparse=allow_sparse)


def tighten_columns(columns: list[Column], lines: list[Line]) -> list[Column]:
    """Thu gon bien cot ve dung phan do lieu, de header canh giua van khop cot."""
    out: list[Column] = []
    for col in columns:
        xs0: list[float] = []
        xs1: list[float] = []
        for line in lines:
            for cell in line.cells:
                if min(cell.x1, col.x1) - max(cell.x0, col.x0) > 0:
                    xs0.append(max(cell.x0, col.x0))
                    xs1.append(min(cell.x1, col.x1))
        if not xs0:
            out.append(col)
            continue
        out.append(
            Column(col.index, min(xs0) - 1.0, max(xs1) + 1.0, col.cx)
        )
    return out


def assign_to_columns(line: Line, columns: list[Column]) -> list[Cell | None]:
    """Gan moi o cua dong vao cot kiem so bang do phu de lon nhat."""
    out: list[Cell | None] = [None] * len(columns)
    if not columns or not line.cells:
        return out
    for cell in line.cells:
        overlap = min(cell.x1, columns[-1].x1) - max(cell.x0, columns[0].x0)
        if overlap <= 0:
            continue
        best_i, best_ov = -1, 0.0
        for col in columns:
            ov = min(cell.x1, col.x1) - max(cell.x0, col.x0)
            if ov > best_ov:
                best_i, best_ov = col.index, ov
        if best_i < 0:
            continue
        need = max(1.0, cell.x1 - cell.x0) * 0.30
        if best_ov < need:
            continue
        if out[best_i] is None:
            out[best_i] = cell
        else:
            prev = out[best_i]
            assert prev is not None
            merged = Cell(
                x0=min(prev.x0, cell.x0),
                y0=min(prev.y0, cell.y0),
                x1=max(prev.x1, cell.x1),
                y1=max(prev.y1, cell.y1),
                text=normalize_text(f"{prev.text} {cell.text}"),
            )
            merged.num = parse_number(merged.text)
            out[best_i] = merged
    return out
