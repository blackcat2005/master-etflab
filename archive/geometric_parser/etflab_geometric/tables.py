"""Dò vùng bảng không viền và khoi phuc bang co duong dan (ticker) + don vi."""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .geometry import (
    Cell,
    Column,
    Line,
    Word,
    assign_to_columns,
    build_lines,
    column_valleys,
    group_cells,
    median_char_width,
    tighten_columns,
    vertical_bands,
    words_in_slab,
)
from .vnnum import normalize_text, parse_number, strip_accents

__all__ = ["Table", "TableExtractor", "TickerExtractor"]


@dataclass(frozen=True, slots=True)
class TableConfig:
    y_tol: float = 3.0
    min_numeric_cells: int = 3
    min_numeric_density: float = 0.25
    min_run_rows: int = 4
    max_header_lines: int = 6
    header_overlap_ratio: float = 0.25
    header_cell_capture: float = 0.40
    min_split_band: float | None = None
    min_cols_per_table: int = 2
    max_cols_single_table: int = 8
    min_group_title_len: int = 12
    min_group_title_cols: int = 2
    min_row_alignment: float = 0.60
    caption_max_lines: int = 4
    caption_pattern: str = r"^\s*(bảng|table)\s*(\d+\s*)?([:.\-–)]|$)"


@dataclass(slots=True)
class Table:
    table_id: str
    source_file: str
    page: int
    page_span: list[int]
    caption: str
    columns: list[str]
    header_levels: list[list[str]]
    rows: list[dict]
    n_rows: int
    n_cols: int
    unit_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    region_y: tuple[float, float] = (0.0, 0.0)
    x_span: tuple[float, float] = (0.0, 0.0)
    header_cell_count: int = 0
    numeric_cols: list[int] = field(default_factory=list)
    row_index_col: int | None = None

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(normalize_key(c) for c in self.columns)

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "source_file": self.source_file,
            "page": self.page,
            "page_span": self.page_span,
            "caption": self.caption,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": self.columns,
            "header_levels": self.header_levels,
            "unit_notes": self.unit_notes,
            "rows": self.rows,
            "warnings": self.warnings,
        }


_UNIT_RE = re.compile(
    r"\((?P<scale>nghìn|ngàn|tr|tỷ|triệu)\s+(?P<what>cổ phiếu|cp|đồng|usd|vnd)\)",
    re.IGNORECASE,
)
_TICKER_RE = re.compile(r"^[A-ZĐ]{2,6}$")
_RUNNING_RE = re.compile(
    r"(https?://|www\.|visit\s|trang\s+\d+|page\s+\d+\s*(of|/)|\bdoi\b|"
    r"\b(19|20)\d{2}\s*;\s*\d+\s*:\s*\d+|^\s*\d+\s*$|\|\s*\d+\s*$)",
    re.IGNORECASE,
)


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(normalize_text(text)).lower()).strip()


class TableExtractor:
    def __init__(self, config: TableConfig | None = None) -> None:
        self.cfg = config or TableConfig()
        # Y cua dong tieu de nhom da dung de tach bang. Giu y (khong xoa dong)
        # de khong pha logic noi trang, chi bo qua khi dat ten cot.
        self._title_ys: set[int] = set()

    def extract(self, page_words: list[Word], page_no: int) -> list[Table]:
        cfg = self.cfg
        if not page_words:
            return []
        char_w = median_char_width(page_words)
        lines = build_lines(page_words, cfg.y_tol)
        for line in lines:
            group_cells(line, char_w)

        regions = self._find_regions(lines, char_w)
        tables: list[Table] = []
        for region in regions:
            tables.extend(self._extract_region(region, page_words, char_w, page_no))
        return tables

    def _find_regions(
        self, lines: list[Line], char_w: float
    ) -> list[tuple[list[Line], list[Line]]]:
        cfg = self.cfg
        flags = [
            (len(ln.numeric_cells) >= cfg.min_numeric_cells
             and ln.numeric_density >= cfg.min_numeric_density)
            for ln in lines
        ]
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i, ok in enumerate(flags):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                if i - start >= cfg.min_run_rows:
                    runs.append((start, i - 1))
                start = None
        if start is not None and len(flags) - start >= cfg.min_run_rows:
            runs.append((start, len(flags) - 1))

        regions: list[tuple[list[Line], list[Line]]] = []
        for lo, hi in runs:
            lo, hi = self._grow_run(lines, lo, hi, char_w)
            data = lines[lo : hi + 1]
            if not self._grid_is_stable(data, char_w):
                continue
            header = self._header_block(lines, lo, data)
            regions.append((lines[header : lo], data))
        return regions

    def _grow_run(
        self, lines: list[Line], lo: int, hi: int, char_w: float
    ) -> tuple[int, int]:
        """Noi them dong thieu o so vao vung bang.

        Mot dong du lieu co nhieu o trong (cot rong) se khong duot nguong "toi
        thieu 3 o so" va bi bo sach lam mat ngam — trong bao cao tai chinh dang
        nay rat pho bien. Dong chi duoc noi khi van con o so VA phu de dung
        luoi cot, nen khong lan sang header hay van ban.
        """
        need_numeric = 1
        for _ in range(2):
            data = lines[lo : hi + 1]
            cols = tighten_columns(column_valleys(data, char_w, allow_sparse=1), data)
            if len(cols) < 2:
                break
            spans = [(c.x0, c.x1) for c in cols]
            need_cells = max(2, int(self.cfg.min_row_alignment * len(cols)))

            def aligned(
                line: Line,
                spans: list[tuple[float, float]] = spans,
                need_cells: int = need_cells,
            ) -> bool:
                if len(line.numeric_cells) < need_numeric:
                    return False
                hits = sum(
                    1
                    for cell in line.cells
                    if any(
                        min(cell.x1, b) - max(cell.x0, a) > 0 for a, b in spans
                    )
                )
                return hits >= need_cells

            i = lo - 1
            while i >= 0 and aligned(lines[i]):
                lo = i
                i -= 1
            i = hi + 1
            while i < len(lines) and aligned(lines[i]):
                hi = i
                i += 1
        return lo, hi

    def _grid_is_stable(self, data: list[Line], char_w: float) -> bool:
        """Vung du lieu co phu de luoi cot deu khong.

        Van ban justify co nhieu con so deu khong duoc coi la bang. Tieu chi la
        so O phu de MOT COT DA XAC DINH, khong phai so O thuc te — dong thieu
        o trong (cot rong) van la dong du lieu hop le.
        """
        cols = tighten_columns(column_valleys(data, char_w, allow_sparse=1), data)
        if len(cols) < 3:
            return False
        spans = [(c.x0, c.x1) for c in cols]
        need_cells = max(2, int(self.cfg.min_row_alignment * len(cols)))
        ok_lines = 0
        for line in data:
            if len(line.numeric_cells) < 2:
                continue
            hits = sum(
                1
                for cell in line.cells
                if any(min(cell.x1, b) - max(cell.x0, a) > 0 for a, b in spans)
            )
            if hits >= need_cells:
                ok_lines += 1
        return ok_lines >= max(self.cfg.min_run_rows, int(0.7 * len(data)))

    def _header_block(self, lines: list[Line], data_start: int, data: list[Line]) -> int:
        cfg = self.cfg
        body_left = min((w.x0 for ln in data for w in ln.words), default=0.0)
        idx = data_start - 1
        limit = max(0, data_start - cfg.max_header_lines)
        caption_re = re.compile(cfg.caption_pattern, re.IGNORECASE)
        while idx >= limit:
            line = lines[idx]
            if len(line.numeric_cells) >= 2:
                break
            if caption_re.match(line.text) or _RUNNING_RE.search(line.text):
                break
            if _is_prose_residue(line, body_left):
                break
            idx -= 1
        return idx + 1

    def _extract_region(
        self,
        region: tuple[list[Line], list[Line]],
        page_words: list[Word],
        char_w: float,
        page_no: int,
    ) -> list[Table]:
        cfg = self.cfg
        header_lines, data_lines = region
        all_lines = header_lines + data_lines
        region_y = (
            min(ln.y0 for ln in all_lines),
            max(ln.y1 for ln in all_lines),
        )

        whole = self._build_subtable(header_lines, data_lines, char_w, None, region_y)
        if whole is None:
            return []
        if whole.n_cols <= cfg.max_cols_single_table:
            return [whole]

        lo = min(ln.x_span[0] for ln in all_lines)
        hi = max(ln.x_span[1] for ln in all_lines)

        # Dong tieu de nhom ("Kich ban 1: ...") la bang chung de tach bang. Sau
        # khi tach xong no la ten bang chu khong phai ten cot. Giu nguyen dong
        # trong header (logic noi trang doi chieu header giua cac trang) va chi
        # danh dau de bo qua khi dat ten cot.
        bands, title_line = self._group_title_bands(header_lines, data_lines, char_w, lo, hi)
        self._title_ys = {round(title_line.y0 * 2)} if title_line is not None else set()

        slabs = self._split_slab(
            header_lines, data_lines, char_w, lo, hi, region_y, title_bands=bands
        )
        if len(slabs) <= 1:
            return [whole]
        out = [
            self._build_subtable(header_lines, data_lines, char_w, (a, b), region_y)
            for a, b in slabs
        ]
        return [t for t in out if t is not None] or [whole]

    def _split_slab(
        self,
        header_lines: list[Line],
        data_lines: list[Line],
        char_w: float,
        lo: float,
        hi: float,
        region_y: tuple[float, float],
        depth: int = 0,
        title_bands: list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, float]]:
        """Cat doi bang bang nhau o mot dai trong doc, chi nhan khi ca hai ben deu la bang."""
        if depth >= 2 or hi <= lo:
            return [(lo, hi)]
        whole = self._build_subtable(header_lines, data_lines, char_w, (lo, hi), region_y)
        if whole is None or whole.n_cols <= self.cfg.max_cols_single_table:
            return [(lo, hi)]
        lines = [
            ln
            for ln in header_lines + data_lines
            if words_in_slab(ln.words, lo, hi)
        ]
        # Uu tien khe trang: no chi tach khi ca hai ben deu la bang. Tieu de
        # nhom chi la du phong cho truong hop khe khong du ro (bang tin Q1/2026).
        bands = vertical_bands(lines, char_w, min_width=self.cfg.min_split_band)
        if not bands and title_bands:
            bands = title_bands
        if not bands:
            return [(lo, hi)]

        # Hai bang dat canh nhau thuong duoc canh giua trang, nen uu tien dai
        # o gan giua vung nhat. Dai trong doc giua cac cot cua cung mot bang
        # cung du dieu kien "hai ben deu la bang" — neu chon sai se tach nham
        # mot bang thanh hai.
        center = (lo + hi) / 2.0
        ranked = sorted(bands, key=lambda ab: abs((ab[0] + ab[1]) / 2.0 - center))
        for a, b in ranked:
            if a <= lo or b >= hi:
                continue
            left = self._build_subtable(header_lines, data_lines, char_w, (lo, a), region_y)
            right = self._build_subtable(header_lines, data_lines, char_w, (b, hi), region_y)
            if not (self._is_plausible(left) and self._is_plausible(right)):
                continue
            return (
                self._split_slab(header_lines, data_lines, char_w, lo, a, region_y, depth + 1)
                + self._split_slab(header_lines, data_lines, char_w, b, hi, region_y, depth + 1)
            )
        return [(lo, hi)]

    def _group_title_bands(
        self,
        header_lines: list[Line],
        data_lines: list[Line],
        char_w: float,
        lo: float,
        hi: float,
    ) -> tuple[list[tuple[float, float]], Line | None]:
        """Khoang cat lay tu dong tieu de NHOM trong header, truoc khe trang.

        Hai bang dat canh nhau doi khi khong he khe: bang tin Q1/2026, khe giua
        hai bang chi 12pt, nho HON khe giua hai cot cua cung mot bang (16-70pt),
        nen ``vertical_bands`` khong tim ra. Header lai co dong tieu de nhom
        ("Kich ban 1: ..." / "Kich ban 2: ..."), moi o la mot bang — bang chung
        manh hon khoang trang, va no khong phu thuoc do rong khe.

        Chi chap nhan dong do khi:
        - moi o deu la chu dai va deu phu it nhat ``min_group_title_cols`` cot;
        - cac o co chu KHAC NHAU. Header nhieu tang that ("Số lượng CP mua/bán"
          lap o ca hai ben) lap lai cung mot nhan; "Kich ban 1" / "Kich ban 2"
          moi la tieu de cua hai bang canh nhau.
        """
        cfg = self.cfg
        centers = _distinct_centers(data_lines)
        if len(centers) < 6:
            return [], None
        for line in header_lines:
            cells = [c for c in line.cells if c.text.strip()]
            if len(cells) < 2:
                continue
            if any(len(c.text.strip()) < cfg.min_group_title_len for c in cells):
                continue
            labels = [" ".join(c.text.split()).casefold() for c in cells]
            if len(set(labels)) != len(labels):
                continue
            if min(sum(1 for x in centers if c.x0 <= x <= c.x1) for c in cells) < (
                cfg.min_group_title_cols
            ):
                continue
            titles = sorted((c.x0 + c.x1) / 2.0 for c in cells)
            cuts = [(titles[i] + titles[i + 1]) / 2.0 for i in range(len(titles) - 1)]
            groups: list[list[float]] = [[] for _ in range(len(titles))]
            for x in centers:
                groups[sum(1 for cut in cuts if x >= cut)].append(x)
            if any(len(g) < cfg.min_cols_per_table + 1 for g in groups):
                continue
            if groups[0][0] <= lo or groups[-1][-1] >= hi:
                continue
            bands = [(groups[i][-1], groups[i + 1][0]) for i in range(len(groups) - 1)]
            if any(b - a < 3.0 * char_w for a, b in bands):
                continue
            return bands, line
        return [], None

    def _build_subtable(
        self,
        header_lines: list[Line],
        data_lines: list[Line],
        char_w: float,
        slab: tuple[float, float] | None,
        region_y: tuple[float, float],
    ) -> Table | None:
        cfg = self.cfg
        local_header = self._restrict(header_lines, slab, char_w)
        local_data = self._restrict(data_lines, slab, char_w)
        if len(local_data) < cfg.min_run_rows:
            return None

        columns = column_valleys(local_data, char_w, allow_sparse=1)
        if len(columns) < cfg.min_cols_per_table:
            return None
        columns = tighten_columns(columns, local_data)

        col_names = self._column_names(local_header, columns, char_w)
        rows = self._rows(local_data, columns)
        levels = self._header_levels(local_header, columns)
        col_names = self._names_from_levels(levels, col_names)
        numeric_cols, row_index_col = self._column_profile(rows, columns)
        spans = [ln.x_span for ln in local_data if ln.words]
        x_span = (
            (min(s[0] for s in spans), max(s[1] for s in spans)) if spans else (0.0, 0.0)
        )

        units: list[str] = []
        for name in col_names:
            for m in _UNIT_RE.finditer(name):
                units.append(f"{m.group('scale')} {m.group('what')}")
        if any("%" in n for n in col_names):
            units.append("percent")

        return Table(
            table_id="",
            source_file="",
            page=0,
            page_span=[],
            caption="",
            columns=col_names,
            header_levels=levels,
            rows=rows,
            n_rows=len(rows),
            n_cols=len(columns),
            unit_notes=sorted(set(units)),
            header_cell_count=len(local_header),
            region_y=region_y,
            x_span=x_span,
            numeric_cols=numeric_cols,
            row_index_col=row_index_col,
        )

    def _column_profile(
        self, rows: list[dict], columns: Sequence[Column]
    ) -> tuple[list[int], int | None]:
        """Cot nao that su chua du lieu so, va cot nao la so thu tu (STT)."""
        numeric_cols: list[int] = []
        row_index_col: int | None = None
        need = max(2, int(0.4 * len(rows)))
        for col in columns:
            values = [r.get(str(col.index)) for r in rows]
            filled = [v for v in values if v not in (None, "")]
            hits = sum(1 for v in filled if parse_number(v) is not None)
            if hits >= need:
                if row_index_col is None and _is_row_index(values):
                    row_index_col = col.index
                else:
                    numeric_cols.append(col.index)
        return numeric_cols, row_index_col

    def _is_plausible(self, table: Table | None) -> bool:
        if table is None:
            return False
        cfg = self.cfg
        return (
            table.n_rows >= cfg.min_run_rows
            and table.n_cols >= cfg.min_cols_per_table
            and len(table.numeric_cols) >= 2
        )

    def _restrict(
        self, lines: list[Line], slab: tuple[float, float] | None, char_w: float
    ) -> list[Line]:
        """Gioi han cac dong vao dai doc va dung lai cell cho do bang toa do moi."""
        if slab is None:
            return lines
        words: list[Word] = []
        for line in lines:
            words.extend(words_in_slab(line.words, slab[0], slab[1]))
        if not words:
            return []
        rebuilt = build_lines(words, self.cfg.y_tol)
        for line in rebuilt:
            group_cells(line, char_w)
        return rebuilt

    def _column_names(
        self, header_lines: list[Line], columns: Sequence[Column], char_w: float
    ) -> list[str]:
        buckets: list[list[tuple[float, float, str]]] = [[] for _ in columns]
        for line in header_lines:
            for cell in _segment_header_cells(line, char_w):
                target = self._match_column(cell, columns, self.cfg.header_cell_capture, char_w)
                if target is not None:
                    buckets[target].append((line.y0, cell.x0, cell.text))
        names: list[str] = []
        for i, bucket in enumerate(buckets):
            if not bucket:
                names.append(f"col_{i}")
                continue
            bucket.sort(key=lambda t: (t[0], t[1]))
            seen: set[str] = set()
            parts: list[str] = []
            for _, _, text in bucket:
                key = normalize_key(text)
                if key in seen:
                    continue
                seen.add(key)
                parts.append(text)
            names.append(normalize_text(" ".join(parts)))
        return names

    def _names_from_levels(
        self, levels: list[list[str]], fallback: list[str]
    ) -> list[str]:
        """Ten cot lay tu cac tang header: o header gop nhieu cot duoc giu o nhieu cot.

        `_header_levels` gan moi o header cho cot co do phu de lon nhat, nen mot
        o gop nhu "So luong CP mua ban" xuat hien o ca cot dau va cot sau.
        """
        names: list[str] = []
        for i in range(len(fallback)):
            seen: set[str] = set()
            parts: list[str] = []
            for row in levels:
                if i >= len(row):
                    continue
                text = row[i]
                if not text:
                    continue
                key = normalize_key(text)
                if key in seen:
                    continue
                seen.add(key)
                parts.append(text)
            name = normalize_text(" ".join(parts))
            names.append(name or fallback[i])
        return names

    def _match_column(
        self, cell, columns: Sequence[Column], ratio: float, char_w: float
    ) -> int | None:
        """Gan o header vao cot: uu tien do phu de, neu khong khop thi dung tam gan nhat.

        Header thuong can giua con du lieu can phai, nen chu 'VN30' co the nam
        lech sang cot ben canh du khong du 40% do phu de.
        """
        best_i, best_ov = -1, 0.0
        for col in columns:
            ov = min(cell.x1, col.x1) - max(cell.x0, col.x0)
            if ov / max(col.width, 1.0) > best_ov:
                best_i, best_ov = col.index, ov / max(col.width, 1.0)
        if best_i >= 0 and best_ov >= ratio:
            return best_i

        centers = sorted((c.cx, c.index) for c in columns)
        spacing = (
            (centers[-1][0] - centers[0][0]) / max(len(centers) - 1, 1)
            if len(centers) > 1
            else 0.0
        )
        tol = max(12.0, spacing * 0.6, char_w * 3.0)
        nearest_i, nearest_d = -1, 1e18
        for cx, idx in centers:
            d = abs(cx - cell.cx)
            if d < nearest_d:
                nearest_i, nearest_d = idx, d
        return nearest_i if nearest_d <= tol else None

    def _header_levels(
        self, header_lines: list[Line], columns: Sequence[Column]
    ) -> list[list[str]]:
        """Ghi tung tang header. Mot o gop se xuat hien o nhieu cot.

        Do phu de tinh theo chieu rong cot bien dong giua cac trang (so kho khac
        nhau), nen o duoc chap nhan khi ho phu >= 25% chieu rong cot HOAC >= 40%
        chieu rong chinh o — tieu chi thu hai on dinh hon.

        O trai het chieu rong bang (chu thich, ghi chu) bi loai: neu giu lai, no
        se duoc gan cho TAT CA cac cot vi do phu/cot ~= 1.0, lam ten cot bien
        thanh mot cau van xu ly. Hai dieu kien loai:

        - o phu tren 3 cot va hon mot nua tong so cot: khong phai header
          tung cap ma la van xu ly;
        - dong tieu de nhom da dung de tach bang ("Kich ban 1: ..."): ten bang,
          khong phai ten cot.
        """
        n_cols = len(columns)
        levels: list[list[str]] = []
        for line in header_lines:
            if round(line.y0 * 2) in self._title_ys:
                continue
            row: list[str] = []
            for ci, col in enumerate(columns):
                best: Cell | None = None
                best_ov = 0.0
                for cell in line.cells:
                    ov = min(cell.x1, col.x1) - max(cell.x0, col.x0)
                    if ov > best_ov:
                        best_ov = ov
                        best = cell
                if best is None:
                    row.append("")
                    continue
                own_w = max(best.x1 - best.x0, 1.0)
                reached = sum(
                    1
                    for c in columns
                    if min(best.x1, c.x1) - max(best.x0, c.x0) >= 0.5 * max(c.width, 1.0)
                )
                if reached >= 3 and reached * 2 > n_cols:
                    row.append("")
                    continue
                ratio = max(best_ov / max(col.width, 1.0), best_ov / own_w)
                if ratio < self.cfg.header_overlap_ratio:
                    row.append("")
                    continue
                row.append(
                    _text_in_column(line, best, columns, ci)
                )
            levels.append(row)
        return levels

    def _rows(self, data_lines: list[Line], columns: Sequence[Column]) -> list[dict]:
        out: list[dict] = []
        for line in data_lines:
            assigned = assign_to_columns(line, columns)
            record: dict = {}
            for col, cell in zip(columns, assigned):
                # Khoa luon la str: JSON chi cho phep khoa chuoi, dung kieu
                # trong RAM thi doc lai tu file se sinh KeyError.
                record[str(col.index)] = None if cell is None else cell.text
            out.append(record)
        return out


def _distinct_centers(lines: list[Line], tol: float = 2.0) -> list[float]:
    """Cac tam o du lieu theo truc x, ghep trung — dung de dem cot cua bang."""
    raw = sorted((c.x0 + c.x1) / 2.0 for ln in lines for c in ln.cells)
    out: list[float] = []
    for x in raw:
        if not out or x - out[-1] > tol:
            out.append(x)
    return out


def _is_prose_residue(line: Line, body_left: float) -> bool:
    """Doi la duoi nhac caption con sot, nam tren dong header.

    Vi du "22/04/2026." va "dữ liệu đến ngày 21/1/2026": no la duoi nhac
    "...tính đến ngày 22/04/2026." nhưng lai đứng giữa caption va header that.
    Dau bieu do lech sang trai ngoài bang (45.4pt so voi mep trai du lieu 48.4pt),
    nen khong the la dong header — neu giu lai, no se gan vao cot STT / Ma CP
    va bien ten cot thanh "22/04/2026. STT".

    Chi chan dong MOT o le; dong nhieu o thi dung cau truc bang roi.
    """
    if len(line.cells) != 1 or not line.has_text:
        return False
    return line.cells[0].x0 < body_left - 2.0


def _is_row_index(values: Sequence[object]) -> bool:
    """Cot so thu tu 1..N (STT) — khong phai cot du lieu do luong."""
    nums: list[int] = []
    for value in values:
        if value in (None, ""):
            continue
        num = parse_number(str(value))
        if num is None:
            return False
        val = num.value
        if abs(val - round(val)) > 1e-9:
            return False
        nums.append(round(val))
    if len(nums) < 4:
        return False
    return all(b - a == 1 for a, b in itertools.pairwise(nums))


def _best_column(word: Word, columns: Sequence[Column]) -> int:
    """Cot nao "nuot" tu nay. Tu tran sang canh khong chot voi cot nao thi
    gan theo cot gan nhat, thay vi bo han."""
    word_w = max(word.x1 - word.x0, 1.0)
    winner = 0
    best = -1e18
    for k, col in enumerate(columns):
        ov = min(word.x1, col.x1) - max(word.x0, col.x0)
        score = ov / word_w if ov > 0.0 else -abs(word.cx - col.cx) / word_w
        if score > best:
            best = score
            winner = k
    return winner


def _text_in_column(line: Line, cell: Cell, columns: Sequence[Column], index: int) -> str:
    """Ten cot chi tach khi o gop la nhieu O THAT bi gop nham.

    ``group_cells`` tach o bang khe ngang, nen hai cot hep can nhau (STT / Ma CP,
    khe ~6pt) bi gop lam mot va ca hai cot cung nhan "STT Ma CP". Dieu kien tach:
    o phu tren nhieu cot VA moi nhom tu deu roi vao dung cot cua no (do phu
    >= 60%).

    Header that duoc canh giua ("Tỷ trọng hiện tại") thi lo ra ca hai ben cot;
    nhung o do chi co mot nhom tu nen khong bi tach, giữ nguyen ca chuoi.
    """
    inside = [w for w in line.words if w.x0 >= cell.x0 - 0.5 and w.x1 <= cell.x1 + 0.5]
    if len(inside) < 2:
        return cell.text
    groups: dict[int, list[Word]] = {}
    for word in inside:
        groups.setdefault(_best_column(word, columns), []).append(word)
    if len(groups) < 2:
        return cell.text
    aligned = 0
    for k, group in groups.items():
        col = columns[k]
        gx0, gx1 = group[0].x0, group[-1].x1
        ov = min(gx1, col.x1) - max(gx0, col.x0)
        if ov >= 0.6 * max(gx1 - gx0, 1.0):
            aligned += 1
    if aligned < 2:
        return cell.text
    mine = groups.get(index)
    if not mine:
        return cell.text
    return normalize_text(" ".join(w.text for w in mine))


def _segment_header_cells(line: Line, char_w: float) -> list[Cell]:
    if not line.words:
        return []
    gap_tol = max(6.0, char_w * 1.15)
    buckets: list[list[Word]] = [[line.words[0]]]
    for prev, cur in zip(line.words, line.words[1:]):
        if cur.x0 - prev.x1 > gap_tol:
            buckets.append([cur])
        else:
            buckets[-1].append(cur)
    cells = []
    for b in buckets:
        cells.append(
            Cell(
                x0=min(w.x0 for w in b),
                y0=min(w.y0 for w in b),
                x1=max(w.x1 for w in b),
                y1=max(w.y1 for w in b),
                text=normalize_text(" ".join(w.text for w in b)),
            )
        )
    return cells


class TickerExtractor:
    """Trich ma chung khoan tu o cot dau tien cua bang."""

    def __init__(self, max_len: int = 6) -> None:
        self.max_len = max_len

    def is_ticker(self, text: str) -> bool:
        t = normalize_text(text)
        if not t or len(t) > self.max_len:
            return False
        return bool(_TICKER_RE.match(t)) and parse_number(t) is None

    def label(self, rows: list[dict], col_key: int) -> str | None:
        hits = sum(1 for r in rows if self.is_ticker(str(r.get(str(col_key)) or "")))
        if hits >= max(2, int(0.6 * len(rows))):
            return "ticker"
        return None
