"""Orchestration: PDF -> JSON (bang + van ban)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .geometry import build_lines, group_cells, median_char_width, words_from_page
from .tables import Table, TableConfig, TableExtractor
from .vnnum import normalize_text

__all__ = ["DocumentResult", "IngestConfig", "PageText", "process_pdf"]

_SENTENCE_END = (".", "?", "!", ":", ";")
_SECTION_RE = re.compile(r"^\s*(\d+\.|[IVX]+\.|[A-ZĐ]{2,6}\.)\s+\S")
_FOOTER_RE = re.compile(r"(visit\s|trang\s+\d+|https?://|www\.)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class IngestConfig:
    table: TableConfig = field(default_factory=TableConfig)
    drop_footer: bool = True
    footer_margin_ratio: float = 0.94
    caption_max_lines: int = 4


@dataclass(slots=True)
class PageText:
    page: int
    blocks: list[dict]

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


@dataclass(slots=True)
class DocumentResult:
    source_file: str
    n_pages: int
    tables: list[Table]
    pages_text: list[PageText]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "n_pages": self.n_pages,
            "n_tables": len(self.tables),
            "tables": [t.to_dict() for t in self.tables],
            "pages_text": [
                {"page": p.page, "blocks": p.blocks} for p in self.pages_text
            ],
            "warnings": self.warnings,
        }


def _find_caption(lines, region_top: float, pattern: re.Pattern[str], max_lines: int) -> str:
    above = [ln for ln in lines if ln.y1 <= region_top]
    hit: str | None = None
    for i in range(len(above) - 1, -1, -1):
        if pattern.match(above[i].text):
            hit = above[i].text
            for j in range(i + 1, min(i + max_lines, len(above))):
                nxt = above[j]
                if _SECTION_RE.match(nxt.text) or pattern.match(nxt.text):
                    break
                hit = f"{hit} {nxt.text}"
                if nxt.text.endswith(_SENTENCE_END):
                    break
            break
    return normalize_text(hit) if hit else ""


def _prose_blocks(lines, table_ys: list[tuple[float, float]], cfg: IngestConfig, page_h: float) -> list[dict]:
    blocks: list[dict] = []
    buf: list = []

    def flush() -> None:
        if not buf:
            return
        text = normalize_text(" ".join(w.text for w in buf))
        if text:
            blocks.append({
                "text": text,
                "y0": min(w.y0 for w in buf),
                "y1": max(w.y1 for w in buf),
            })
        buf.clear()

    for line in lines:
        if not line.has_text:
            continue
        if any(y0 - 1 <= line.cy <= y1 + 1 for y0, y1 in table_ys):
            flush()
            continue
        if cfg.drop_footer and line.y0 > page_h * cfg.footer_margin_ratio:
            continue
        if _FOOTER_RE.search(line.text):
            flush()
            continue
        if buf and abs(buf[0].y1 - line.y0) > max(6.0, buf[0].height * 1.9):
            flush()
        buf.extend(line.words)
    flush()
    return blocks


def process_pdf(path: str | Path, cfg: IngestConfig | None = None) -> DocumentResult:
    cfg = cfg or IngestConfig()
    path = Path(path)
    doc = pymupdf.open(path)
    extractor = TableExtractor(cfg.table)
    caption_re = re.compile(cfg.table.caption_pattern, re.IGNORECASE)

    tables: list[Table] = []
    pages_text: list[PageText] = []
    warnings: list[str] = []
    empty_pages: list[int] = []

    for pno, page in enumerate(doc, start=1):
        words = words_from_page(page)
        if not words:
            empty_pages.append(pno)
        char_w = median_char_width(words)
        lines = build_lines(words, cfg.table.y_tol)
        for line in lines:
            group_cells(line, char_w)

        page_tables = extractor.extract(words, pno)
        for t in page_tables:
            t.source_file = path.name
            t.page = pno
            t.page_span = [pno]
            t.caption = _find_caption(lines, t.region_y[0], caption_re, cfg.caption_max_lines)
        tables = _merge_continuations(tables, page_tables)
        pages_text.append(
            PageText(
                page=pno,
                blocks=_prose_blocks(lines, [t.region_y for t in page_tables], cfg, page.rect.height),
            )
        )

    doc.close()
    tables = _assign_ids(tables)
    if len(empty_pages) == len(pages_text):
        warnings.append(
            f"KHONG CO LOP TEXT: {len(empty_pages)}/{len(pages_text)} trang rong — "
            "PDF dang scan, can OCR truoc khi ingestion."
        )
    elif empty_pages:
        shown = ", ".join(str(p) for p in empty_pages[:10])
        more = "..." if len(empty_pages) > 10 else ""
        warnings.append(
            f"{len(empty_pages)}/{len(pages_text)} trang khong co lop text "
            f"(trang {shown}{more}) — can OCR."
        )
    return DocumentResult(
        source_file=path.name,
        n_pages=len(pages_text),
        tables=tables,
        pages_text=pages_text,
        warnings=warnings,
    )


def _merge_continuations(previous: list[Table], incoming: list[Table]) -> list[Table]:
    """Ghep bang bi cat giua 2 trang. Khop bang chu ky cot + vi tri ngang."""
    for table in incoming:
        target = _continuation_target(previous, table)
        if target is None:
            previous.append(table)
            continue
        target.rows.extend(table.rows)
        target.n_rows = len(target.rows)
        target.page_span = sorted(set(target.page_span + table.page_span))
        target.region_y = (target.region_y[0], table.region_y[1])
        target.x_span = (
            min(target.x_span[0], table.x_span[0]),
            max(target.x_span[1], table.x_span[1]),
        )
        if not target.caption:
            target.caption = table.caption
    return previous


def _continuation_target(previous: list[Table], table: Table) -> Table | None:
    sig = table.signature
    if not sig or sum(1 for s in sig if s) < 2:
        return None
    cx = (table.x_span[0] + table.x_span[1]) / 2.0
    best: Table | None = None
    best_d = 1e18
    for cand in previous:
        if cand.signature != sig:
            continue
        if (
            cand.page == table.page
            and abs(cand.region_y[1] - table.region_y[0]) >= 120.0
        ):
            continue
        if (
            min(cand.x_span[1], table.x_span[1])
            - max(cand.x_span[0], table.x_span[0])
            <= 0
        ):
            continue
        d = abs((cand.x_span[0] + cand.x_span[1]) / 2.0 - cx)
        if d < best_d:
            best, best_d = cand, d
    return best


def _assign_ids(tables: list[Table]) -> list[Table]:
    stem_counts: dict[str, int] = {}
    for table in tables:
        stem = table.source_file.rsplit(".", 1)[0][:48]
        idx = stem_counts.get(stem, 0) + 1
        stem_counts[stem] = idx
        safe = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
        table.table_id = f"{safe}__t{idx:02d}"
    return tables
