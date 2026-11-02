#!/usr/bin/env python
"""Parse PDF bang ``unstructured`` (hi_res + Table Transformer) -> cache JSON.

Chay trong venv rieng co ``unstructured`` (khong can src/pymupdf):

    C:/_venvs/unstr/Scripts/python.exe scripts/ingest/parse_unstructured.py
    C:/_venvs/unstr/Scripts/python.exe scripts/ingest/parse_unstructured.py --only VNDiamond --force

Vi sao khong parse HTML o day: buoc nay chi can layout + table structure.
Viec chuan hoa HTML -> luoi nam trong ``src/uelements.py`` (stdlib), de
ca ba chien luoc chunking dung chung mot cach doc bang.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed_unstructured"

# Cung bo tai lieu voi gold QA (7 PDF co bang so).
INCLUDE_HINTS = ("du_bao", "thay_doi_thanh_phan")

KIND_MAP = {
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
}


def eval_docs() -> list[str]:
    """Ten file (source_file) cua 7 PDF nam trong bo danh gia."""
    names: list[str] = []
    for path in sorted(PROCESSED.glob("*.json")):
        if not any(h in path.stem.lower() for h in INCLUDE_HINTS):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        names.append(data["source_file"])
    return names


def find_pdf(source_file: str) -> Path | None:
    hits = sorted(RAW.rglob(source_file))
    return hits[0] if hits else None


def parse_one(pdf: Path, source_file: str) -> dict:
    from unstructured.partition.pdf import partition_pdf

    t0 = time.time()
    elements = partition_pdf(
        filename=str(pdf),
        strategy="hi_res",
        infer_table_structure=True,
        languages=["vie"],
        extract_images_in_pdf=False,
        include_page_breaks=False,
    )
    out_elements: list[dict] = []
    n_tables = 0
    pages: set[int] = set()
    for el in elements:
        kind = KIND_MAP.get(getattr(el, "category", type(el).__name__),
                            "narrative")
        page = int(getattr(el.metadata, "page_number", 1) or 1)
        pages.add(page)
        text = (el.text or "").strip()
        if kind == "table":
            html = getattr(el.metadata, "text_as_html", None) or ""
            if not html.strip():
                # Khong co HTML thi khong the dung row/JSON — bo qua.
                continue
            n_tables += 1
            out_elements.append({
                "kind": "table", "page": page,
                "table_id": f"{Path(source_file).stem}#ut{n_tables}",
                "html": html, "text": text,
            })
            continue
        if text:
            out_elements.append({"kind": kind, "page": page, "text": text})

    return {
        "source_file": source_file,
        "parser": "unstructured hi_res + infer_table_structure",
        "n_pages": max(pages) if pages else 0,
        "n_tables": n_tables,
        "seconds": round(time.time() - t0, 1),
        "elements": out_elements,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse PDF bang unstructured")
    ap.add_argument("--only", default=None,
                    help="chi parse file co chuoi nay trong ten")
    ap.add_argument("--force", action="store_true", help="parse lai du da co cache")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    names = eval_docs()
    if args.only:
        names = [n for n in names if args.only.lower() in n.lower()]
    print(f"[i] {len(names)} tai lieu, out={OUT.relative_to(ROOT)}", file=sys.stderr)

    for i, source_file in enumerate(names, 1):
        dest = OUT / f"{Path(source_file).stem}.json"
        if dest.exists() and not args.force:
            print(f"[=] {i}/{len(names)} {source_file} (da co)", file=sys.stderr)
            continue
        pdf = find_pdf(source_file)
        if pdf is None:
            print(f"[!] {i}/{len(names)} khong thay PDF: {source_file}",
                  file=sys.stderr)
            continue
        if args.dry_run:
            print(f"[dry] {pdf}", file=sys.stderr)
            continue
        print(f"[>] {i}/{len(names)} {pdf.name}", file=sys.stderr)
        try:
            payload = parse_one(pdf, source_file)
        except Exception as exc:  # noqa: BLE001 - giu vet loi, khong dung ca batch
            print(f"[x] {pdf.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"[<] {payload['n_pages']} trang, {payload['n_tables']} bang, "
              f"{payload['seconds']}s -> {dest.name}", file=sys.stderr)

    print("[i] xong", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
