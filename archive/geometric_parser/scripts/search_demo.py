#!/usr/bin/env python
"""Demo truy van: PDF -> chunks -> hybrid search.

Dung:
    python scripts/search_demo.py
    python scripts/search_demo.py --query "ty trong moi cua BSR"
    python scripts/search_demo.py --dense          # them vector search (can tai model)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).resolve().parents[1]
for _p in (ROOT, ARCHIVE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from etflab_geometric import IngestConfig, build_chunks, process_pdf
from etflab_geometric.retrieval import (
    DocFilter,
    HybridIndex,
    RetrievalConfig,
    infer_doc_filter,
)

DEFAULT_QUERIES = [
    "tỷ trọng mới của BSR",
    "tỷ trọng mới của BSR Q2",
    "thay đổi số lượng cổ phiếu của DGC",
    "2.976,9 là số lượng của cổ phiếu nào",
    "số phiên giao dịch tương đương của MCH",
]


def load_chunks(raw_dir: Path) -> list:
    pdfs = sorted(raw_dir.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"[!] khong tim thay PDF nao trong {raw_dir}")
    chunks: list = []
    for pdf in pdfs:
        result = process_pdf(pdf, IngestConfig())
        chunks.extend(build_chunks(result))
        for w in result.warnings:
            print(f"[!] {pdf.name}: {w}", file=sys.stderr)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo hybrid retrieval tren ETF corpus")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--query", action="append", help="cau hoi (lap lai de nhieu cau)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--dense", action="store_true",
                    help="bat them vector search (can tai embedding model)")
    ap.add_argument("--no-filter", action="store_true",
                    help="khong tu suy ra ky/chi so tu cau hoi")
    ap.add_argument("--period", help="ep ky, vd Q2 hoac Q2/2026")
    ap.add_argument("--index", help="ep chi so, vd VN30")
    ap.add_argument("--doc-type", help="ep loai tai lieu: review | forecast | charter")
    args = ap.parse_args()

    print("Dang doc PDF va tao chunks...", file=sys.stderr)
    chunks = load_chunks(args.raw)
    cfg = RetrievalConfig(use_dense=args.dense, top_k=args.top_k)
    if args.dense:
        print("Dang tai embedding model (lan dau se rat cham)...", file=sys.stderr)
    index = HybridIndex(chunks, cfg)
    if args.dense and index.dense_error:
        print(f"[!] vector search bat len loi: {index.dense_error}", file=sys.stderr)
    print(f"San sang {len(chunks)} chunks.", file=sys.stderr)

    forced = DocFilter(
        doc_type=args.doc_type, index=args.index, period=args.period
    )
    for query in (args.query or DEFAULT_QUERIES):
        # Co "ep" thi ghi de; khong thi tu suy ra tu cau hoi.
        inferred = DocFilter() if args.no_filter else infer_doc_filter(query)
        doc_filter = DocFilter(
            doc_type=forced.doc_type or inferred.doc_type,
            index=forced.index or inferred.index,
            quarter=forced.quarter or inferred.quarter,
            year=forced.year or inferred.year,
            period=forced.period or inferred.period,
            source_file=forced.source_file or inferred.source_file,
        )
        print(f"\n{'=' * 78}\n>>> {query}")
        if doc_filter.is_empty():
            print("  (khong co rang buoc tai lieu)")
        else:
            print(f"  [loc] {doc_filter.describe()}")
        hits = index.search(query, doc_filter=doc_filter)
        if not hits:
            print("  (khong co ket qua trong pham vi loc)")
        for n, hit in enumerate(hits, 1):
            chunk = hit.chunk
            where = f"{chunk.source_file} | trang {chunk.page}"
            if chunk.table_id:
                where += f" | {chunk.table_id}"
            print(f"  {n}. [{chunk.kind}] score={hit.score:.4f}  ({where})")
            print(f"     {chunk.text[:300]}")
            if hit.parent_text:
                print(f"     [cha] {hit.parent_text[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())