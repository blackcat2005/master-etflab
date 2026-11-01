#!/usr/bin/env python
"""Xây "trang vàng" độc lập parser cho bộ gold QA.

Vì sao cần: bảng tài chính trải qua nhiều trang. ``gold_qa.jsonl`` ghi
``page`` = trang *bắt đầu* của bảng, nên các dòng nằm ở trang tiếp theo bị
chấm sai nếu hệ retrieval trả về đúng trang chứa dòng đó. File này tìm, cho
từng câu hỏi, **mọi trang PDF** chứa cả entity (mã CK) và value (số đã chuẩn
hoá) bằng text layer của PyMuPDF — không dùng parser bảng nào.

Kết quả: ``data/eval/gold_pages.json``
    {"method": "...", "pages": {qid: [page, ...]}, "fallbacks": [...]}

Dùng:
    python scripts/eval/build_gold_pages.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import metric_common as mc

GOLD = ROOT / "data" / "eval" / "gold_qa.jsonl"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "eval" / "gold_pages.json"


def main() -> int:
    import pymupdf

    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    pdfs = {p.name: p for p in RAW.rglob("*.pdf")}
    texts: dict[str, list[str]] = {}
    pages: dict[str, list[int]] = {}
    fallbacks: list[dict] = []

    for g in gold:
        src = g["source_file"]
        if src not in texts:
            with pymupdf.open(pdfs[src]) as doc:
                texts[src] = [
                    mc.strip_accents(page.get_text()).lower() for page in doc
                ]
        entity = mc.strip_accents(g["entity"]).lower()
        hits = []
        for page_no, text in enumerate(texts[src], 1):
            if not entity or entity not in text:
                continue
            if g["value_norm"] is not None:
                target = round(float(g["value_norm"]), 6)
                if target not in mc.numbers_in(text):
                    continue
            hits.append(page_no)
        if not hits:
            hits = [int(g["page"])]
            fallbacks.append({"qid": g["qid"], "page": hits[0]})
        pages[g["qid"]] = hits

    OUT.write_text(json.dumps({
        "method": "pymupdf text layer; entity + normalized value; "
                  "all matching pages per question",
        "n_gold": len(gold),
        "n_multi_page": sum(1 for v in pages.values() if len(v) > 1),
        "pages": pages,
        "fallbacks": fallbacks,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    n_multi = sum(1 for v in pages.values() if len(v) > 1)
    print(f"[i] {len(gold)} câu | {n_multi} câu có nhiều trang đúng | "
          f"{len(fallbacks)} fallback")
    print(f"[i] đã ghi {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
