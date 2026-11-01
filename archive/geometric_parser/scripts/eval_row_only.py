#!/usr/bin/env python
"""Ablation: CHI dung ``table_row`` cua etflab (bo table_card + prose).

Vi sao can: ket qua cu (``eval_chunking.py``) do *ca bo* chunk
(row + card + prose) va dat R@1 73,1%. Khi tach rieng, chi row dat 87,7%.
Nghia la card/prose khong "bo sung" ma **pha loang** xep hang BM25 cho cau
hoi o bang. Script nay co lap ket qua do de lam moc so sanh cho:

  * ``row_json`` (unstructured, cung y tuong 1 dong = 1 chunk)
  * ``chipper`` / ``dual`` (unstructured, chunk lon / 2 tang)

Dung:
    python scripts/eval/eval_row_only.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eval_chunking as ec
from colpali_retrieval import BM25

GOLD = ROOT / "data" / "eval" / "gold_qa.jsonl"
OUT = ROOT / "data" / "eval" / "row_only.json"


def main() -> int:
    docs = ec._load_docs()
    gold = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    ks = [1, 3, 5, 10]

    chunks = {d["source_file"]: [c for t in d["tables"] for c in ec.row_chunks(t)]
              for d in docs}
    all_chunks = [c for cs in chunks.values() for c in cs]

    # Chunk-level: tai su dung dung evaluate cua eval_chunking (content-based).
    res = ec.evaluate(chunks, gold, ks)
    n = len(gold)
    chunk_metrics = {
        "n_units": len(all_chunks),
        "avg_chars": round(statistics.mean(len(c.text) for c in all_chunks), 1),
        "recall": {str(k): round(sum(r.recall[k] for r in res) / n, 4) for k in ks},
        "mrr": round(sum(r.rr for r in res) / n, 4),
        "coverage": round(sum(
            1 for g in gold
            if any(ec._matches(c, g["entity"], g["value_norm"])
                   for c in chunks.get(g["source_file"], []))
        ) / n, 4),
    }

    # Page-level: max-pool diem BM25 len 27 trang.
    pages = [{"doc": d["source_file"], "page": int(p["page"])}
             for d in docs for p in d["pages_text"]]
    idx = {(p["doc"], p["page"]): i for i, p in enumerate(pages)}
    bm = {src: BM25([(c.sparse_text or c.text) for c in cs], ec._tok)
          for src, cs in chunks.items()}
    hits = {k: 0 for k in ks}
    rr = 0.0
    for g in gold:
        gi = idx[(g["source_file"], int(g["page"]))]
        sc = bm[g["source_file"]].scores(g["query"])
        by_page: dict[int, float] = {}
        for c, s in zip(chunks[g["source_file"]], sc):
            by_page[int(c.page)] = max(by_page.get(int(c.page), -1e9), s)
        scores = [by_page.get(p["page"], -1e9) if p["doc"] == g["source_file"]
                  else -1e9 for p in pages]
        order = sorted(range(len(pages)), key=lambda j: -scores[j])
        pos = order.index(gi) + 1
        rr += 1.0 / pos
        for k in ks:
            if pos <= k:
                hits[k] += 1
    page_metrics = {
        "n": n,
        "recall": {str(k): round(hits[k] / n, 4) for k in ks},
        "mrr": round(rr / n, 4),
    }

    print(f"etflab row-only: {chunk_metrics['n_units']} chunk, "
          f"avg {chunk_metrics['avg_chars']:.0f} ky tu, "
          f"coverage {chunk_metrics['coverage']*100:.1f}%")
    print("chunk-level " + "  ".join(
        f"R@{k} {chunk_metrics['recall'][str(k)]*100:.1f}%" for k in ks)
        + f"  MRR {chunk_metrics['mrr']:.3f}")
    print("page-level  " + "  ".join(
        f"R@{k} {page_metrics['recall'][str(k)]*100:.1f}%" for k in ks)
        + f"  MRR {page_metrics['mrr']:.3f}")

    OUT.write_text(json.dumps({"chunk": chunk_metrics, "page": page_metrics},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[i] da ghi {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
