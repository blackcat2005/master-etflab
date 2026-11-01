#!/usr/bin/env python
"""Retrieval o muc TRANG: BM25(page) vs ColPali/ColSmol.

Vi sao page-level?
  ColPali chi tra ve *trang* (khong chunk, khong o bang). De so sanh cong bang,
  moi he duoc quy ve cung mot don vi: TRANG.

Protocol (giong paper ColPali):
  * BM25 page: text cua trang = toan bo element unstructured tren trang do
    (prose + bang da parse), khong dung parser hinh hoc.
  * Relevance: trang duoc coi la dung neu trung (source_file, page) cua gold.
  * Do 2 pham vi:
      - corpus: xep hang tren TOAN BO 27 trang cua 8 PDF (kho hon, giong paper);
      - file  : chi xep hang trong file cua cau hoi (de so voi eval chunk cu).

Dung:
    python scripts/eval/colpali_retrieval.py --no-colpali
    python scripts/eval/colpali_retrieval.py --model vidore/colSmol-256M
    python scripts/eval/colpali_retrieval.py --model vidore/colpali-v1.2 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
UDATA = DATA / "processed_unstructured"
RAW = DATA / "raw"
GOLD = DATA / "eval" / "gold_qa.jsonl"
GOLD_PAGES = DATA / "eval" / "gold_pages.json"
OUT = DATA / "eval" / "colpali_results.json"

import metric_common as mc

from etflab.uelements import load_unstructured_doc, skip_kind


# ---------------------------------------------------------------------------
# BM25 (k1=1.5, b=0.75) tren mot danh sach van ban.
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, texts: list[str], tok, k1: float = 1.5, b: float = 0.75):
        self.tok = tok
        self.k1, self.b = k1, b
        toks = [tok(t) for t in texts]
        self.n = len(toks)
        self.tf = [Counter(t) for t in toks]
        self.len = [len(t) for t in toks]
        self.avgdl = (sum(self.len) / self.n) if self.n else 1.0
        df: Counter[str] = Counter()
        for t in toks:
            df.update(set(t))
        self.idf = {
            w: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5))
            for w, c in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q = self.tok(query)
        out = [0.0] * self.n
        for i in range(self.n):
            tf, dl = self.tf[i], self.len[i]
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                idf = self.idf.get(w)
                if idf is None:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                s += idf * f * (self.k1 + 1.0) / denom
            out[i] = s
        return out


# ---------------------------------------------------------------------------
# Du lieu: trang, gold.
# ---------------------------------------------------------------------------
def load_pages() -> list[dict]:
    """Text cua trang = toan bo element unstructured tren trang do.

    Baseline BM25 trang la mot he text RAG *cong bang*: no cung nhin thay noi
    dung bang (Table Transformer da parse), chi khac ColPali o cho khong dung anh.
    """
    texts: dict[tuple[str, int], str] = {}
    for path in sorted(UDATA.glob("*.json")):
        doc = load_unstructured_doc(path)
        for el in doc["elements"]:
            if skip_kind(el.kind):
                continue
            key = (doc["source_file"], int(el.page))
            body = el.text.strip()
            if not body:
                continue
            texts[key] = f"{texts[key]}\n{body}".strip() if key in texts else body
    return [{"doc": src, "page": page, "text": text}
            for (src, page), text in texts.items()]


def find_pdf(source_file: str) -> Path | None:
    hits = sorted(RAW.rglob(source_file))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Cham diem.
# ---------------------------------------------------------------------------
def bm25_page_scores(pages: list[dict], gold: list[dict], tok) -> list[list[float]]:
    bm = BM25([p["text"] for p in pages], tok)
    return [bm.scores(g["query"]) for g in gold]


# ---------------------------------------------------------------------------
# ColPali.
# ---------------------------------------------------------------------------
def _pick_model(name: str):
    from colpali_engine.models import (
        ColIdefics3,
        ColIdefics3Processor,
        ColPali,
        ColPaliProcessor,
        ColQwen2,
        ColQwen2Processor,
    )

    low = name.lower()
    if "colsmol" in low or "idefics" in low:
        return ColIdefics3, ColIdefics3Processor
    if "qwen" in low:
        return ColQwen2, ColQwen2Processor
    return ColPali, ColPaliProcessor


def _embed(model, processor, images, batch: int):
    import torch

    embs = []
    for i in range(0, len(images), batch):
        batch_feat = processor.process_images(images[i:i + batch]).to(model.device)
        with torch.no_grad():
            embs.append(model(**batch_feat).float().cpu())
        if (i // batch) % 10 == 0:
            print(f"    ... {i + len(images[i:i + batch])}/{len(images)} trang", file=sys.stderr)
    # Trang cung kho A4 -> cung so token; neu khac (trang ngang) thi giu list.
    if len({tuple(e.shape[1:]) for e in embs}) == 1:
        return torch.cat(embs, dim=0)
    return embs


def _score_query(q, P):
    """MaxSim: voi moi token query lay patch giong nhat, roi cong lai."""
    import torch

    if isinstance(P, list):
        # Moi phan tu: (1, n_patch, dim) — bo batch dim roi moi nhan.
        return torch.stack([(q @ p[0].T).max(dim=1).values.sum() for p in P])
    sim = torch.einsum("qd,pnd->pqn", q, P)          # (pages, nq, nd)
    return sim.max(dim=2).values.sum(dim=1)          # (pages,)


def _embed_queries(model, processor, queries: list[str], batch: int):
    import torch

    embs = []
    for i in range(0, len(queries), batch):
        batch_feat = processor.process_queries(queries[i:i + batch]).to(model.device)
        with torch.no_grad():
            out = model(**batch_feat).float().cpu()
        # Moi batch co the pad khac nhau -> tach tung query, khong cat theo batch.
        embs.extend([out[j] for j in range(out.shape[0])])
    return embs


def colpali_scores(model_name: str, pages: list[dict], gold: list[dict],
                   dpi: int, dtype: str, batch_pages: int, batch_queries: int,
                   threads: int = 0):
    import pymupdf
    import torch
    from PIL import Image

    Model, Processor = _pick_model(model_name)
    torch_dtype = getattr(torch, dtype)
    if threads > 0:
        torch.set_num_threads(threads)
    print(f"[i] nap {model_name} ({dtype}, {torch.get_num_threads()} threads) ...", file=sys.stderr)
    t0 = time.time()
    model = Model.from_pretrained(model_name, dtype=torch_dtype, device_map="cpu")
    model.eval()
    processor = Processor.from_pretrained(model_name)
    print(f"[i] model san sang sau {time.time()-t0:.1f}s", file=sys.stderr)

    # Render tung trang theo thu tu `pages`.
    images = []
    for i, p in enumerate(pages):
        pdf = find_pdf(p["doc"])
        if pdf is None:
            raise SystemExit(f"khong tim thay PDF cho {p['doc']}")
        with pymupdf.open(pdf) as doc:
            pix = doc[p["page"] - 1].get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
        if i == 0:
            print(f"[i] render {pix.width}x{pix.height} @ {dpi} dpi", file=sys.stderr)

    key = hashlib.sha1(
        (f"{model_name}|{dpi}|"
         + "|".join(f"{p['doc']}#{p['page']}" for p in pages)).encode()
    ).hexdigest()[:16]
    cache_path = Path("data/eval/cache") / f"pages_{type(model).__name__}_{key}.pt"
    t1 = time.time()
    if cache_path.exists():
        P = torch.load(cache_path, weights_only=False)
        print(f"[i] dung cache {cache_path} ({time.time()-t1:.1f}s)", file=sys.stderr)
    else:
        P = _embed(model, processor, images, batch_pages)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(P, cache_path)
        print(f"[i] embed {len(images)} trang: {time.time()-t1:.1f}s | "
              f"{'list' if isinstance(P, list) else tuple(P.shape)}", file=sys.stderr)

    t2 = time.time()
    qkey = hashlib.sha1("\n".join(g["query"] for g in gold).encode()).hexdigest()[:12]
    qcache = cache_path.with_name(f"queries_{type(model).__name__}_{key}_{qkey}.pt")
    if qcache.exists():
        Q = torch.load(qcache, weights_only=False)
        print(f"[i] dung cache query ({time.time()-t2:.1f}s)", file=sys.stderr)
    else:
        Q = _embed_queries(model, processor, [g["query"] for g in gold], batch_queries)
        torch.save(Q, qcache)
        print(f"[i] embed {len(Q)} query: {time.time()-t2:.1f}s", file=sys.stderr)

    out = []
    for q in Q:
        out.append(_score_query(q, P))
    return torch.stack(out).tolist()


# ---------------------------------------------------------------------------
# Danh gia.
# ---------------------------------------------------------------------------
def evaluate(scores: list[list[float]], pages: list[dict], gold: list[dict],
             scope: str, ks: list[int], gold_pages: dict[str, list[int]]):
    pos = {(p["doc"], p["page"]): i for i, p in enumerate(pages)}
    ranks, missing = [], 0
    for g, row in zip(gold, scores):
        rel_pages = gold_pages.get(g["qid"], [int(g["page"])])
        rel_idx = {
            pos[(g["source_file"], int(page))]
            for page in rel_pages
            if (g["source_file"], int(page)) in pos
        }
        if not rel_idx:
            missing += 1
            continue
        if scope == "file":
            cand = [i for i, p in enumerate(pages) if p["doc"] == g["source_file"]]
        else:
            cand = list(range(len(pages)))
        rel_in_cand = rel_idx & set(cand)
        if not rel_in_cand:
            missing += 1
            continue
        order = sorted(cand, key=lambda i: (-row[i], i))
        rank = min(order.index(i) + 1 for i in rel_in_cand)
        ranks.append(rank)

    n = len(ranks) or 1
    rec = {k: sum(r <= k for r in ranks) / n for k in ks}
    mrr = sum(1.0 / r for r in ranks) / n
    return rec, mrr, len(ranks), missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval page-level: BM25 vs ColPali")
    ap.add_argument("--model", default="vidore/colSmol-256M")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--batch-pages", type=int, default=1)
    ap.add_argument("--batch-queries", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=8, help="so thread CPU (0 = mac dinh)")
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--limit", type=int, default=0, help="chi chay N cau dau (debug)")
    ap.add_argument("--no-colpali", action="store_true", help="chi chay baseline text")
    args = ap.parse_args()

    ks = sorted(set(args.topk))
    gold = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        gold = gold[: args.limit]
    gold_pages = (json.loads(GOLD_PAGES.read_text(encoding="utf-8"))["pages"]
                  if GOLD_PAGES.exists() else {})

    pages = load_pages()
    n_docs = len({p["doc"] for p in pages})
    print(f"[i] {len(gold)} cau | {n_docs} file | {len(pages)} trang", file=sys.stderr)

    systems: dict[str, dict] = {}

    # --- BM25 theo trang ---
    t0 = time.time()
    s = bm25_page_scores(pages, gold, mc.tok)
    systems["bm25_page"] = {"scores": s, "seconds": time.time() - t0}

    # --- ColPali ---
    if not args.no_colpali:
        t0 = time.time()
        s = colpali_scores(args.model, pages, gold, args.dpi, args.dtype,
                           args.batch_pages, args.batch_queries, args.torch_threads)
        systems[f"colpali:{args.model}"] = {"scores": s, "seconds": time.time() - t0}

    # --- Bang ket qua ---
    rows = []
    for name, info in systems.items():
        entry = {"seconds": round(info["seconds"], 1)}
        for scope in ("corpus", "file"):
            rec, mrr, n, missing = evaluate(
                info["scores"], pages, gold, scope, ks, gold_pages
            )
            entry[scope] = {
                "recall": {str(k): round(v, 4) for k, v in rec.items()},
                "mrr": round(mrr, 4),
                "n": n,
                "missing": missing,
            }
            if scope == "corpus":
                rows.append((name, scope, n, rec, mrr, info["seconds"]))
        rows.append((name, "file", entry["file"]["n"], {
            k: entry["file"]["recall"][str(k)] for k in ks
        }, entry["file"]["mrr"], info["seconds"]))
        systems[name] = entry

    print()
    print(f"{'He thong':<28} {'scope':<7} {'n':>4} " +
          " ".join(f"R@{k:<4}" for k in ks) + f" {'MRR':>6} {'s':>7}")
    print("-" * (28 + 7 + 5 + len(ks) * 7 + 15))
    for name, scope, n, rec, mrr, sec in rows:
        cells = " ".join(f"{rec[k]*100:6.1f}%" for k in ks)
        print(f"{name:<28} {scope:<7} {n:>4} {cells} {mrr:6.3f} {sec:7.1f}")

    payload = {
        "model": None if args.no_colpali else args.model,
        "dtype": args.dtype,
        "dpi": args.dpi,
        "n_gold": len(gold),
        "n_pages": len(pages),
        "n_docs": n_docs,
        "systems": systems,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[i] da ghi {OUT.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
