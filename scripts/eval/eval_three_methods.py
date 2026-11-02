#!/usr/bin/env python
"""Danh gia 3 cach chunking tren cung dau vao ``unstructured``.

So sanh:
  * ``chipper``        — 4 quy tac paper arXiv:2402.05131 (bang giu nguyen).
  * ``chipper_prefix`` — them metadata "Prefix & Table Description".
  * ``dual``           — dual-granularity (Peng et al.): truy hoi khoi nho,
                         xep hang lai bang trung binh khoi nho + khoi lon.
  * ``row_json``       — moi dong bang la mot JSON tu chua.

Hai muc do:
  * CHUNK level — trong pham vi 1 file, de so cac cach chunking voi nhau.
  * PAGE level  — max-pool diem chunk len trang, xep hang 27 trang, de so
    voi baseline BM25 theo trang.

Metric chunk-level la CONTENT-BASED: chunk dung neu chua ca entity (ma CK)
va value (so da chuan hoa) — khong phu thuoc chunk_id. Kem theo:
  * ``coverage``  — ty le cau co it nhat 1 chunk chua dap an (tran cua recall,
                    phan anh mat mat do parser/OCR chu khong phai do ranking);
  * ``chars@1st`` — do dai chunk dau tien chua dap an (chi phi ngu canh).

Dung:
    python scripts/eval/eval_three_methods.py
    python scripts/eval/eval_three_methods.py --no-dense --limit 30
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import metric_common as mc

from etflab.chunking_methods import (
    MethodConfig,
    chipper_chunks,
    dual_granularity_chunks,
    row_json_chunks,
)
from etflab.uelements import load_unstructured_doc

UDATA = ROOT / "data" / "processed_unstructured"
GOLD = ROOT / "data" / "eval" / "gold_qa.jsonl"
GOLD_PAGES = ROOT / "data" / "eval" / "gold_pages.json"
OUT = ROOT / "data" / "eval" / "three_methods.json"

Match = mc.matches
tok = mc.tok

# Cache chunk dual theo (file, cfg, embedder): 3 bien the dual dung chung
# ket qua, neu khong se embed lai toan bo cau 3 lan.
_DUAL_CACHE: dict[tuple[str, int, int], tuple[list, list]] = {}


# ---------------------------------------------------------------------------
# BM25 (tra ve diem tho, khong sap xep) — cung cong thuc voi eval_chunking.
# ---------------------------------------------------------------------------
def bm25_scores(texts: list[str], query: str) -> list[float]:
    toks = [tok(t) for t in texts]
    q = tok(query)
    n = len(toks)
    if n == 0:
        return []
    tf = [Counter(t) for t in toks]
    length = [len(t) for t in toks]
    avgdl = sum(length) / n or 1.0
    df = Counter()
    for t in toks:
        df.update(set(t))
    idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    out = []
    for i in range(n):
        s = 0.0
        for w in q:
            if w not in idf or not tf[i].get(w):
                continue
            denom = tf[i][w] + 1.5 * (1 - 0.75 + 0.75 * length[i] / avgdl)
            s += idf[w] * tf[i][w] * 2.5 / denom
        out.append(s)
    return out


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# ---------------------------------------------------------------------------
# Embedder nho (MiniLM multilingual) — chi de (a) gop cau theo cosine,
# (b) bien the dense cua dual-granularity.
# ---------------------------------------------------------------------------
class Embedder:
    """e5-small multilingual (384d) — da co san trong cache HF cua may.

    Paper dual-granularity dung bge-large-zh; o day dung e5-small vi (a) da
    ho tro tieng Viet, (b) chay CPU, (c) khong phai tai them model. Giu
    nguyen theta=0.9 nhu paper va ghi ro day la surrogate.
    """

    name = "intfloat/multilingual-e5-small"

    def __init__(self, threads: int = 8) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if threads > 0:
            torch.set_num_threads(threads)
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name)
        self.model.eval()

    def __call__(self, texts: list[str], kind: str = "passage") -> list[list[float]]:
        torch = self._torch
        prefix = "query: " if kind == "query" else "passage: "
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = self.tok([prefix + t for t in texts[i:i + 32]],
                             padding=True, truncation=True,
                             max_length=256, return_tensors="pt")
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vec = torch.nn.functional.normalize(vec, dim=-1)
            out.extend(vec.tolist())
        return out


# ---------------------------------------------------------------------------
# Doc + chien luoc.
# ---------------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    units: dict[str, list] = field(default_factory=dict)      # don vi truy hoi
    context: dict[str, dict] = field(default_factory=dict)    # id -> chunk tra ve
    mode: str = "bm25"                                        # bm25 | dual_bm25 | dual_dense
    parent_of: dict[str, dict] = field(default_factory=dict)  # id -> large chunk
    small_of: dict[str, list] = field(default_factory=dict)   # file -> small (dual)
    dense_small: dict[str, list[list[float]]] = field(default_factory=dict)
    dense_large: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    qcache: dict[str, list[float]] = field(default_factory=dict)


def load_udocs() -> list[dict]:
    docs = []
    for path in sorted(UDATA.glob("*.json")):
        docs.append(load_unstructured_doc(path))
    return docs


def build_strategy(name: str, udocs: list[dict], cfg: MethodConfig,
                   embed: Embedder | None) -> Strategy:
    st = Strategy(name=name)
    for doc in udocs:
        src, els = doc["source_file"], doc["elements"]
        if name in ("chipper", "chipper_prefix"):
            chunks = chipper_chunks(doc, els, cfg, add_prefix=name.endswith("prefix"))
            st.units[src] = chunks
            st.context.update({c.chunk_id: c for c in chunks})
        elif name == "row_json":
            chunks = row_json_chunks(doc, els, cfg)
            st.units[src] = chunks
            st.context.update({c.chunk_id: c for c in chunks})
        elif name in ("dual", "dual_small", "dual_dense"):
            key = (src, id(cfg), id(embed))
            if key not in _DUAL_CACHE:
                _DUAL_CACHE[key] = dual_granularity_chunks(doc, els, cfg, embed=embed)
            small, large = _DUAL_CACHE[key]
            st.units[src] = small
            st.small_of[src] = small
            st.parent_of.update({c.chunk_id: c for c in large})
            if name == "dual_small":
                st.context.update({c.chunk_id: c for c in small})
                st.mode = "bm25"
            else:
                st.context.update({c.chunk_id: c for c in large})
                st.mode = "dual_bm25" if name == "dual" else "dual_dense"
        else:
            raise ValueError(name)
    return st


def rank(st: Strategy, src: str, query: str,
         embed: Embedder | None) -> tuple[list[int], list[float]]:
    """Tra ve (thu tu chi so, diem) cho cac don vi cua 1 file."""
    units = st.units.get(src, [])
    if not units:
        return [], []
    texts = [c.sparse_text or c.text for c in units]
    if st.mode == "bm25":
        scores = bm25_scores(texts, query)
    elif st.mode == "dual_bm25":
        small = bm25_scores(texts, query)
        parents = {st.parent_of[c.parent_id].chunk_id: st.parent_of[c.parent_id]
                   for c in units if c.parent_id in st.parent_of}
        parent_list = list(parents.values())
        idx = {p.chunk_id: i for i, p in enumerate(parent_list)}
        lscore = bm25_scores([p.sparse_text or p.text for p in parent_list], query)
        zs, zl = _minmax(small), _minmax(lscore)
        scores = [
            0.5 * (zs[i] + zl[idx[st.parent_of[c.parent_id].chunk_id]])
            if c.parent_id in st.parent_of else zs[i]
            for i, c in enumerate(units)
        ]
    elif st.mode == "dual_dense":
        assert embed is not None
        if query not in st.qcache:
            st.qcache[query] = embed([query], kind="query")[0]
        qv = st.qcache[query]
        if src not in st.dense_small:
            st.dense_small[src] = embed(texts)
            parents = {st.parent_of[c.parent_id].chunk_id: st.parent_of[c.parent_id]
                       for c in units if c.parent_id in st.parent_of}
            parent_list = list(parents.values())
            st.dense_large[src] = {
                p.chunk_id: v for p, v in
                zip(parent_list, embed([p.text for p in parent_list]))
            }
        sv = st.dense_small[src]
        pv = st.dense_large[src]
        scores = []
        for i, c in enumerate(units):
            dot_small = sum(a * b for a, b in zip(qv, sv[i]))
            parent = st.parent_of.get(c.parent_id)
            dot_large = (sum(a * b for a, b in zip(qv, pv[parent.chunk_id]))
                         if parent is not None else dot_small)
            scores.append(0.5 * (dot_small + dot_large))
    else:
        raise ValueError(st.mode)
    order = sorted(range(len(units)), key=lambda i: -scores[i])
    return order, scores


# ---------------------------------------------------------------------------
# Cham diem.
# ---------------------------------------------------------------------------
def eval_chunk_level(st: Strategy, gold: list[dict], ks: list[int],
                     embed: Embedder | None) -> dict:
    maxk = max(ks)
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    covered = 0
    first_hit_chars: list[int] = []
    all_units = [c for units in st.units.values() for c in units]
    n_units = len(all_units)
    sizes = [len(c.text) for c in all_units]
    per_q = []

    for g in gold:
        src = g["source_file"]
        units = st.units.get(src, [])
        order, _ = rank(st, src, g["query"], embed)

        def ctx_of(i: int, _units=units):
            c = _units[i]
            return st.context.get(c.chunk_id, c)

        if any(Match(ctx_of(i), g["entity"], g["value_norm"])
               for i in range(len(units))):
            covered += 1

        rr = 0.0
        first_pos = None
        for pos, i in enumerate(order, 1):
            if pos > maxk:
                break
            ctx = ctx_of(i)
            if not Match(ctx, g["entity"], g["value_norm"]):
                continue
            for k in ks:
                if pos <= k:
                    hits[k] += 1
            first_pos = pos
            rr = 1.0 / pos
            first_hit_chars.append(len(ctx.text))
            break
        rr_sum += rr
        per_q.append({
            "qid": g["qid"], "rank": first_pos, "rr": rr,
        })

    n = len(gold)
    return {
        "n": n,
        "n_units": n_units,
        "avg_chars": round(statistics.mean(sizes), 1) if sizes else 0,
        "p95_chars": (sorted(sizes)[int(0.95 * (len(sizes) - 1))] if sizes else 0),
        "recall": {str(k): round(hits[k] / n, 4) for k in ks},
        "mrr": round(rr_sum / n, 4),
        "coverage": round(covered / n, 4),
        "chars_at_first_hit": (round(statistics.mean(first_hit_chars), 1)
                               if first_hit_chars else None),
        "per_q": per_q,
    }


def eval_page_level(st: Strategy, gold: list[dict], pages: list[dict],
                    ks: list[int], embed: Embedder | None,
                    gold_pages: dict[str, list[int]]) -> dict:
    """Max-pool diem don vi len trang, xep hang toan bo 27 trang.

    ``gold_pages`` la tap trang dung cho tung cau (xem
    ``build_gold_pages.py``): bang trai nhieu trang thi moi trang chua dap an
    deu tinh, rank = trang dung dau tien.
    """
    page_index = {(p["doc"], int(p["page"])): i for i, p in enumerate(pages)}
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    skipped = 0

    for g in gold:
        rel_pages = gold_pages.get(g["qid"], [int(g["page"])])
        rel_idx = {
            page_index[(g["source_file"], int(page))]
            for page in rel_pages
            if (g["source_file"], int(page)) in page_index
        }
        if not rel_idx:
            skipped += 1
            continue
        order, scores = rank(st, g["source_file"], g["query"], embed)
        units = st.units.get(g["source_file"], [])
        by_page: dict[int, float] = {}
        for i in order:
            pg = int(units[i].page)
            by_page[pg] = max(by_page.get(pg, -1e9), scores[i])
        page_scores = [
            by_page.get(int(p["page"]), -1e9) if p["doc"] == g["source_file"]
            else -1e9
            for p in pages
        ]
        order_pages = sorted(range(len(pages)), key=lambda j: -page_scores[j])
        rank_pos = min(order_pages.index(i) + 1 for i in rel_idx)
        rr_sum += 1.0 / rank_pos
        for k in ks:
            if rank_pos <= k:
                hits[k] += 1

    n = len(gold) - skipped
    return {
        "n": n,
        "skipped": skipped,
        "recall": {str(k): round(hits[k] / n, 4) if n else 0 for k in ks},
        "mrr": round(rr_sum / n, 4) if n else 0,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="3 chunking methods tren unstructured")
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--gold", default=str(GOLD),
                    help="gold jsonl (vd data/gold/gold_qa_v2.jsonl)")
    ap.add_argument("--exclude-hard", action="store_true",
                    help="bo cau hoi mo ho (variant=hard, khong co ky/chi so)")
    args = ap.parse_args()
    ks = sorted(set(args.topk))

    gold_path = Path(args.gold)
    gold = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.exclude_hard:
        gold = [g for g in gold if not g.get("hard")]
    if args.limit:
        gold = gold[: args.limit]

    udocs = load_udocs()
    if not udocs:
        print(f"[x] chua co cache trong {UDATA}; chay "
              f"scripts/ingest/parse_unstructured.py truoc", file=sys.stderr)
        return 2
    print(f"[i] {len(gold)} cau | {len(udocs)} tai lieu unstructured | "
          f"{sum(len(d['elements']) for d in udocs)} phan tu", file=sys.stderr)

    cfg = MethodConfig()
    t0 = time.time()
    embed = None if args.no_dense else Embedder(args.threads)
    if embed is not None:
        print(f"[i] embedder {embed.name} ({time.time()-t0:.1f}s)", file=sys.stderr)

    # Trang corpus = 27 trang co phan tu unstructured (de so voi BM25 trang).
    pages = []
    seen_pages: set[tuple[str, int]] = set()
    for d in udocs:
        for el in d["elements"]:
            key = (d["source_file"], int(el.page))
            if key not in seen_pages:
                seen_pages.add(key)
                pages.append({"doc": key[0], "page": key[1]})

    names = ["chipper", "chipper_prefix", "row_json", "dual", "dual_small"]
    if embed is not None:
        names.append("dual_dense")

    if all("gold_pages" in g for g in gold):
        gold_pages = {g["qid"]: g["gold_pages"] for g in gold}
    elif GOLD_PAGES.exists():
        gold_pages = json.loads(GOLD_PAGES.read_text(encoding="utf-8"))["pages"]
    else:
        gold_pages = {}
        print("[!] thieu gold_pages.json — dung page goc cua gold_qa", file=sys.stderr)

    results: dict[str, dict] = {}
    for name in names:
        t = time.time()
        st = build_strategy(name, udocs, cfg, embed)
        chunk_metrics = eval_chunk_level(st, gold, ks, embed)
        page_metrics = eval_page_level(st, gold, pages, ks, embed, gold_pages)
        results[name] = {
            "seconds": round(time.time() - t, 1),
            "chunk": chunk_metrics,
            "page": page_metrics,
        }
        print(f"[i] {name}: xong sau {time.time()-t:.1f}s", file=sys.stderr)

    # Bang chunk-level.
    print(f"\n{'Chien luoc':<16} {'#chunk':>7} {'avg':>6} {'cover':>6} "
          + " ".join(f"R@{k:<4}" for k in ks) + "  MRR   chars@1")
    print("-" * 82)
    for name in names:
        m = results[name]["chunk"]
        rec = " ".join(f"{m['recall'][str(k)]*100:6.1f}%" for k in ks)
        chars = m["chars_at_first_hit"]
        print(f"{name:<16} {m['n_units']:>7} {m['avg_chars']:>6.0f} "
              f"{m['coverage']*100:5.1f}% {rec}  {m['mrr']:.3f} "
              f"{chars if chars is not None else '-':>9}")

    # Bang page-level.
    print(f"\n{'Chien luoc':<16} " + " ".join(f"R@{k:<4}" for k in ks) + "  MRR")
    print("-" * 60)
    for name in names:
        m = results[name]["page"]
        rec = " ".join(f"{m['recall'][str(k)]*100:6.1f}%" for k in ks)
        print(f"{name:<16} {rec}  {m['mrr']:.3f}")

    payload = {
        "config": {
            "chipper_target": cfg.chipper_target,
            "large_target": cfg.large_target,
            "small_theta": cfg.small_theta,
            "embedder": None if embed is None else embed.name,
            "parser": "unstructured hi_res + infer_table_structure",
            "n_gold": len(gold),
            "n_pages": len(pages),
        },
        "systems": results,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\n[i] da ghi {Path(args.out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
