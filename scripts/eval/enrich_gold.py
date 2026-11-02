#!/usr/bin/env python
"""Làm giàu gold QA: nhiều câu hơn, nhiều kiểu câu hỏi hơn, có kiểm chứng.

Gold cũ (`data/eval/gold_qa.jsonl`, 130 câu) chỉ lấy ~10 ô/bảng và một mẫu câu
hỏi duy nhất. Script này:

1. Lấy TOÀN BỘ ô số từ bảng đã parse (parser hình học cũ, dùng làm nguồn gold
   độc lập với `unstructured`) — hơn 1.800 ô.
2. Lấy mẫu phân tầng theo bảng (`--per-table`), luôn giữ 130 câu cũ.
3. Thêm **cặp đối chứng theo kỳ**: cùng (mã, cột, chỉ số), khác kỳ.
   Đo được lỗi "trả nhầm kỳ".
4. Viết câu hỏi theo 4 mẫu (`variant`): full, natural, no_index, hard.
   `hard` bỏ kỳ + chỉ số nên mơ hồ — chỉ để xem, không tính vào điểm chính.
5. Kiểm chứng chéo bằng text layer PDF (PyMuPDF): mã và giá trị phải cùng
   xuất hiện trên một trang. Ô không qua kiểm chứng bị gắn `text_layer_ok=false`.

Ra: `data/gold/gold_qa_v2.jsonl` (+ `gold_qa_v2.stats.json`).
Cần pymupdf (`pip install -e ".[eval]"`):

    python scripts/eval/enrich_gold.py
    python scripts/eval/enrich_gold.py --per-table 80
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import metric_common as mc  # noqa: E402

OLD_GOLD = ROOT / "data" / "eval" / "gold_qa.jsonl"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "gold" / "gold_qa_v2.jsonl"
GEN = ROOT / "archive" / "geometric_parser" / "scripts" / "gen_gold_qa.py"

VARIANTS = ("full", "natural", "no_index", "hard")


def _load_generator():
    spec = importlib.util.spec_from_file_location("_gen_gold_qa", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def render_query(fact: dict, variant: str) -> str:
    e, col = fact["entity"], fact["col_label"]
    period, index = fact["period"], fact["index"]
    if variant == "natural":
        return f"{e} có {col.lower()} là bao nhiêu trong kỳ {period} của chỉ số {index}?"
    if variant == "no_index":
        return f"{col} của {e} {period}"
    if variant == "hard":
        return f"{col} của {e}"
    return f"{col} của {e} kỳ {period} chỉ số {index}"


def pick_variant(qid: str) -> str:
    """Chọn mẫu câu ổn định theo qid (tái lập, phân bố gần đều)."""
    h = int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16)
    return VARIANTS[h % len(VARIANTS)]


def _key(f: dict) -> tuple:
    return (f["entity"], f["col_label"].strip().lower(), f["index"])


def select(facts: list[dict], old_ids: set[str], per_table: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_table: dict[str, list[dict]] = collections.defaultdict(list)
    for f in facts:
        by_table[f["table_id"]].append(f)

    chosen: dict[str, dict] = {}
    for tid, rows in by_table.items():
        rows = sorted(rows, key=lambda f: f["qid"])
        rng.shuffle(rows)
        keep = [f for f in rows if f["qid"] in old_ids]
        rest = [f for f in rows if f["qid"] not in old_ids]
        for f in (keep + rest)[: max(per_table, len(keep))]:
            chosen[f["qid"]] = f

    # Cặp đối chứng: cùng (mã, cột, chỉ số), khác kỳ.
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for f in facts:
        groups[_key(f)].append(f)
    for f in list(chosen.values()):
        for other in groups[_key(f)]:
            if other["period"] != f["period"] and other["qid"] not in chosen:
                chosen[other["qid"]] = other
    return sorted(chosen.values(), key=lambda f: f["qid"])


def text_layer_check(rows: list[dict]) -> None:
    """Gắn `text_layer_ok` + `gold_pages` bằng text layer PyMuPDF."""
    import pymupdf

    pdfs = {p.name: p for p in RAW.rglob("*.pdf")}
    cache: dict[str, list[str]] = {}
    for r in rows:
        src = r["source_file"]
        if src not in cache:
            with pymupdf.open(pdfs[src]) as doc:
                cache[src] = [mc.strip_accents(p.get_text()).lower() for p in doc]
        entity = mc.strip_accents(r["entity"]).lower()
        target = round(float(r["value_norm"]), 6)
        pages = [
            n for n, text in enumerate(cache[src], 1)
            if entity in text and target in mc.numbers_in(text)
        ]
        r["gold_pages"] = pages
        r["text_layer_ok"] = bool(pages)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-table", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    gen = _load_generator()
    facts = gen.build_facts(None, args.seed)
    old_ids = {json.loads(l)["qid"] for l in OLD_GOLD.read_text(encoding="utf-8").splitlines()
               if l.strip()} if OLD_GOLD.exists() else set()

    rows = select(facts, old_ids, args.per_table, args.seed)
    periods_of = collections.defaultdict(set)
    for f in facts:
        periods_of[_key(f)].add(f["period"])

    for r in rows:
        r["source"] = "orig" if r["qid"] in old_ids else "new"
        r["variant"] = pick_variant(r["qid"])
        r["query"] = render_query(r, r["variant"])
        r["multi_period"] = len(periods_of[_key(r)]) > 1
        r["hard"] = r["variant"] == "hard"
    text_layer_check(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r["text_layer_ok"]]
    stats = {
        "total": len(rows),
        "from_old_gold": sum(r["source"] == "orig" for r in rows),
        "new": sum(r["source"] == "new" for r in rows),
        "text_layer_ok": len(ok),
        "text_layer_fail": len(rows) - len(ok),
        "by_variant": dict(collections.Counter(r["variant"] for r in rows)),
        "by_file": dict(collections.Counter(r["source_file"] for r in rows)),
        "by_period_index": dict(collections.Counter(f"{r['period']}|{r['index']}" for r in rows)),
        "multi_period": sum(r["multi_period"] for r in rows),
        "distinct_columns": len({r["col_label"] for r in rows}),
        "distinct_entities": len({r["entity"] for r in rows}),
    }
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[+] {len(rows)} câu -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
