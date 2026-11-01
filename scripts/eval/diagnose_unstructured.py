#!/usr/bin/env python
"""Do "khoang cach trich xuat" cua unstructured so voi gold QA.

Gold QA sinh tu o bang cua parser hinh hoc cu (nay da luu tru). Khi thay
parser, mot phan cau hoi co the KHONG CON dap an trong bat ky chunk nao (mat
dong, mat o, vo ky tu). Script nay do phan do *truoc* khi do chat luong ranking,
de khong quy loi cho chunking.

Phan loai moi cau hoi khong tim thay:
  * ``entity_missing``  — khong chunk nao chua ma CK  -> mat dong/ca bang
  * ``value_missing``   — co ma CK nhung khong co so  -> mat o / sai so
  * ``both_missing``    — khong ca hai                 -> mat ca dong

Dung:
    python scripts/eval/diagnose_unstructured.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import metric_common as mc

from etflab.uelements import load_unstructured_doc

UDATA = ROOT / "data" / "processed_unstructured"
GOLD = ROOT / "data" / "eval" / "gold_qa.jsonl"
OUT = ROOT / "data" / "eval" / "extraction_gap.json"


def entity_in(text: str, entity: str) -> bool:
    return bool(entity) and mc.strip_accents(entity).lower() in mc.strip_accents(text).lower()


def value_in(text: str, value_norm: float | None) -> bool:
    if value_norm is None:
        return True
    return round(float(value_norm), 6) in mc.numbers_in(mc.strip_accents(text).lower())


def main() -> int:
    gold = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    udocs = {d["source_file"]: d for d in
             (load_unstructured_doc(p) for p in sorted(UDATA.glob("*.json")))}

    per_file: dict[str, dict] = {}
    misses: list[dict] = []
    for g in gold:
        src = g["source_file"]
        doc = udocs.get(src)
        text = "\n".join(el.text for el in doc["elements"]) if doc else ""
        has_e = entity_in(text, g["entity"])
        has_v = value_in(text, g["value_norm"]) if has_e else False
        rec = per_file.setdefault(src, {"n": 0, "both": 0, "entity_only": 0,
                                        "value_only": 0, "neither": 0,
                                        "ut_tables": sum(1 for el in (doc or {}).get("elements", [])
                                                         if el.is_table)})
        rec["n"] += 1
        if has_e and has_v:
            rec["both"] += 1
        else:
            reason = ("value_missing" if has_e else
                      "entity_missing" if has_v else "both_missing")
            rec["entity_only" if has_e else "value_only" if has_v else "neither"] += 1
            misses.append({
                "qid": g["qid"], "source_file": src, "page": g["page"],
                "entity": g["entity"], "value_raw": g["value_raw"],
                "value_norm": g["value_norm"], "reason": reason,
            })

    print(f"{'file':<52} {'n':>3} {'both':>5} {'miss':>5}  ut tables")
    print("-" * 86)
    for src, r in per_file.items():
        print(f"{src[:52]:<52} {r['n']:>3} {r['both']:>5} {r['n']-r['both']:>5}  "
              f"{r['ut_tables']}")
    n = sum(r["n"] for r in per_file.values())
    both = sum(r["both"] for r in per_file.values())
    print("-" * 86)
    print(f"{'TONG':<52} {n:>3} {both:>5} {n-both:>5}  cover={both/n*100:.1f}%")
    from collections import Counter
    print("Ly do mat:", Counter(m["reason"] for m in misses))
    for m in misses[:8]:
        print(f"  - {m['reason']:<14} {m['entity']:<6} {m['value_raw']:<10} p{m['page']} {m['qid'][-14:]}")

    payload = {"per_file": per_file,
               "coverage": round(both / n, 4) if n else 0,
               "misses": misses}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[i] da ghi {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
