#!/usr/bin/env python
"""Sinh bo cau hieu vang (gold QA) tu bang da parse — KHONG can viet tay.

Moi "su kien" (fact) la mot o trong bang: (row_label, col_label) -> value.
Tu do sinh mot cau hoi tu nhien + ghi nhan gold de danh gia retrieval.

Thiet ke de danh gia CHUNKING, khong danh gia generation:
  * Gold duoc dinh nghia o muc SU KIEN (source_file, table_id, row, col),
    khong gan buoc vao chunk_id cua bat ky chien luoc nao.
  * Khi danh gia, mot chunk duoc coi la RELEVANT neu no chua ca row_label
    VA value (da chuan hoa) — noi dung tuyet doi, hoat dong voi moi chien luoc
    (fixed-size, element, etflab) deu duoc.

Dung:
    python scripts/eval/gen_gold_qa.py                       # toan bo -> data/eval/gold_qa.jsonl
    python scripts/eval/gen_gold_qa.py --per-table 8         # it hon
    python scripts/eval/gen_gold_qa.py --seed 7              # mau khac nhau
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]
ROOT = ARCHIVE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import ``vnnum`` TRUC TIEP tu file, minh tranh ``etflab/__init__.py`` keo
# theo pymupdf (van chua can thiet khi chi sinh gold tu JSON da parse).
import importlib.util

_vn = importlib.util.spec_from_file_location(
    "_vnnum", ARCHIVE / "etflab_geometric" / "vnnum.py"
)
_vnmod = importlib.util.module_from_spec(_vn)
sys.modules["_vnnum"] = _vnmod  # dataclass(slots=True) can module trong sys.modules
_vn.loader.exec_module(_vnmod)
is_number = _vnmod.is_number
parse_number = _vnmod.parse_number

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "eval"

# Chi sinh QA cho 2 nhom co bang (forecast + review). Dieu le la scan -> 0 bang.
INCLUDE_HINTS = ("du_bao", "thay_doi_thanh_phan")

_GENERIC_COL = re.compile(r"^cot\s*\d+$", re.IGNORECASE)
# Ma co phieu VN: 2-5 chu cai in hoa, co the keo theo so (VD "VPB", "MCH", "GMD").
_TICKER_RE = re.compile(r"^[A-Z]{2,5}\d{0,3}$")
# Cot chua ma/ten co phieu.
_ENTITY_NAME_RE = re.compile(r"\bmã\b|\bma\b|\bcode\b|ticker|\bcp\b|\bck\b", re.IGNORECASE)
# Gia tri thay the trong bang rong — khong phai entity that.
_PLACEHOLDER = {"-", "—", "–", "n/a", "na", "none", "null", ""}


def _find_entity_col(table: dict) -> int:
    """Tim cot chua ma co phieu (entity) — khong phai luon la cot 0.

    Uu tien ten cot co 'ma'/'cp'/'ck'; neu khong co, tim cot co >60% gia tri
    dang ma co phieu (VPB, MCH, HDB...). Dung khi bang co STT o cot 0.
    """
    cols = table["columns"]
    for ci, name in enumerate(cols):
        if _ENTITY_NAME_RE.search(name or ""):
            return ci
    for ci in range(table["n_cols"]):
        vals = _col_values(table, ci)
        if not vals:
            continue
        hits = sum(1 for v in vals if _TICKER_RE.match(v))
        if hits / len(vals) >= 0.6:
            return ci
    return 0


def _find_row_index_col(table: dict) -> int | None:
    """Cot STT day so tu 1..N — loai khoi cau hoi vi no khong phai thong tin."""
    for ci in range(table["n_cols"]):
        vals = _col_values(table, ci)
        if len(vals) < 3:
            continue
        try:
            nums = [int(float(v.replace(",", "."))) for v in vals]
        except ValueError:
            continue
        if nums and nums[0] in (0, 1) and nums == list(range(nums[0], nums[0] + len(nums))):
            return ci
    return None


def _col_values(table: dict, ci: int) -> list[str]:
    out = []
    for row in table.get("rows", []):
        v = row.get(str(ci))
        if v is not None and str(v).strip():
            out.append(str(v).strip())
    return out


def _is_numeric_col(table: dict, ci: int) -> bool:
    """Cot tra loi la cot >40% gia tri dang so/% — loai 'Danh sach CP', 'Ma CP'."""
    vals = _col_values(table, ci)
    if not vals:
        return False
    numeric = sum(1 for v in vals if _norm_value(v) is not None)
    return numeric / len(vals) >= 0.4


def _norm_value(raw: str) -> float | None:
    """Chuan hoa so Viet Nam de khop noi dung. Tra ve None neu khong phai so."""
    token = raw.strip()
    if not token or not is_number(token):
        return None
    try:
        num = parse_number(token)
    except (ValueError, OverflowError):
        return None
    return None if num is None else num.value


def _fmt_num(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def _is_answer_col(label: str, col_idx: int, table: dict) -> bool:
    """Loai cac cot khong phai 'dap an': STT, cot nhan (ma CK), cot text dai."""
    if not label or _GENERIC_COL.match(label):
        return False
    if col_idx == table.get("_row_index_col"):
        return False
    if col_idx == table.get("_entity_col"):
        return False
    # Dap an phai la cot co gia tri so/% dang nghia.
    return _is_numeric_col(table, col_idx)


def _row_label_of(table: dict, row: dict) -> str | None:
    """Gia tri cot dinh danh (VD 'BSR') = 'entity' trong cau hoi."""
    idx = table.get("_entity_col", 0)
    val = row.get(str(idx))
    if val is None:
        return None
    text = str(val).strip()
    if text.lower() in _PLACEHOLDER or not text:
        return None
    # Entity la ma co phieu (2-5 chu cai), khong phai so hay cau van.
    if not _TICKER_RE.match(text):
        return None
    return text


def _load_tables() -> list[tuple[str, dict]]:
    """Tra ve (source_file_stem, table_dict) cho cac file co bang."""
    out: list[tuple[str, dict]] = []
    for path in sorted(PROCESSED.glob("*.json")):
        stem = path.stem
        low = stem.lower()
        if not any(h in low for h in INCLUDE_HINTS):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for table in data.get("tables", []):
            if table.get("n_rows", 0) > 0:
                out.append((stem, table))
    return out


def build_facts(per_table: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    facts: list[dict] = []
    for stem, table in _load_tables():
        cols = table["columns"]
        # Danh gia cot truoc: entity (ma CK) + STT de loai, roi chon cot dap an.
        table["_entity_col"] = _find_entity_col(table)
        table["_row_index_col"] = _find_row_index_col(table)
        answer_cols = [
            (ci, label)
            for ci, label in enumerate(cols)
            if _is_answer_col(label, ci, table)
        ]
        table_facts: list[dict] = []
        for r_idx, row in enumerate(table.get("rows", [])):
            entity = _row_label_of(table, row)
            if not entity:
                continue
            for ci, col_label in answer_cols:
                raw = row.get(str(ci))
                if raw is None or not str(raw).strip():
                    continue
                raw = str(raw).strip()
                norm = _norm_value(raw)
                # Chi giu cac o co GIA TRI ro rang (so hoac %) — tranh text la.
                if norm is None:
                    continue
                table_facts.append(_make_fact(stem, table, r_idx, ci,
                                              entity, col_label, raw, norm))
        rng.shuffle(table_facts)
        if per_table is not None:
            table_facts = table_facts[:per_table]
        facts.extend(table_facts)
    rng.shuffle(facts)
    return facts


def _make_fact(stem, table, r_idx, ci, entity, col_label, raw, norm) -> dict:
    period = _period_from_stem(stem)
    index = _index_from_stem(stem)
    query = _render_query(entity, col_label, period, index)
    return {
        "qid": f"{table['table_id']}#r{r_idx}c{ci}",
        "query": query,
        "source_file": table["source_file"],
        "table_id": table["table_id"],
        "page": table["page"],
        "row_index": r_idx,
        "col_index": ci,
        "entity": entity,          # "BSR"
        "col_label": col_label,    # "Tỷ trọng mới"
        "value_raw": raw,          # "0,87%"
        "value_norm": norm,        # 0.87
        "period": period,          # "Q2 2026"
        "index": index,            # "VN30"
    }


def _render_query(entity: str, col_label: str, period: str, index: str) -> str:
    """Cau hoi tu nhien, doi template de khong lac quan. Dinh ke hop ly de BM25 co the kho."""
    core = f"{col_label} của {entity}"
    if period and index:
        variants = [
            f"{core} kỳ {period} chỉ số {index}",
            f"{core} {period}",
            f"{core}",
        ]
    elif period:
        variants = [f"{core} {period}", f"{core}"]
    else:
        variants = [f"{core}"]
    # Chon dau tien co ky + index — day la cau "thuc" nhat va kiem tra duoc doc filter.
    return variants[0]


def _period_from_stem(stem: str) -> str:
    flat = stem.replace("_", " ").replace("-", " ")
    m = re.search(r"\bq\s*([1-4])\s*(\d{4})", flat, re.IGNORECASE)
    return f"Q{m.group(1)} {m.group(2)}" if m else ""


def _index_from_stem(stem: str) -> str:
    flat = re.sub(r"[\s_\-]+", "", stem).upper()
    for name in ("VNFINLEAD", "VNDIAMOND", "VN30"):
        if name in flat:
            return name
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Sinh gold QA tu bang da parse")
    ap.add_argument("--per-table", type=int, default=10,
                    help="so cau toi da moi bang (mac dinh 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "gold_qa.jsonl")
    args = ap.parse_args()

    facts = build_facts(args.per_table, args.seed)
    if not facts:
        sys.exit("[!] khong sinh duoc fact nao — kiem tra data/processed/")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for fact in facts:
            fh.write(json.dumps(fact, ensure_ascii=False) + "\n")

    # Tom tat ngan.
    files = sorted({f["source_file"] for f in facts})
    print(f"[+] Da sinh {len(facts)} cau hieu vang -> {args.out.relative_to(ROOT)}")
    print(f"    {len(files)} file, {len({f['table_id'] for f in facts})} bang")
    print("\n    Vi du:")
    for fact in facts[:5]:
        print(f"      [{fact['period'] or '?'}] {fact['query']!r} -> {fact['value_raw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
