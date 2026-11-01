#!/usr/bin/env python
"""Danh gia CHUNKING: 4 chien luoc, tren gold QA da sinh, do Recall@k + MRR.

Muc dich: tra loi "paper Chipper (element-based) co TOT HON fixed-size tren
du lieu minh khong?" — bang cach giu NGUYEN retrieval (BM25) va chi doi don vi
phân manh. Day la biet bien don (single-variable) ma paper van ton trong.

Bon chien luoc:
  * fixed_256 / fixed_512 — cat theo do dai co dinh (baseline yeu cua paper).
  * element  — dung 4 quy tac Chipper (Title mo chunk, Table giu nguyen, gop
               doan <2048) — NHUNG thay model Chipper bang element da parse
               cua etflab. Day la phien ban chay duoc thuat toan paper.
  * etflab   — table_row + table_card + prose (code hien tai).

Metric (chunk-agnostic, dua tren NOI DUNG):
  * Recall@k — trong top-k co chunk chua CA entity (VD "BSR") VA value
               (VD "0.87") khong.
  * MRR      — vi tri cua chunk dung dau tien.
Vi khong gan vao chunk_id, metric nay cong bang voi moi chien luoc.

Dung:
    python scripts/eval/eval_chunking.py
    python scripts/eval/eval_chunking.py --topk 1 3 5 10
    python scripts/eval/eval_chunking.py --strategies fixed_512 etflab
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]
ROOT = ARCHIVE.parents[1]
for _p in (ROOT, ARCHIVE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
PACKAGE = ARCHIVE / "etflab_geometric"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Import submodule etflab MA KHONG chay etflab/__init__.py (no keo theo pymupdf).
# ---------------------------------------------------------------------------
import types


def _load_pkg_submodules() -> tuple:
    pkg = types.ModuleType("etflab_geometric")
    pkg.__path__ = [str(PACKAGE)]
    sys.modules["etflab_geometric"] = pkg
    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"etflab_geometric.{name}", PACKAGE / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"etflab_geometric.{name}"] = mod
        setattr(pkg, name, mod)
        spec.loader.exec_module(mod)
        return mod
    vnnum = _load("vnnum")
    docmeta = _load("docmeta")
    tables = _load("tables")
    chunking = _load("chunking")
    return vnnum, docmeta, tables, chunking


_vnnum, _docmeta, _tables, _chunking = _load_pkg_submodules()
Chunk = _chunking.Chunk
build_chunks = _chunking.build_chunks
row_chunks = _chunking.row_chunks
Table = _tables.Table
is_number = _vnnum.is_number
parse_number = _vnnum.parse_number
strip_accents = _vnnum.strip_accents

PROCESSED = ROOT / "data" / "processed"
GOLD = ROOT / "data" / "eval" / "gold_qa.jsonl"
INCLUDE_HINTS = ("du_bao", "thay_doi_thanh_phan")


# ---------------------------------------------------------------------------
# Tai lai Table + DocumentResult tu JSON (KHONG can pymupdf).
# ---------------------------------------------------------------------------
def _table_from_dict(d: dict) -> Table:
    return Table(
        table_id=d["table_id"],
        source_file=d["source_file"],
        page=d["page"],
        page_span=d.get("page_span", []),
        caption=d.get("caption", ""),
        columns=d["columns"],
        rows=d["rows"],
        n_rows=d["n_rows"],
        n_cols=d["n_cols"],
        unit_notes=d.get("unit_notes", []),
        header_levels=d.get("header_levels", []),
        row_index_col=d.get("row_index_col"),
    )


def _load_docs() -> list[dict]:
    """Moi doc = {source_file, tables:[Table], pages_text:[{page,blocks}]}."""
    docs = []
    for path in sorted(PROCESSED.glob("*.json")):
        low = path.stem.lower()
        if not any(h in low for h in INCLUDE_HINTS):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        docs.append({
            "source_file": data["source_file"],
            "tables": [_table_from_dict(t) for t in data.get("tables", [])],
            "pages_text": data.get("pages_text", []),
        })
    return docs


class _Result:
    """Giong DocumentResult de ``build_chunks`` dung duoc."""
    def __init__(self, source_file, tables, pages_text):
        self.source_file = source_file
        self.tables = tables
        self.pages_text = [_Page(**p) for p in pages_text]


class _Page:
    def __init__(self, page, blocks):
        self.page = page
        self.blocks = blocks


# ---------------------------------------------------------------------------
# Bon chien luoc chunking.
# ---------------------------------------------------------------------------
def strat_etflab(doc) -> list[Chunk]:
    return build_chunks(_Result(doc["source_file"], doc["tables"], doc["pages_text"]))


def _serialize_tables(tables: list[Table]) -> list[str]:
    """Moi bang -> mot chuoi van ban giong nguoi doc (Title + header + tung dong)."""
    out = []
    for t in tables:
        lines = [t.caption or f"Bảng {t.n_rows}x{t.n_cols}", " | ".join(t.columns)]
        for row in t.rows:
            cells = [("" if row.get(str(c)) is None else str(row.get(str(c))))
                     for c in range(t.n_cols)]
            lines.append(" | ".join(cells))
        out.append("\n".join(lines))
    return out


def _element_sequence(doc) -> list[tuple[str, str]]:
    """Danh sach (loai, text) giong element cua Chipper: title/table/prose."""
    els: list[tuple[str, str]] = []
    for text in _serialize_tables(doc["tables"]):
        els.append(("table", text))
    for p in doc["pages_text"]:
        for block in p["blocks"]:
            txt = (block.get("text") or "").strip()
            if txt:
                els.append(("prose", txt))
    return els


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def strat_element(doc, target_chars: int = 2048) -> list[Chunk]:
    """4 quy tac Chipper tren element da parse:
      1/2. gop element cho du ``target_chars``, khong cat vto element;
      3.   Title -> mo chunk moi (o day: moi table co caption -> mo moi);
      4.   Table -> giu nguyen toan bo, mo chunk rieng.
    """
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    src = doc["source_file"]

    def flush():
        nonlocal buf, buf_len
        if buf:
            text = "\n".join(buf).strip()
            if text:
                chunks.append(Chunk(
                    chunk_id=f"{src}#el{len(chunks)}", kind="element",
                    text=text, sparse_text=text, source_file=src, page=1,
                ))
        buf, buf_len = [], 0

    for kind, text in _element_sequence(doc):
        if kind == "table":
            # Quy tac 4: bang giu nguyen, chunk rieng.
            flush()
            chunks.append(Chunk(
                chunk_id=f"{src}#elT{len(chunks)}", kind="element_table",
                text=text, sparse_text=text, source_file=src, page=1,
            ))
            continue
        # prose: gop ca doan (khong cat) neu van duoi nguong.
        for sentence in _split_sentences(text):
            s_len = len(sentence)
            if buf_len + s_len > target_chars and buf:
                flush()
            buf.append(sentence)
            buf_len += s_len
    flush()
    return chunks


def strat_fixed(doc, size: int) -> list[Chunk]:
    """Cat toan bo noi dung thanh khoi ``size`` ky tu — bo qua cau truc."""
    full = "\n".join(_serialize_tables(doc["tables"]))
    for p in doc["pages_text"]:
        for block in p["blocks"]:
            t = (block.get("text") or "").strip()
            if t:
                full += "\n" + t
    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    src = doc["source_file"]
    chunks = []
    step = size  # khong overlap: giu don gian nhu baseline cua paper
    for off in range(0, max(len(full), 1), step):
        piece = full[off:off + size]
        if piece.strip():
            chunks.append(Chunk(
                chunk_id=f"{src}#f{size}_{off}", kind="fixed",
                text=piece, sparse_text=piece, source_file=src, page=1,
            ))
    return chunks


# ---------------------------------------------------------------------------
# Metric.
# ---------------------------------------------------------------------------
def _norm_for_match(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


# So Viet Nam: co the la "21.094" (nghin), "0,87" (thap phan), "-41,6", "3.274.147".
# Chu so dang ke toan "(36,2)" cung la so AM. Neu khong bat dau bang dau
# ngoac, regex se khop "36,2" va tra ve +36.2 -> cau hoi ve o am bi tinh la
# "mat dap an" o moi phuong phap (loi metric, khong phai loi parser).
_NUM_SPAN_RE = re.compile(r"-?\(?\d+(?:[.,]\d+)*\)?%?")

# Don vi trong header, vd "('000 cổ phiếu)" hoac "(Triệu cổ phiếu)", khong phai
# gia tri. Neu khong xoa, moi chunk cua bang deu chua so 0 -> cau hoi gold co
# gia tri 0,0 se khop voi BAT KY dong nao co ma CK (9/130 cau trong bo gold).
_UNIT_ZERO_RE = re.compile(r"[\'’]\s*0{3}")


def _numbers_in(text: str) -> set[float]:
    """Trich moi so theo quy tac Viet Nam (dau cham=nghin, dau phay=thap phan)."""
    text = _UNIT_ZERO_RE.sub(" ", text)
    out: set[float] = set()
    for m in _NUM_SPAN_RE.finditer(text):
        token = m.group().rstrip("%")
        for dot_as_thousands in (True, False):
            try:
                num = parse_number(token, dot_as_thousands=dot_as_thousands)
            except (ValueError, OverflowError):
                continue
            if num is not None:
                out.add(round(num.value, 6))
    return out


def _matches(chunk: Chunk, entity: str, value_norm: float | None) -> bool:
    """Chunk 'dung' neu chua ca MA CO PHIEU va GIA TRI (da chuan hoa so Viet Nam)."""
    text = strip_accents(chunk.text).lower()
    if entity and strip_accents(entity).lower() not in text:
        return False
    if value_norm is None:
        return True
    # Khop so da parse bang chinh quy tac so Viet Nam — khong thay dau tu nhien.
    target = round(float(value_norm), 6)
    return target in _numbers_in(text)


@dataclass
class QResult:
    qid: str
    recall: dict[int, bool]
    rr: float  # reciprocal rank


def evaluate(chunks_by_file: dict, gold: list[dict], ks: list[int]) -> list[QResult]:
    maxk = max(ks)
    out = []
    for g in gold:
        chunks = chunks_by_file.get(g["source_file"], [])
        # Xep hang BM25 trong tung file (da theo nguon nhu doc_filter).
        ranked = _rank_by_score(chunks, g["query"])
        rr = 0.0
        recall = {k: False for k in ks}
        for pos, chunk in enumerate(ranked, 1):
            if pos > maxk:
                break
            if _matches(chunk, g["entity"], g["value_norm"]):
                if rr == 0.0:
                    rr = 1.0 / pos
                for k in ks:
                    if pos <= k:
                        recall[k] = True
        out.append(QResult(g["qid"], recall, rr))
    return out


def _rank_by_score(chunks: list[Chunk], query: str) -> list[Chunk]:
    """Xep hang bang diem BM25 don gian (dung tokenize cua etflab)."""
    import math
    from collections import Counter
    toks = [_chunk_tokens(c) for c in chunks]
    q = _q_tokens(query)
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
    scores = []
    for i in range(n):
        s = 0.0
        for w in q:
            if w not in idf or not tf[i].get(w):
                continue
            denom = tf[i][w] + 1.5 * (1 - 0.75 + 0.75 * length[i] / avgdl)
            s += idf[w] * tf[i][w] * 2.5 / denom
        scores.append(s)
    order = sorted(range(n), key=lambda i: -scores[i])
    return [chunks[i] for i in order]


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _chunk_tokens(chunk: Chunk) -> list[str]:
    return _tok(chunk.sparse_text or chunk.text)


def _q_tokens(query: str) -> list[str]:
    return _tok(query)


def _tok(text: str) -> list[str]:
    folded = strip_accents(text).lower().replace("đ", "d")
    return [w for w in _WORD_RE.findall(folded)]


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Danh gia 4 chien luoc chunking")
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--strategies", nargs="+",
                    default=["fixed_256", "fixed_512", "element", "etflab"])
    args = ap.parse_args()
    ks = sorted(set(args.topk))

    gold = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = _load_docs()
    print(f"[i] {len(gold)} cau hieu vang | {len(docs)} tai lieu", file=sys.stderr)

    builders = {
        "fixed_256": lambda d: strat_fixed(d, 256),
        "fixed_512": lambda d: strat_fixed(d, 512),
        "element":   strat_element,
        "etflab":    strat_etflab,
    }

    rows = []
    for name in args.strategies:
        by_file = {d["source_file"]: builders[name](d) for d in docs}
        total_chunks = sum(len(v) for v in by_file.values())
        results = evaluate(by_file, gold, ks)
        rec = {k: sum(1 for r in results if r.recall[k]) / len(results) for k in ks}
        mrr = sum(r.rr for r in results) / len(results)
        rows.append((name, total_chunks, rec, mrr))

    # In bang ket qua.
    print(f"\n{'Chien luoc':<12} {'#chunk':>8} " + " ".join(f"R@{k:<4}" for k in ks) + "  MRR")
    print("-" * (22 + 8 + len(ks) * 7 + 6))
    for name, nchunk, rec, mrr in rows:
        cells = " ".join(f"{rec[k]*100:6.1f}%" for k in ks)
        print(f"{name:<12} {nchunk:>8} {cells}  {mrr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
