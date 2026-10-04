"""Phase 1 — hybrid retrieval: BM25 + vector, ghep bang RRF.

Ly do dung BM25 cho bang tai chinh: nguoi dung hoi bang **ma co phieu** va **con
so** ("BSR ty trong moi la bao nhieu", "2.976,9"). BM25 khop chinh xac cac token
nay; vector search thi hay bo qua. Nguoc lai, vector search hieu "quy mo ta
gia tang manh" ma BM25 khong hieu. RRF ghep hai bang xep hang, khong can chuan
hoa thang do hai ben.

Token hoa rieng cho tieng Viet tai chinh:
  * bo dau de "ty trong" khop "Tỷ trọng";
  * ``2.976,9`` -> ``2976.9`` + ``2`` + ``976`` + ``9`` nen hoi "2976.9" hay
    "2.976,9" deu ra ket qua;
  * ``0,87%`` -> them token ``phan_tram``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chunking import Chunk
from .docmeta import DocMeta
from .vnnum import is_number, parse_number, strip_accents

__all__ = [
    "BM25",
    "DocFilter",
    "Hit",
    "HybridIndex",
    "RetrievalConfig",
    "infer_doc_filter",
    "tokenize",
]

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
# So Viet Nam phai biet TRUOC khi tach chu: "2.976,9" la mot so, con "VN30" thi
# khong phai (co chu dinh truoc) — lookbehind chan "VN30" bi cat thanh "VN"+"30".
_NUM_SPAN_RE = re.compile(r"(?<![^\W_])\d+(?:[.,]\d+)*%?")

# KHONG dua "trong" vao day: bo dau xong "Trọng" ra "trong" — bo tu nay se xoa
# sach chinh nhan cot quan trong nhat cua bao cao tai chinh Viet Nam.
STOPWORDS = frozenset({
    "va", "cua", "cho", "voi", "la", "co", "cac", "nhung", "ngoi", "duoc",
    "de", "tu", "den", "theo", "nhu", "hay", "vao", "ra", "mot", "nhay",
    "bang", "do", "nay", "khi", "da", "se", "ve", "tai", "o", "y",
})


def _fmt_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def _fold(text: str) -> str:
    """Bo dau; them 'đ' -> 'd' vi NFD khong tach duoc chu nay."""
    return strip_accents(text).lower().replace("đ", "d")


def _number_tokens(token: str) -> list[str] | None:
    """Sinh token cho mot chuoi so.

    "2.976,9" trong tieng Viet la 2976.9 (dau cham phan tach nghin, dau phay
    thap phan) — nhung nguoi dung lai go "2976.9" nghia la 2976.9. De khop duoc
    ca hai, index ca hai cach hieu thay vi phai chon.
    """
    out: list[str] = []
    for dots_as_thousands in (True, False):
        try:
            num = parse_number(token, dot_as_thousands=dots_as_thousands)
        except (ValueError, OverflowError):
            continue
        if num is None or not is_number(token):
            continue
        out.append(_fmt_number(num.value))
        if num.is_percent:
            out.append("phan_tram")
    if not out:
        return None
    out.extend(p for p in re.split(r"[.,]", token.rstrip("%")) if p)
    return _dedupe(out)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def tokenize(text: str) -> list[str]:
    """Rach thanh token chuan hoa cho BM25 (bo dau, tach so Viet Nam)."""
    if not text:
        return []
    folded = _fold(text)
    out: list[str] = []
    cursor = 0
    for match in _NUM_SPAN_RE.finditer(folded):
        out.extend(
            w
            for w in _WORD_RE.findall(folded[cursor : match.start()])
            if w not in STOPWORDS
        )
        cursor = match.end()
        numbers = _number_tokens(match.group())
        out.extend(numbers if numbers else [match.group()])
    out.extend(w for w in _WORD_RE.findall(folded[cursor:]) if w not in STOPWORDS)
    return out


@dataclass(frozen=True, slots=True)
class DocFilter:
    """Rang buoc cap TAI LIEU cho truy van.

    Vi sao can bo loc cung khi da dua ky vao ``sparse_text``: ``q2`` la
    *rang buoc* ("ban Q2"), khong phai *so thich*. BM25 cong cac diem nen
    ``bsr`` (idf 4.1, chi 12 chunk) lam cho ``q2`` (idf 1.2, 223 chunk) khong
    bao gio du chong lai — cau hoi "ty trong moi cua BSR Q2" van tra ve Q3.
    Loc cung loai bo cac chunk khop tu khoa nhung sai tai lieu.
    """

    doc_type: str | None = None
    index: str | None = None
    quarter: str | None = None
    year: str | None = None
    period: str | None = None
    source_file: str | None = None

    def is_empty(self) -> bool:
        return not any(
            (self.doc_type, self.index, self.quarter, self.year, self.period,
             self.source_file)
        )

    def describe(self) -> str:
        bits = []
        for name in ("doc_type", "index", "quarter", "period", "year", "source_file"):
            value = getattr(self, name)
            if value:
                bits.append(f"{name}={value}")
        return ", ".join(bits)

    def matches(self, doc: DocMeta) -> bool:
        if self.source_file and doc.source_file != self.source_file:
            return False
        if self.doc_type and doc.doc_type != self.doc_type:
            return False
        if self.index and _fold(self.index) != _fold(doc.index):
            return False
        if self.quarter and self.quarter.upper() != doc.quarter.upper():
            return False
        if self.year and self.year != doc.year:
            return False
        return not (self.period and self.period != doc.period)


# "Q2", "quý 2", "quý II" — chi nhan dang ky ro rang, khong doan tu so thuong.
_Q_PAT = re.compile(r"\bQ\s*([1-4])\b", re.IGNORECASE)
_Q_WORD = re.compile(
    r"\bqu[yỳ]\s*(?:([1-4])|(i{1,3}v?|iv)\b)", re.IGNORECASE
)
_YEAR_PAT = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_ROMAN_NUM = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 4}
_INDEX_PAT = re.compile(
    r"\b(VN\s*FIN\s*LEAD|VN\s*DIAMOND|VN\s*30)\b", re.IGNORECASE
)
_TYPE_PAT = (
    ("review", re.compile(r"thay\s*doi\s*thanh\s*phan", re.IGNORECASE)),
    ("forecast", re.compile(r"du\s*bao", re.IGNORECASE)),
    ("charter", re.compile(r"dieu\s*le|quy\s*dinh", re.IGNORECASE)),
)


def infer_doc_filter(query: str) -> DocFilter:
    """Tu nhan ra rang buoc trong cau hoi: ky, chi so, loai, nam.

    Chi nhan dang ky ton tai (ky so 1-4, nam 4 so, ten chi so) — khong suy
    dien. Cau hoi khong co rang buoc nao tra ve ``DocFilter()`` rong.
    """
    folded = _fold(query)
    quarter = ""
    m = _Q_PAT.search(folded)
    if m:
        quarter = f"Q{m.group(1)}"
    else:
        m = _Q_WORD.search(folded)
        if m:
            raw = m.group(1) or m.group(2)
            q = _ROMAN_NUM.get(raw.upper()) if raw else None
            if q is None and raw:
                q = int(raw)
            if q and 1 <= q <= 4:
                quarter = f"Q{q}"
    index = ""
    m = _INDEX_PAT.search(folded)
    if m:
        index = re.sub(r"\s+", "", m.group(1)).upper()
    doc_type = ""
    for name, rx in _TYPE_PAT:
        if rx.search(folded):
            doc_type = name
            break
    ym = _YEAR_PAT.search(folded)
    return DocFilter(
        doc_type=doc_type or None,
        index=index or None,
        quarter=quarter or None,
        year=ym.group(1) if ym else None,
    )


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int = 8
    rrf_k: int = 60.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # Mac dinh TAT: bat dense se tai model ~500MB ngay lan dung dau tien.
    # Hay de trong nhay, chi bat khi can vector search that su.
    use_dense: bool = False
    embed_model: str = "intfloat/multilingual-e5-small"
    expand_parent: bool = True
    parent_max_chars: int = 420
    dense_weight: float = 1.0
    bm25_weight: float = 1.0


class BM25:
    """Okapi BM25. ``docs`` la list token hoa san."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.tf: list[Counter[str]] = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = (sum(self.len) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: Sequence[str]) -> list[float]:
        scores = [0.0] * self.n
        if not self.n:
            return scores
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.len[i] / (self.avgdl or 1.0)
                )
                scores[i] += idf * freq * (self.k1 + 1) / denom
        return scores


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    parent_text: str = ""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
            "parent_text": self.parent_text,
        }


@dataclass(slots=True)
class HybridIndex:
    """Chi muc lai cho mot tap chunk. ``embed_fn`` co the inject de test."""

    chunks: list[Chunk]
    cfg: RetrievalConfig = field(default_factory=RetrievalConfig)
    embed_fn: Callable[[Sequence[str]], Any] | None = None
    dense_error: str | None = field(default=None, init=False, repr=False)
    _tokens: list[list[str]] = field(default_factory=list, init=False, repr=False)
    _bm25: Any = field(default=None, init=False, repr=False)
    _by_id: dict[str, Chunk] = field(default_factory=dict, init=False, repr=False)
    _parents: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _vectors: Any = field(default=None, init=False, repr=False)
    _dense_ready: bool = field(default=False, init=False, repr=False)
    _docs: list[DocMeta] | None = field(default=None, init=False, repr=False)
    _resolved_embed: Callable[[Sequence[str]], Any] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._tokens = [tokenize(c.sparse_text or c.text) for c in self.chunks]
        self._bm25 = BM25(self._tokens, k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)
        self._by_id = {c.chunk_id: c for c in self.chunks}
        # KHOA theo ``table_id`` chu khong phai ``chunk_id`` — chunk dong tro
        # ``parent_id`` = table_id, trong khi chunk cha co id "...#card".
        self._parents = {
            c.table_id: c.text
            for c in self.chunks
            if c.kind == "table_card" and c.table_id
        }

    # ---------------------------------------------------------------- dense
    def _default_embed_fn(self) -> Callable[[Sequence[str]], Any] | None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        model = SentenceTransformer(self.cfg.embed_model)

        def embed(texts: Sequence[str]):
            # e5 can tien to "query: "/"passage: " de input/output gap nhau.
            prefix = "query: " if len(texts) == 1 else "passage: "
            return model.encode(
                [prefix + t for t in texts],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        return embed

    def _ensure_dense(self) -> bool:
        if self._dense_ready:
            return self._vectors is not None
        self._dense_ready = True
        if not self.cfg.use_dense or not self.chunks:
            return False
        fn = self.embed_fn or self._default_embed_fn()
        if fn is None:
            return False
        self._resolved_embed = fn
        try:
            self._vectors = np.asarray(
                fn([c.text for c in self.chunks]), dtype="float32"
            )
        except (RuntimeError, ValueError, OSError) as exc:
            self._vectors = None
            self.dense_error = str(exc)
            return False
        return True

    def _query_vector(self, query: str):
        if self._resolved_embed is None:
            self._resolved_embed = self.embed_fn or self._default_embed_fn()
        if self._resolved_embed is None:
            raise RuntimeError("khong co ham embedding")
        return self._resolved_embed([query])

    @staticmethod
    def _cosine(vec: Any, matrix: Any):
        q = np.asarray(vec, dtype="float32").reshape(-1)
        m = np.asarray(matrix, dtype="float32")
        if m.ndim == 1:
            m = m.reshape(1, -1)
        denom = np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)
        return (m @ q) / np.where(denom == 0, 1.0, denom)

    # --------------------------------------------------------------- search
    def _allowed(self, doc_filter: DocFilter | None) -> set[int] | None:
        """Chi so khop ``doc_filter``, hoac ``None`` de khong loc gi."""
        if doc_filter is None or doc_filter.is_empty():
            return None
        if self._docs is None:
            self._docs = [c.doc for c in self.chunks]
        return {i for i, doc in enumerate(self._docs) if doc_filter.matches(doc)}

    def _ranks(
        self, scores: Sequence[float], allowed: set[int] | None = None
    ) -> dict[int, int]:
        if allowed is None:
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
        else:
            order = sorted(allowed, key=lambda i: -scores[i])
        return {idx: rank for rank, idx in enumerate(order)}

    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_filter: DocFilter | None = None,
    ) -> list[Hit]:
        if not self.chunks:
            return []
        cfg = self.cfg
        k = top_k or cfg.top_k
        pool = k * 4

        # Loc cung TRUOC khi tinh diem: chunk sai tai lieu khong nen ton
        # tai trong chi muc khi da lo.
        allowed = self._allowed(doc_filter)
        if allowed is not None and not allowed:
            return []

        bm25_scores = self._bm25.score(tokenize(query))
        bm25_ranks = self._ranks(bm25_scores, allowed)
        rrf: dict[int, float] = {}
        for idx in sorted(bm25_ranks, key=lambda i: bm25_ranks[i]):
            rank = bm25_ranks[idx]
            if rank >= pool:
                break
            rrf[idx] = rrf.get(idx, 0.0) + cfg.bm25_weight / (cfg.rrf_k + rank + 1)

        dense_ranks: dict[int, int] = {}
        if self._ensure_dense():
            sims = self._cosine(self._query_vector(query), self._vectors)
            dense_ranks = self._ranks(list(sims), allowed)
            for idx in sorted(dense_ranks, key=lambda i: dense_ranks[i]):
                rank = dense_ranks[idx]
                if rank >= pool:
                    break
                rrf[idx] = rrf.get(idx, 0.0) + cfg.dense_weight / (
                    cfg.rrf_k + rank + 1
                )

        hits: list[Hit] = []
        for idx in sorted(rrf, key=lambda i: -rrf[i])[:k]:
            chunk = self.chunks[idx]
            parent = ""
            if cfg.expand_parent and chunk.parent_id:
                parent = self._parents.get(chunk.parent_id, "")[: cfg.parent_max_chars]
            hits.append(
                Hit(
                    chunk=chunk,
                    score=rrf[idx],
                    bm25_rank=bm25_ranks.get(idx),
                    dense_rank=dense_ranks.get(idx),
                    parent_text=parent,
                )
            )
        return hits