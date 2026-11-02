"""Unit test hybrid retrieval — khong can model that, inject embed_fn gia."""

from __future__ import annotations

import numpy as np
import pytest

from src.chunk import Chunk
from src.retrieval import BM25, HybridIndex, RetrievalConfig, tokenize


def fake_embed(docs) -> np.ndarray:
    """Embed gia: khoa vector theo token 'bsr'/'tcx' de kiem chung RRF."""
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(len(docs), 16)).astype("float32")
    for i, text in enumerate(docs):
        low = text.lower()
        for word in ("bsr", "tcx", "dgc", "mch"):
            if word in low:
                mat[i, ord(word[0]) % 16] += 5.0
    return mat


# --------------------------------------------------------------- tokenize
def test_tokenize_bo_dau() -> None:
    assert tokenize("Tỷ trọng") == ["ty", "trong"]


def test_tokenize_chu_d_bien_thanh_d() -> None:
    assert tokenize("điều lệ") == ["dieu", "le"]


def test_tokenize_so_viet_nam_co_token_chuan_hoa() -> None:
    assert "2976.9" in tokenize("2.976,9")


def test_tokenize_cau_hoi_dot_va_tai_lieu_tu_khop() -> None:
    assert set(tokenize("2976,9")) & set(tokenize("2.976,9"))


def test_tokenize_cau_hoi_dau_cham_khop_tai_lieu_dau_phay() -> None:
    assert "2976.9" in tokenize("2976.9")


def test_tokenize_giu_ma_co_phieu_co_so() -> None:
    assert tokenize("VN30") == ["vn30"]
    assert "vn30" in tokenize("chi so VN30 va VNFIN LEAD")


def test_tokenize_phan_tram_co_token_rieng() -> None:
    assert "phan_tram" in tokenize("0,87%")


def test_tokenize_so_am() -> None:
    assert "69559" in tokenize("(69.559)")


def test_tokenize_bo_stopword() -> None:
    assert "cua" not in tokenize("tỷ trọng của BSR")


def test_tokenize_rong() -> None:
    assert tokenize("") == []


def test_tokenize_khong_lap_token() -> None:
    toks = tokenize("30/06/2026")
    assert len(toks) == len(set(toks))


# ------------------------------------------------------------------- BM25
def test_bm25_chuan_hoa_diem_khi_do_dai_bang() -> None:
    bm = BM25([tokenize("cổ phiếu tăng mạnh"), tokenize("cổ phiếu giảm mạnh")])
    scores = bm.score(tokenize("cổ phiếu"))
    assert scores[0] > 0
    assert scores[0] == pytest.approx(scores[1])


def test_bm25_do_dai_ngan_diem_cao_hon() -> None:
    bm = BM25([tokenize("cổ phiếu tăng mạnh"), tokenize("cổ phiếu giảm")])
    assert bm.score(tokenize("cổ phiếu"))[1] > bm.score(tokenize("cổ phiếu"))[0]


def test_bm25_cai_thien_theo_do_khop() -> None:
    bm = BM25([tokenize("cổ phiếu tăng mạnh"), tokenize("cổ phiếu giảm")])
    scores = bm.score(tokenize("tăng"))
    assert scores[0] > 0 > scores[1] or scores[0] > scores[1]


def test_bm25_rma_khong_co_token() -> None:
    bm = BM25([tokenize("abc")])
    assert bm.score(tokenize("xyz")) == [0.0]


def test_bm25_chi_muc_rong() -> None:
    bm = BM25([])
    assert bm.score(tokenize("abc")) == []


# ------------------------------------------------------------ HybridIndex
def make_chunks() -> list[Chunk]:
    """4 chunk gia lap: 1 the bang + 3 dong (khong phu thuoc parser)."""
    src = "doc__t01"
    card_text = (
        "Bảng: Tỷ trọng mới. Cột: Mã CP; Tỷ trọng mới. "
        "Số dòng: 3. Số cột: 2."
    )
    chunks = [Chunk(
        chunk_id=f"{src}#card", kind="table_card", label="Bảng: Tỷ trọng mới",
        text=card_text, sparse_text=card_text,
        source_file="a.pdf", page=1, table_id=src,
    )]
    for i, (ticker, value) in enumerate(
        [("BSR", "0,87%"), ("TCX", "1.423,1"), ("DGC", "5,00%")]
    ):
        text = (f"Bảng: Tỷ trọng mới\nMã CP = {ticker}\n"
                f"Tỷ trọng mới = {value}")
        chunks.append(Chunk(
            chunk_id=f"{src}#r{i + 1}", kind="table_row",
            label=f"Mã CP = {ticker}", text=text,
            sparse_text=f"Mã CP {ticker} Tỷ trọng mới {value}",
            source_file="a.pdf", page=1, parent_id=src, table_id=src,
            row_index=i,
        ))
    return chunks


@pytest.fixture()
def index() -> HybridIndex:
    return HybridIndex(make_chunks(), RetrievalConfig(use_dense=False))


def test_index_co_ca_the_bang_va_dong(index) -> None:
    assert len(index.chunks) == 4


def test_tim_dung_dong_theo_ma_co_phieu(index) -> None:
    hits = index.search("ty trong moi cua TCX")
    assert hits
    assert hits[0].chunk.label == "Mã CP = TCX"


def test_tim_dung_dong_theo_gia_tri_tai_lieu(index) -> None:
    hits = index.search("1423,1")
    assert hits[0].chunk.label == "Mã CP = TCX"


def test_hit_co_metadata_day_du(index) -> None:
    hit = index.search("BSR")[0]
    assert hit.chunk.table_id == "doc__t01"
    assert hit.chunk.row_index == 0
    assert hit.chunk.parent_id == "doc__t01"
    assert hit.bm25_rank is not None


def test_hit_mo_rong_nguon_cha(index) -> None:
    hit = index.search("BSR")[0]
    assert "Tỷ trọng mới" in hit.parent_text
    assert len(hit.parent_text) <= 420


def test_tat_mo_rong_cha() -> None:
    chunks = make_chunks()
    idx = HybridIndex(chunks, RetrievalConfig(use_dense=False, expand_parent=False))
    assert idx.search("BSR")[0].parent_text == ""


def test_khong_co_chunk_thi_tra_rong() -> None:
    assert HybridIndex([], RetrievalConfig()).search("bsr") == []


def test_top_k_dieu_chinh(index) -> None:
    assert len(index.search("tỷ trọng", top_k=2)) <= 2


def test_thu_tu_diem_giam_dan(index) -> None:
    scores = [h.score for h in index.search("tỷ trọng mới BSR TCX DGC", top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_index_rong_dense(index) -> None:
    """Khong co model thi van tra ket qua bang BM25."""
    assert index._ensure_dense() is False
    assert index.search("BSR")


def test_rrf_gop_hai_nguon() -> None:
    chunks = make_chunks()
    idx = HybridIndex(chunks, RetrievalConfig(use_dense=True), embed_fn=fake_embed)
    hits = idx.search("TCX")
    top = hits[0]
    assert top.chunk.label == "Mã CP = TCX"
    assert top.bm25_rank is not None and top.dense_rank is not None


def test_dense_bat_dung_khi_co_embed_fn() -> None:
    chunks = make_chunks()
    idx = HybridIndex(chunks, RetrievalConfig(use_dense=True), embed_fn=fake_embed)
    assert idx._ensure_dense() is True
    assert idx.search("BSR")


def test_cau_hinh_tu_dinh_khong_tu_tai_model() -> None:
    """Mac dinh phai la BM25: bat dense se tai model ngay o lan dung dau."""
    cfg = RetrievalConfig()
    assert cfg.use_dense is False
    assert cfg.rrf_k == 60.0


def test_hit_to_dict() -> None:
    chunks = make_chunks()
    idx = HybridIndex(chunks, RetrievalConfig(use_dense=False))
    data = idx.search("BSR")[0].to_dict()
    assert set(data) == {
        "chunk_id", "score", "bm25_rank", "dense_rank", "text", "metadata", "parent_text",
    }


def test_chunk_rong_khong_gay_loi() -> None:
    empty = Chunk(chunk_id="e", kind="prose", text="", sparse_text="",
                  source_file="a.pdf", page=1)
    idx = HybridIndex([empty], RetrievalConfig(use_dense=False))
    assert idx.search("bsr") is not None