"""Test metadata cap tai lieu + loc cung theo ky.

Tach rieng vi day la lop sua loi "hoi ban Q2 lai tra ve Q3": cau hoi co
chua rang buoc tai lieu (ky, chi so) nhung chunk khong mang thong tin do.
"""

from __future__ import annotations

import pytest
from test_chunking import make_table

from etflab.chunking import ChunkConfig, build_chunks
from etflab.docmeta import parse_doc_meta
from etflab.pipeline import DocumentResult
from etflab.retrieval import (
    DocFilter,
    HybridIndex,
    RetrievalConfig,
    infer_doc_filter,
)

# ------------------------------------------------- parse metadata tu ten file


@pytest.mark.parametrize(
    ("filename", "doc_type", "index", "period"),
    [
        ("Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf", "review", "VN30", "Q2/2026"),
        ("Thay_doi_thanh_phan_chi_so_VN30_Q3_2026.pdf", "review", "VN30", "Q3/2026"),
        (
            "Thay_doi_thanh_phan_chi_so_VNDiamond_VN30_VNFIN_Q1_2026.pdf",
            "review",
            "VNDIAMOND",
            "Q1/2026",
        ),
        (
            "Du_bao_thay_doi_danh_muc_chi_so_VN30_VNFINLEAD_Q1_2026.pdf",
            "forecast",
            "VNFINLEAD",
            "Q1/2026",
        ),
        (
            "Du_bao_thay_doi_danh_muc_chi_so_VNDiamond_Q2_2026.pdf",
            "forecast",
            "VNDIAMOND",
            "Q2/2026",
        ),
        ("20260519 - FUESSVFL - Dieu Le 2026.pdf", "charter", "", ""),
    ],
)
def test_parse_doc_meta_tu_ten_file(filename, doc_type, index, period) -> None:
    meta = parse_doc_meta(filename)
    assert meta.doc_type == doc_type
    assert meta.index == index
    assert meta.period == period


def test_parse_doc_meta_ten_file_co_ngay_dau() -> None:
    meta = parse_doc_meta("20260508 - FUEVFVND - Dieu le quy ngay 08.05.2026.pdf")
    assert meta.date == "2026-05-08"


def test_parse_doc_meta_fallback_sang_trang_dau() -> None:
    """Ten file khong co ky thi lay tu trang dau ("ky Q2/2026")."""
    meta = parse_doc_meta("BCB VNFIN LEAD TIENG VIET.pdf", "Bản tin kỳ Q2/2026")
    assert meta.quarter == "Q2"
    assert meta.year == "2026"
    assert meta.index == "VNFINLEAD"


@pytest.mark.parametrize(
    "hint", ["Bản tin kỳ Q3/2026", "bản tin quý II/2026", "Kỳ Q1 năm 2026"]
)
def test_parse_doc_meta_doc_bien_dang_ky_quy(hint) -> None:
    meta = parse_doc_meta("x.pdf", hint)
    assert meta.quarter in {"Q1", "Q2", "Q3"}


def test_label_khong_lap_chu_chi_so() -> None:
    label = parse_doc_meta("Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf").label
    assert label.count("chỉ số") == 1
    assert "kỳ Q2/2026" in label


# --------------------------------------------- suy ra rang buoc tu cau hoi


@pytest.mark.parametrize(
    ("query", "quarter", "index"),
    [
        ("Tỷ trọng mới của BSR Q2", "Q2", None),
        ("tỷ trọng mới của BSR Q3", "Q3", None),
        ("quý II/2026 có thay đổi gì", "Q2", None),
        ("quý 4 năm 2026", "Q4", None),
        ("tỷ trọng VN30 Q1", "Q1", "VN30"),
        ("dự báo VNDiamond Q2", "Q2", "VNDIAMOND"),
        ("điều lệ quỹ", None, None),
        ("tỷ trọng mới của BSR", None, None),
        ("BSR năm 2026", None, None),
    ],
)
def test_infer_doc_filter(query, quarter, index) -> None:
    got = infer_doc_filter(query)
    assert got.quarter == quarter
    assert got.index == index


def test_infer_doc_filter_khong_vao_bay_so() -> None:
    """'2.976,9' la gia tri, khong phai ky — khong duoc suy ra Q9."""
    assert infer_doc_filter("2.976,9 la so luong cua co phieu nao").quarter is None


def test_infer_doc_filter_biet_loai_tai_lieu() -> None:
    assert infer_doc_filter("bản tin dự báo VN30").doc_type == "forecast"
    assert infer_doc_filter("điều lệ quỹ E1VFVN30").doc_type == "charter"


# ------------------------------------------------------- loc chunk theo ky


def _corpus() -> list:
    chunks = []
    for quarter in ("Q2", "Q3"):
        name = f"Thay_doi_thanh_phan_chi_so_VN30_{quarter}_2026.pdf"
        result = DocumentResult(
            source_file=name,
            n_pages=1,
            tables=[make_table(source_file=name)],
            pages_text=[],
            warnings=[],
        )
        chunks.extend(build_chunks(result, ChunkConfig()))
    return chunks


def _index() -> HybridIndex:
    return HybridIndex(_corpus(), RetrievalConfig(use_dense=False))


def test_chunk_mang_metadata_tai_lieu() -> None:
    chunk = _corpus()[0]
    assert chunk.doc.quarter == "Q2"
    assert "Q2/2026" in chunk.doc.label
    assert chunk.sparse_text.startswith(chunk.doc.label)
    assert chunk.metadata["doc"]["quarter"] == "Q2"


def test_loc_ky_tra_dung_tai_lieu() -> None:
    """Loi goc: hoi ban Q2 phai ra chunk cua ban Q2, khong phai Q3."""
    hits = _index().search("Tỷ trọng mới của BSR", doc_filter=DocFilter(quarter="Q2"))
    assert hits
    assert {h.chunk.doc.quarter for h in hits} == {"Q2"}


def test_loc_ky_q3_va_q2_tach_biet() -> None:
    idx = _index()
    for quarter in ("Q2", "Q3"):
        hits = idx.search("Tỷ trọng mới của BSR", doc_filter=DocFilter(quarter=quarter))
        assert {h.chunk.source_file for h in hits} == {
            f"Thay_doi_thanh_phan_chi_so_VN30_{quarter}_2026.pdf"
        }


def test_loc_ky_khong_co_ket_qua() -> None:
    assert _index().search("BSR", doc_filter=DocFilter(quarter="Q1")) == []


def test_loc_rong_thi_khong_loc() -> None:
    idx = _index()
    unfiltered = idx.search("Tỷ trọng mới của BSR")
    assert {h.chunk.doc.quarter for h in unfiltered} == {"Q2", "Q3"}
    assert idx.search("Tỷ trọng mới của BSR", doc_filter=DocFilter()) == unfiltered


def test_loc_theo_chi_so() -> None:
    hits = _index().search("Tỷ trọng mới", doc_filter=DocFilter(index="VNDiamond"))
    assert hits == []


def test_doc_filter_rong_khong_loc() -> None:
    assert DocFilter().is_empty()
    assert not DocFilter(quarter="Q2").is_empty()
    assert DocFilter(quarter="Q2").describe() == "quarter=Q2"


def test_matches_so_sanh_khong_phan_biet_hoa_thuong() -> None:
    doc = parse_doc_meta("Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf")
    assert DocFilter(index="vn30").matches(doc)
    assert DocFilter(quarter="q2").matches(doc)
    assert not DocFilter(quarter="Q3").matches(doc)