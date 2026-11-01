"""Kiem chung tren PDF that cua ETF Viet Nam (gia tri mau kiem chung)."""

from __future__ import annotations

import json

import pytest
from conftest import CHARTERS, NEWSLETTER_PDF, REVIEW_PDF
from etflab_geometric import IngestConfig, process_pdf

pytestmark = pytest.mark.skipif(
    not (REVIEW_PDF.exists() and NEWSLETTER_PDF.exists()),
    reason="khong tim thay PDF mau trong repo",
)


@pytest.fixture(scope="module")
def review():
    return process_pdf(REVIEW_PDF, IngestConfig())


@pytest.fixture(scope="module")
def newsletter():
    return process_pdf(NEWSLETTER_PDF, IngestConfig())


def test_review_mot_bang_doc_trang_1_2(review) -> None:
    assert len(review.tables) == 1
    t = review.tables[0]
    assert t.page_span == [1, 2]
    assert (t.n_rows, t.n_cols) == (31, 7)
    assert t.columns == [
        "Mã CK",
        "Tỷ trọng mới",
        "Tỷ trọng hiện tại",
        "% Thay đổi",
        "Thay đổi số lượng CP",
        "KLGD trung bình 20 phiên",
        "Số ngày giao dịch",
    ]
    assert t.unit_notes == ["percent"]


def test_review_dong_dau_va_dong_cuoi(review) -> None:
    rows = review.tables[0].rows
    assert rows[0] == {
        "0": "BSR",
        "1": "0,87%",
        "2": "0,00%",
        "3": "0,87%",
        "4": "3.274.147",
        "5": "12.930.183",
        "6": "0,25",
    }
    assert rows[-1]["0"] == "DGC"
    assert rows[-1]["4"] == "(1.792.841)"


def test_review_co_caption(review) -> None:
    caption = review.tables[0].caption
    assert caption.startswith("Bảng:")
    assert "Ước tính thay đổi" in caption


def test_ban_tin_hai_bang_dat_canh_nhau(newsletter) -> None:
    assert len(newsletter.tables) == 2
    shapes = sorted((t.n_rows, t.n_cols) for t in newsletter.tables)
    assert shapes == [(41, 5), (41, 6)]
    assert all(t.page_span == [2, 3] for t in newsletter.tables)


def test_ban_tin_ten_cot_va_don_vi(newsletter) -> None:
    wide = max(newsletter.tables, key=lambda t: t.n_cols)
    assert wide.columns[0] == "STT"
    assert wide.columns[1] == "Mã CP"
    assert wide.columns[2].startswith("Số lượng CP mua/bán")
    assert "Các quỹ ETF mô phỏng VN30" in wide.columns[2]
    assert "SSIAM VNFIN LEAD ETF" in wide.columns[3]
    assert wide.columns[4] == "Tổng (Nghìn cổ phiếu)"
    assert wide.columns[5] == "Số phiên GD tương đương"
    assert wide.unit_notes == ["Nghìn cổ phiếu"]


def test_ban_tin_gop_dung_theo_ma_cp_khong_lo(newsletter) -> None:
    """Hai bang phai ghep theo cot STT/Ma CP, khong ghep cheo."""
    tables = sorted(newsletter.tables, key=lambda t: t.n_cols)
    assert tables[0].rows[0]['0'] == "MCH"
    assert tables[0].rows[-1]['0'] == "SSB"
    assert tables[1].rows[0]['1'] == "MCH"
    assert tables[1].rows[-1]['1'] == "SSB"


def test_ban_tin_o_rong_giu_null(newsletter) -> None:
    """MCH khong nam trong VNFIN LEAD nen cot SSIAM rong, phai la null."""
    wide = max(newsletter.tables, key=lambda t: t.n_cols)
    assert wide.rows[0]['3'] is None
    assert wide.rows[0]['2'] is not None


def test_van_ban_duoc_tach_rieng_khoi_bang(review) -> None:
    page1 = review.pages_text[0]
    assert page1.n_blocks > 0
    joined = " ".join(b["text"] for b in page1.blocks)
    assert "Bản tin ETF" in joined
    for block in page1.blocks:
        assert "3.274.147" not in block["text"]


def test_khoa_dong_la_chuoi_roi_json_doc_lai_duoc(review) -> None:
    """Hoi quy: rows phai dung khoa str o ca RAM va JSON.

    Neu khoa la int trong RAM, json.dumps bien thanh str -> doc lai tu file se
    KeyError, va ma sinh SQL/PoT sau nay se hong.
    """
    table = review.tables[0]
    assert all(isinstance(k, str) for row in table.rows for k in row)
    roundtrip = json.loads(json.dumps(table.to_dict(), ensure_ascii=False))
    assert roundtrip["rows"][0] == table.rows[0]


@pytest.mark.parametrize("name", sorted(p.name for p in CHARTERS.glob("*.pdf")))
def test_dieu_le_bang_scan_bao_lo(name: str) -> None:
    result = process_pdf(CHARTERS / name, IngestConfig())
    assert result.tables == []
    assert result.warnings
    assert any("OCR" in w for w in result.warnings)