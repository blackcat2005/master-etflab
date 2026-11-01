"""Kiem chung toan bo corpus va hoi qu giua cac PDF.

Tach hai loai kiem chung:
- ``test_*`` xac nhan hien trang DUNG (duoc khoa la chuan);
- ``xfail(strict=True)`` ghi nhan loi BIET, se bao "XPASS" ngay khi sua xong
  de nho go marker thay vi quen mang loi vao JSON/chi so cuoi cung.

Hien tai khong con loi biet nao: 3 loi cu da duoc sua va chuyen thanh test that
(``test_broker_q1_...``, ``test_ten_cot_khong_bat_dau_bang_ngay_thang``,
``test_vn30_q3_va_ban_tin_q1_tach_roi_stt_va_ma_cp``).
"""

from __future__ import annotations

import re

import pymupdf
import pytest
from conftest import (
    BROKER,
    BROKER_Q1_PDF,
    BROKER_VND_Q2_PDF,
    CHARTERS,
    NEWSLETTER_PDF,
    REVIEW_Q3_PDF,
    REVIEW_VND_Q1_PDF,
    REVIEW_VND_Q2_PDF,
    SSI_REVIEWS,
)
from etflab_geometric import IngestConfig, process_pdf

TEXT_PDFS = sorted({*SSI_REVIEWS.glob("*.pdf"), *BROKER.glob("*.pdf")})
SCAN_PDFS = sorted(CHARTERS.glob("*.pdf"))

# So lieu Viet Nam: dau cham ngan cach nhom, phay thap phan, ngoac la am,
# duoi % giu nguyen. "-" la o rong, khong phai so.
_NUMBER = re.compile(r"^\(?-?[\d.]+(?:,\d+)?\)?%?$")


def _starts_with_date(name: str) -> bool:
    """"dd/mm/yyyy" dan dau ten cot — dau chung cu chu thich bi lan vao header."""
    head = name.split()[0].rstrip(".,") if name.split() else ""
    bits = head.split("/")
    return len(bits) == 3 and all(b.isdigit() for b in bits) and len(bits[2]) == 4



def _is_numberish(value: str) -> bool:
    """"-" va chuoi rong la o rong hop le; con lai phai parse duoc thanh so."""
    text = value.strip()
    if text in {"", "-"}:
        return True
    return bool(_NUMBER.match(text))


def _ingest(path) -> object:
    return process_pdf(path, IngestConfig())


# ---------------------------------------------------------------- hien trang


def test_corpus_chia_hai_nhom_text_va_scan() -> None:
    assert len(TEXT_PDFS) == 8
    assert len(SCAN_PDFS) == 4
    for path in SCAN_PDFS:
        with pymupdf.open(path) as doc:
            # Deu lanh ban scan: >= 80% trang khong co text layer. Mot so trang
            # con con dau hanh/mo son con chu co the khong rong, nen dung nguong
            # ti le thay vi "tat ca deu rong".
            blank = sum(1 for page in doc if not page.get_text().strip())
            assert blank >= 0.8 * doc.page_count, f"{path.name}: {blank}/{doc.page_count}"


@pytest.mark.parametrize("path", TEXT_PDFS, ids=lambda p: p.stem[:34])
def test_moi_pdf_co_text_dau_ra_bang(path) -> None:
    result = _ingest(path)
    assert result.tables, f"{path.name}: co text layer nhung khong tach duoc bang"
    assert not any("OCR" in w for w in result.warnings)


@pytest.mark.parametrize("path", TEXT_PDFS, ids=lambda p: p.stem[:34])
def test_ten_cot_khong_phai_prose(path) -> None:
    """Hoi qu bug: o chu thich trai het bang bi gan cho moi cot.

    ``_header_levels`` nhan o theo do phu nen cau "Can luu y..." phu > 25% cot
    o, lam moi cot cung ten mot cau van. Ten cot phai la nhan, khong phai van.
    """
    for table in _ingest(path).tables:
        for name in table.columns:
            assert "Cần lưu ý" not in name, f"{path.name}: {name!r}"
            assert "Lưu ý:" not in name, f"{path.name}: {name!r}"
            assert not name.startswith("Bảng"), f"{path.name}: {name!r}"


@pytest.mark.parametrize("path", TEXT_PDFS, ids=lambda p: p.stem[:34])
def test_khoa_dong_la_chuoi_toan_corpus(path) -> None:
    """Hoi qu khoa str: bat buoc cho khoa chuoi de JSON/SQL/PoT doc lai duoc."""
    for table in _ingest(path).tables:
        assert all(isinstance(k, str) for row in table.rows for k in row)
        assert all(str(i) in row for row in table.rows for i in range(table.n_cols))


@pytest.mark.parametrize("path", TEXT_PDFS, ids=lambda p: p.stem[:34])
def test_ranh_dong_luon_co_gia_tri(path) -> None:
    """Dong duoc khoi duoc thi phai co it nhat mot o khong rong."""
    for table in _ingest(path).tables:
        for row in table.rows:
            assert any(v not in (None, "") for v in row.values())


@pytest.mark.parametrize("path", TEXT_PDFS, ids=lambda p: p.stem[:34])
def test_o_so_parse_duoc(path) -> None:
    """O cua cot so chi duoc chua so hoac dau giong '-' (dong khong doi)."""
    for table in _ingest(path).tables:
        for index in table.numeric_cols:
            for row in table.rows:
                value = row[str(index)]
                if value in (None, ""):
                    continue
                assert _is_numberish(value), (
                    f"{path.name} cot {index} ({table.columns[index]!r}): {value!r}"
                )


# --------------------------------------------------- SSI VN30 Q3/2026 (moi)


def test_vn30_q3_tach_hai_bang_dat_canh_nhau() -> None:
    result = _ingest(REVIEW_Q3_PDF)
    assert len(result.tables) == 2
    assert sorted(t.n_cols for t in result.tables) == [4, 7]
    assert all(t.n_rows == 32 for t in result.tables)
    assert all(t.page_span == [2] for t in result.tables)


def test_vn30_q3_ten_cot_khong_bi_chu_thich_rau() -> None:
    wide = max(_ingest(REVIEW_Q3_PDF).tables, key=lambda t: t.n_cols)
    assert wide.columns[2:] == [
        "Tỷ lệ Free- float",
        "Giới hạn tỷ trọng vốn hóa",
        "Vốn hóa tự do",
        "Tỷ trọng mới",
        "Tỷ trọng hiện tại",
    ]
    narrow = min(_ingest(REVIEW_Q3_PDF).tables, key=lambda t: t.n_cols)
    assert narrow.columns == [
        "% thay đổi",
        "Thay đổi số lượng CP (cổ phiếu)",
        "KLGD trung bình (cổ phiếu)",
        "Số ngày giao dịch (ngày)",
    ]


def test_vn30_q3_dong_dau_cuoi_khop_theo_ma_cp() -> None:
    tables = _ingest(REVIEW_Q3_PDF).tables
    wide = max(tables, key=lambda t: t.n_cols)
    narrow = min(tables, key=lambda t: t.n_cols)
    assert wide.rows[0]["1"] == "MCH" and narrow.rows[0]["0"] == "3,07%"
    assert wide.rows[-1]["1"] == "PLX" and narrow.rows[-1]["0"] == "-0,40%"


def test_vn30_q3_dong_khong_doi_dung_dau_giong() -> None:
    """Ma CK khong nam trong danh muc moi van duoc giu, so la '-'."""
    wide = max(_ingest(REVIEW_Q3_PDF).tables, key=lambda t: t.n_cols)
    row = wide.rows[-1]
    assert row["0"] == "-" and row["2"] == "-"
    assert row["6"] == "0,40%"


# --------------------------------------------------- broker VNDiamond Q2 (moi)


def test_broker_vndiamond_q2_mot_bang_doc_trang_2() -> None:
    result = _ingest(BROKER_VND_Q2_PDF)
    assert len(result.tables) == 1
    table = result.tables[0]
    assert (table.n_rows, table.n_cols) == (18, 8)
    assert table.page_span == [2]
    assert table.columns == [
        "CP",
        "Danh sách CP",
        "Tỷ trọng mới",
        "Tỷ trọng hiện tại",
        "% Thay đổi",
        "Ước tính thay đổi số cổ phiếu",
        "KLGD bình quân",
        "Số ngày giao dịch",
    ]
    assert table.unit_notes == ["percent"]


def test_broker_vndiamond_q2_va_tinh_thuoc_dong_dau_cuoi() -> None:
    rows = _ingest(BROKER_VND_Q2_PDF).tables[0].rows
    assert rows[0]["0"] == "FPT"
    assert rows[0]["1"] == "77.700 Duy trì"
    assert rows[0]["2"] == "15,0%"
    assert rows[-1]["0"] == "KDH"
    assert rows[-1]["1"].endswith("Chờ loại")


def test_broker_vndiamond_q2_co_caption() -> None:
    caption = _ingest(BROKER_VND_Q2_PDF).tables[0].caption
    assert caption.startswith("Bảng:")
    assert "DCVFMVNDiamond ETF" in caption


# --------------------------------------------------- broker VN30 Q1/2026 (moi)


def test_broker_q1_bang_tong_hop_co_so_lieu() -> None:
    """Bang tin Q1/2026 co HAI bang dat canh nhau, phu deu doc deu 40 ma.

    Ke giua hai bang chi 12pt, nho hon ke giua hai cot cua cung mot bang
    (16-70pt), nen ``vertical_bands`` khong tach ra. Chi dong tieu de nhom
    ("Kich ban 1" / "Kich ban 2") chung minh day la hai bang.
    """
    result = _ingest(BROKER_Q1_PDF)
    assert len(result.tables) == 2
    left, right = result.tables
    assert (left.n_rows, left.n_cols) == (40, 6)
    assert (right.n_rows, right.n_cols) == (40, 5)
    assert left.page_span == [3]
    assert right.page_span == [3]
    assert left.row_index_col == 0
    assert left.unit_notes == ["Triệu cổ phiếu"]
    assert right.unit_notes == ["Triệu cổ phiếu"]
    assert left.columns[0] == "STT"
    assert left.columns[1] == "Mã CP"
    assert right.columns[0] == "Mã CP"
    assert "Kịch bản" not in " ".join(str(c) for c in left.columns)
    assert "Kịch bản" not in " ".join(str(c) for c in right.columns)
    assert left.rows[0]["1"] == "VPB"
    assert right.rows[0]["0"] == "VPB"
    assert left.rows[-1]["0"] == "40"
    assert left.rows[-1]["1"] == "VIC"
    assert right.rows[-1]["0"] == "VIC"


# ----------------------------------------- SSI VNDiamond Q1/Q2/2026 (moi)


def test_vndiamond_q1_nam_bang_theo_bang() -> None:
    result = _ingest(REVIEW_VND_Q1_PDF)
    assert len(result.tables) == 5
    assert [t.n_rows for t in result.tables] == [31, 18, 18, 23, 46]
    assert [t.n_cols for t in result.tables] == [8, 7, 3, 5, 8]
    assert [t.page_span for t in result.tables] == [[3], [4], [4], [4], [5]]
    captions = [t.caption for t in result.tables]
    assert captions[0].startswith("Bảng 1:")
    assert captions[3].startswith("Bảng 3:")
    assert captions[4].startswith("Bảng 4:")


def test_vndiamond_q1_bang_1_co_du_lieu_viet() -> None:
    table = _ingest(REVIEW_VND_Q1_PDF).tables[0]
    assert table.row_index_col == 0
    assert table.rows[0]["1"] == "HDB"
    assert table.rows[-1]["1"] == "VIC"
    # Ngoac la so am.
    assert table.rows[-1]["3"] == "(1.635,1)"


def test_vndiamond_q2_tach_bang_theo_chieu_rong() -> None:
    result = _ingest(REVIEW_VND_Q2_PDF)
    assert len(result.tables) == 3
    assert [t.n_rows for t in result.tables] == [18, 18, 45]
    assert [t.n_cols for t in result.tables] == [5, 4, 7]
    assert [t.page_span for t in result.tables] == [[2], [2], [3]]


def test_vndiamond_q2_bang_1_va_bang_2_khop_theo_ma_cp() -> None:
    """Bang 5 cot va bang 4 cot la cung mot danh muc, phai ghep theo Ma CP."""
    tables = _ingest(REVIEW_VND_Q2_PDF).tables
    narrow = min(tables, key=lambda t: t.n_cols)
    assert narrow.columns == [
        "% thay đổi",
        "Thay đổi số lượng CP",
        "KLGD trung bình",
        "Số ngày giao dịch",
    ]
    assert narrow.rows[0]["1"] == "5.521.033"
    assert narrow.rows[-1]["1"] == "(4.252.080)"
    assert narrow.unit_notes == ["percent"]


def test_vndiamond_q2_bang_tong_hop_va_tinh_thuoc() -> None:
    table = max(_ingest(REVIEW_VND_Q2_PDF).tables, key=lambda t: t.n_cols)
    assert table.row_index_col == 0
    assert table.rows[0]["1"] == "FPT"
    assert table.rows[-1]["1"] == "KDH"
    # VNDiamond khong nam trong VN30 nen cot ETF mo phong VN30 rong.
    assert table.rows[0]["2"] == "1.638.065"


# --------------------------------------------- loi da biet, da ghi nhan bang xfail


def test_ten_cot_khong_bat_dau_bang_ngay_thang() -> None:
    """Duoi nhac caption con sot khong duoc lam ten cot.

    "...tính đến ngày 22/04/2026." bị xuống dòng, đứng giữa caption và header
    thật; đầu nó lệch trái ra ngoài bảng nên _header_block phải cắt.
    """
    for path in TEXT_PDFS:
        for table in _ingest(path).tables:
            for name in table.columns:
                assert not _starts_with_date(name), f"{path.name}: {name!r}"


def test_vn30_q3_va_ban_tin_q1_tach_roi_stt_va_ma_cp() -> None:
    """Hai cot STT / Ma CP ke nhau (khe ~6pt) phai tach, khong gop chung.

    ``group_cells`` dung khe ngang nen o rat hep bi gop, ca hai cot cung nhan
    "STT Ma CP". _text_in_column tach khi moi nhom tu deu roi vao dung cot.
    """
    wide = max(_ingest(REVIEW_Q3_PDF).tables, key=lambda t: t.n_cols)
    assert wide.columns[0] == "STT"
    assert wide.columns[1] == "Mã CP"
    newsletter = max(_ingest(NEWSLETTER_PDF).tables, key=lambda t: t.n_cols)
    assert newsletter.columns[0] == "STT"
    assert newsletter.columns[1] == "Mã CP"