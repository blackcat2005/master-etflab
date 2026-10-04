"""Khoi phuc bang khong vien tu toa do tu (khong dung find_tables)."""

from __future__ import annotations

from etflab.geometry import Word
from etflab.tables import TableConfig, TableExtractor, normalize_key

CHAR_W = 4.0
LINE_H = 9.0
ROW_H = 16.0


def build_words(
    rows: list[list[str | None]],
    col_x: list[float],
    *,
    y0: float = 120.0,
) -> list[Word]:
    """Dựng trang gia lap: moi o la mot chuoi chay chinh phai tai x cho truoc."""
    words: list[Word] = []
    for r, row in enumerate(rows):
        top = y0 + r * ROW_H
        for text, x in zip(row, col_x):
            if text is None:
                continue
            width = CHAR_W * len(text)
            words.append(
                Word(x0=x, y0=top, x1=x + width, y1=top + LINE_H, text=text)
            )
    return words


def extract(words: list[Word], **cfg_kwargs) -> list:
    return TableExtractor(TableConfig(**cfg_kwargs)).extract(words, page_no=1)


def test_bang_khong_vien_co_du_lieu() -> None:
    col_x = [50.0, 150.0, 260.0, 380.0]
    rows = [
        ["Mã CK", "Tỷ trọng", "% Thay đổi", "KLGD"],
        ["BSR", "0,87%", "0,87%", "12.930.183"],
        ["HPG", "9,43%", "0,35%", "35.428.055"],
        ["ACB", "3,55%", "0,14%", "11.652.776"],
        ["VPB", "4,31%", "0,17%", "14.770.810"],
        ["MBB", "3,70%", "0,15%", "14.868.669"],
    ]
    tables = extract(build_words(rows, col_x))
    assert len(tables) == 1
    t = tables[0]
    assert (t.n_rows, t.n_cols) == (5, 4)
    assert t.columns == ["Mã CK", "Tỷ trọng", "% Thay đổi", "KLGD"]
    assert t.rows[0] == {"0": "BSR", "1": "0,87%", "2": "0,87%", "3": "12.930.183"}
    assert t.rows[-1]["0"] == "MBB"


def test_o_rong_duoc_giu_la_null() -> None:
    col_x = [50.0, 150.0, 260.0, 380.0]
    rows = [
        ["Mã CK", "Tỷ trọng", "% Thay đổi", "KLGD"],
        ["BSR", "0,87%", None, "12.930.183"],
        ["HPG", "9,43%", "0,35%", "35.428.055"],
        ["ACB", "3,55%", "0,14%", "11.652.776"],
        ["VPB", "4,31%", "0,17%", "14.770.810"],
        ["MBB", "3,70%", "0,15%", "14.868.669"],
    ]
    tables = extract(build_words(rows, col_x))
    assert len(tables) == 1
    assert tables[0].rows[0]["2"] is None
    assert tables[0].rows[1]["2"] == "0,35%"


def test_duoi_nguong_so_dong_bi_lo() -> None:
    col_x = [50.0, 150.0, 260.0, 380.0]
    rows = [
        ["Mã CK", "Tỷ trọng", "% Thay đổi", "KLGD"],
        ["BSR", "0,87%", "0,87%", "12.930.183"],
        ["HPG", "9,43%", "0,35%", "35.428.055"],
    ]
    assert extract(build_words(rows, col_x)) == []


def test_van_ban_nhieu_so_khong_phai_bang() -> None:
    """Doan van ban justify chua nhieu con so deu khong duoc coi la bang."""
    lines: list[Word] = []
    widths = (9, 11, 10, 12, 9, 11, 10)
    for r, n in enumerate(widths):
        y = 120.0 + r * ROW_H
        cursor = 50.0
        for w, text in enumerate(
            ["Trong", "năm", "2026", "chi", "số", "12", "34", "56", "78", "9.012"][
                : n
            ]
        ):
            lines.append(
                Word(
                    x0=cursor,
                    y0=y,
                    x1=cursor + CHAR_W * len(text),
                    y1=y + LINE_H,
                    text=text,
                )
            )
            cursor += CHAR_W * len(text) + 6.0 + (3.0 if (r + w) % 3 == 0 else 0.0)
    assert extract(lines) == []


def test_tach_bang_dat_canh_nhau() -> None:
    """Hai bang doc lap cach nhau bang dai trong doc phai tach ra hai bang."""
    left_x = [30.0, 60.0, 130.0, 200.0, 260.0]
    right_x = [340.0, 420.0, 480.0, 540.0]
    rows = [
        ["STT", "Mã CP", "2.976,9", "-8.649,1", "11.316,4", "MCH", "3.001,7", "-9.000,0", "3.001,7"],
        ["2", "TCX", "1.000,0", "-1.000,0", "0,0", "TCX", "900,0", "-900,0", "0,0"],
        ["3", "VHM", "-500,0", "500,0", "0,0", "VHM", "400,0", "-400,0", "0,0"],
        ["4", "GVR", "250,0", "-250,0", "0,0", "GVR", "300,0", "-300,0", "0,0"],
        ["5", "BID", "120,0", "-120,0", "0,0", "BID", "110,0", "-110,0", "0,0"],
    ]
    header = [
        "STT", "Mã CP", "VN30", "SSIAM", "Tổng", "Mã CP", "VN30", "SSIAM", "Tổng",
    ]
    words = build_words([header] + rows, left_x + right_x)
    tables = extract(words)
    assert len(tables) == 2
    assert sorted(t.n_cols for t in tables) == [4, 5]
    assert all(t.n_rows == 5 for t in tables)
    assert all("col_" not in c for t in tables for c in t.columns)


def test_cot_so_thu_tu_stt_bi_nhan_dien() -> None:
    col_x = [40.0, 90.0, 190.0, 300.0]
    rows = [
        ["STT", "Mã CP", "VN30", "SSIAM"],
        ["1", "MCH", "2.976,9", "-8.649,1"],
        ["2", "TCX", "1.000,0", "-1.000,0"],
        ["3", "VHM", "-500,0", "500,0"],
        ["4", "GVR", "250,0", "-250,0"],
        ["5", "BID", "120,0", "-120,0"],
    ]
    t = extract(build_words(rows, col_x))[0]
    assert t.row_index_col == 0
    assert 0 not in t.numeric_cols
    # Cot 1 la ma chung khoan (MCH, TCX, ...) nen khong phai cot so do.
    assert t.numeric_cols == [2, 3]


def test_header_nhieu_tang_b_o_gop() -> None:
    """O header gop ('So luong CP mua ban') phai xuat hien o ca cot dau va cot sau."""
    col_x = [40.0, 90.0, 200.0, 290.0, 380.0]
    rows = [
        ["STT", "Mã CP", "2.976,9", "1.000,0", "11.316,4"],
        ["2", "TCX", "1.000,0", "-1.000,0", "0,0"],
        ["3", "VHM", "-500,0", "500,0", "0,0"],
        ["4", "GVR", "250,0", "-250,0", "0,0"],
        ["5", "BID", "120,0", "-120,0", "0,0"],
    ]
    # 6 dong header: 2 dong nhom, 3 dong nhan + 1 dong don vi
    header_lines = [
        ["", "", "Số lượng CP mua/bán", "Số lượng CP mua/bán", ""],
        ["", "", "Các quỹ ETF", "SSIAM", "Tổng"],
        ["STT", "Mã CP", "mô phỏng", "VNFIN LEAD", ""],
        ["", "", "VN30", "ETF", ""],
        ["", "", "(Nghìn cổ phiếu)", "(Nghìn cổ phiếu)", "(Nghìn cổ phiếu)"],
    ]
    words: list[Word] = []
    for r, line in enumerate(header_lines):
        top = 120.0 + r * 10.0
        for text, x in zip(line, col_x):
            if not text:
                continue
            words.append(
                Word(x0=x, y0=top, x1=x + CHAR_W * len(text), y1=top + 8.0, text=text)
            )
    data_y = 120.0 + len(header_lines) * 10.0
    for r, row in enumerate(rows):
        top = data_y + r * ROW_H
        for text, x in zip(row, col_x):
            words.append(
                Word(x0=x, y0=top, x1=x + CHAR_W * len(text), y1=top + LINE_H, text=text)
            )

    t = extract(words)[0]
    assert t.n_rows == 5
    assert "Nghìn cổ phiếu" in t.unit_notes
    assert t.columns[2].startswith("Số lượng CP mua/bán")
    assert "Các quỹ ETF" in t.columns[2] and "VN30" in t.columns[2]
    assert "SSIAM" in t.columns[3]


def test_chuan_hoa_ten_cot() -> None:
    assert normalize_key("  Tỷ  trọng   Mới ") == "ty trong moi"