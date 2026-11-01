"""Unit test cho lop chunking (Phase 1) — dung bang gia lap de chay nhanh."""

from __future__ import annotations

from etflab_geometric.chunking import (
    ChunkConfig,
    build_chunks,
    header_paths,
    row_chunks,
    short_caption,
    table_card,
)
from etflab_geometric.geometry import Line, Word
from etflab_geometric.pipeline import DocumentResult, PageText
from etflab_geometric.tables import Table

SRC = "test.pdf"


def make_table(**kw) -> Table:
    base: dict = {
        "table_id": "doc__t01",
        "source_file": SRC,
        "page": 2,
        "page_span": [2, 3],
        "caption": "Bảng: Tổng hợp thay đổi về số lượng cổ phiếu của các quỹ liên quan đến VN30, dữ liệu đến ngày 30/06/2026. Cần lưu ý chỉ số có hiệu lực 03/08/2026.",
        "n_rows": 2,
        "n_cols": 3,
        "columns": ["STT", "Mã CP", "Tỷ trọng mới (%)"],
        "header_levels": [["", "", "Tỷ trọng"], ["STT", "Mã", "mới"]],
        "unit_notes": ["%"],
        "rows": [
            {"0": "1", "1": "BSR", "2": "0,87%"},
            {"0": "2", "1": "DGC", "2": "5,00%"},
        ],
        "warnings": [],
        "numeric_cols": [2],
        "region_y": (0.0, 50.0),
        "row_index_col": 0,
    }
    base.update(kw)
    return Table(**base)


def test_header_paths_dung_ten_cot_flatten() -> None:
    t = make_table()
    assert header_paths(t) == ["STT", "Mã CP", "Tỷ trọng mới"]


def test_header_paths_loai_placeholder_col_n() -> None:
    t = make_table(columns=["col_0", "Mã CP", "Tỷ trọng"], n_cols=3)
    assert header_paths(t) == ["Cot 0", "Mã CP", "Tỷ trọng"]


def test_header_paths_fallback_header_levels() -> None:
    t = make_table(columns=[])
    assert header_paths(t) == ["STT", "Mã", "Tỷ trọng mới"]


def test_short_caption_cat_ve_cau() -> None:
    cap = short_caption(make_table())
    assert not cap.startswith("Bảng:")
    assert len(cap) <= 111
    assert not cap.endswith(" dữ")


def test_moi_dong_mot_chunk() -> None:
    chunks = row_chunks(make_table())
    assert [c.chunk_id for c in chunks] == ["doc__t01#r0", "doc__t01#r1"]
    assert all(c.kind == "table_row" for c in chunks)
    assert all(c.parent_id == "doc__t01" for c in chunks)


def test_chunk_dong_co_nhan_va_gia_tri_chuan_hoa() -> None:
    row = row_chunks(make_table())[0]
    assert row.label == "Mã CP = BSR"
    assert "Mã CP: BSR" in row.text
    assert "Mã CP = BSR" in row.sparse_text
    cells = {c["label"]: c for c in row.cells}
    assert cells["Mã CP"]["raw"] == "BSR"
    assert cells["Mã CP"]["value"] is None
    assert cells["Tỷ trọng mới"]["value"] == 0.87
    assert cells["Tỷ trọng mới"]["unit"] == "%"


def test_don_vi_khong_bi_nhan_doi() -> None:
    """Raw da chua '%' thi khong cong them '%' nua."""
    row = row_chunks(make_table())[0]
    assert "0,87% %" not in row.text
    assert row.text.count("%") == 1 + row.text.count("Don vi")


def test_phan_tram_va_am_dung_chuan() -> None:
    t = make_table(rows=[{"0": "1", "1": "X", "2": "(69.559)"}])
    cell = row_chunks(t)[0].cells[2]
    assert cell["value"] == -69559.0


def test_o_rong_giu_null_khong_do_nham() -> None:
    t = make_table(rows=[{"0": "1", "1": "MCH", "2": None}])
    chunks = row_chunks(t)
    assert chunks == []


def test_chi_giu_dong_co_it_nhat_mot_so() -> None:
    t = make_table(rows=[{"0": "1", "1": "ABC", "2": None}])
    assert row_chunks(t) == []
    assert row_chunks(t, ChunkConfig(min_row_values=0))


def test_don_vi_bang_khong_bi_gan_nham_vao_o() -> None:
    """Cot 'So phien GD' khong phai nghin co phieu — khong gan don vi cap bang."""
    t = make_table(
        columns=["STT", "Mã CP", "Số phiên GD tương đương"],
        unit_notes=["Nghìn cổ phiếu"],
        rows=[{"0": "1", "1": "TCX", "2": "8,75"}],
    )
    cell = row_chunks(t)[0].cells[2]
    assert cell["unit"] == ""
    assert cell["value"] == 8.75


def test_sparse_text_khong_lap_caption_dai() -> None:
    row = row_chunks(make_table())[0]
    assert "Tổng hợp thay đổi" not in row.sparse_text
    assert "BSR" in row.sparse_text


def test_table_card_lam_chunk_cha() -> None:
    card = table_card(make_table())
    assert card.kind == "table_card"
    assert card.parent_id is None
    assert "Tỷ trọng mới" in card.text
    assert "Nghìn cổ phiếu" not in card.text


def test_table_card_khong_caption_thi_van_co_noi_dung() -> None:
    card = table_card(make_table(caption=""))
    assert card.kind == "table_card"
    assert "Tỷ trọng mới" in card.text


def test_van_ban_dong_nhat() -> None:
    page = PageText(page=1, blocks=[{"text": "Câu một. Câu hai.", "y0": 0, "y1": 1}])
    result = DocumentResult(
        source_file=SRC, n_pages=1, tables=[], pages_text=[page], warnings=[]
    )
    chunks = build_chunks(result)
    assert len(chunks) == 1
    assert chunks[0].kind == "prose"
    assert chunks[0].chunk_id.startswith(SRC)


def test_prose_long_duoc_tach_theo_cau() -> None:
    body = ". ".join(f"Câu số {i} nội dung" for i in range(60)) + "."
    page = PageText(page=4, blocks=[{"text": body, "y0": 0, "y1": 1}])
    result = DocumentResult(
        source_file=SRC, n_pages=1, tables=[], pages_text=[page], warnings=[]
    )
    pieces = build_chunks(result, ChunkConfig(max_prose_chars=200))
    assert len(pieces) > 1
    assert all(len(p.text) <= 260 for p in pieces)
    assert all(p.page == 4 for p in pieces)


def test_block_rong_bi_bo() -> None:
    page = PageText(page=1, blocks=[{"text": "   ", "y0": 0, "y1": 1}])
    result = DocumentResult(
        source_file=SRC, n_pages=1, tables=[], pages_text=[page], warnings=[]
    )
    assert build_chunks(result) == []


def test_build_chunks_day_du_cha_con() -> None:
    page = PageText(page=1, blocks=[{"text": "Văn bản.", "y0": 0, "y1": 1}])
    result = DocumentResult(
        source_file=SRC, n_pages=1, tables=[make_table()], pages_text=[page], warnings=[]
    )
    chunks = build_chunks(result)
    kinds = [c.kind for c in chunks]
    assert kinds == ["table_card", "table_row", "table_row", "prose"]
    ids = {c.chunk_id for c in chunks}
    assert len(ids) == len(chunks)
    assert {c.parent_id for c in chunks if c.kind == "table_row"} == {"doc__t01"}


def test_chunk_to_dict_giu_metadata() -> None:
    data = row_chunks(make_table())[0].to_dict()
    assert data["table_id"] == "doc__t01"
    assert data["row_index"] == 0
    assert isinstance(data["cells"], list)


def test_khong_con_ranh_gia_du_lieu() -> None:
    t = make_table(rows=[{"0": "1", "1": "BSR", "2": "0,87%"}, {"0": "1", "1": "BSR", "2": "0,87%"}])
    chunks = row_chunks(t)
    assert len(chunks) == 2
    assert chunks[0].row_index != chunks[1].row_index


def test_bang_khong_co_cot_stt_thi_cot_so_dau_duoc_giu() -> None:
    """Hoi quy: bang SSI khong co cot STT.

    Neu dem cot nhan theo vi tri, cot 'Tỷ trọng mới' (so) se bi loai khoi
    ``sparse_text`` va gia tri cua no mat khoi chi muc BM25.
    """
    t = make_table(
        columns=["Mã CK", "Tỷ trọng mới", "Tỷ trọng hiện tại"],
        n_cols=3,
        header_levels=[["Mã CK", "Tỷ trọng", "Tỷ trọng"], ["", "mới", "hiện tại"]],
        unit_notes=[],
        row_index_col=None,
        rows=[{"0": "BSR", "1": "0,87%", "2": "0,00%"}],
    )
    chunk = row_chunks(t)[0]
    assert chunk.label == "Mã CK = BSR"
    assert "Tỷ trọng mới" in chunk.sparse_text
    assert "0,87" in chunk.sparse_text


def test_cot_stt_dau_bang_khong_duoc_dung_lam_nhan() -> None:
    t = make_table(
        columns=["STT", "Mã CP", "Tỷ trọng mới"],
        n_cols=3,
        rows=[{"0": "1", "1": "BSR", "2": "0,87%"}],
        row_index_col=0,
    )
    chunk = row_chunks(t)[0]
    assert chunk.label == "Mã CP = BSR"
    assert "Tỷ trọng mới" in chunk.sparse_text


def test_bang_con_ky_tu_o_dau_thi_dung_lam_nhan() -> None:
    t = make_table(
        columns=["Ngày", "Mã CP", "Giá"],
        n_cols=3,
        rows=[{"0": "06/06/2026", "1": "BSR", "2": "12.930,183"}],
        row_index_col=None,
    )
    assert row_chunks(t)[0].label == "Ngày = 06/06/2026"


def test_bang_tat_ca_so_thi_khong_co_nhan() -> None:
    t = make_table(
        columns=["Mã CP", "Tỷ trọng mới"],
        n_cols=2,
        rows=[{"0": "12.345,6", "1": "0,87%"}],
        row_index_col=None,
    )
    chunk = row_chunks(t)[0]
    assert chunk.label == ""
    assert "12.345,6" in chunk.sparse_text


def test_line_word_helpers_still_importable() -> None:
    assert Line is not None and Word is not None