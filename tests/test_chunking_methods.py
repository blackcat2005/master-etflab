"""Test cho 3 cach chunking tren output ``unstructured`` (khong can model)."""

from __future__ import annotations

import json

from etflab.chunking_methods import (
    MethodConfig,
    chipper_chunks,
    dual_granularity_chunks,
    row_json_chunks,
)
from etflab.uelements import UElement, UTable, parse_table_html

SRC = "Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf"

HTML = """
<table>
  <thead>
    <tr><th>Mã</th><th colspan="2">Tỷ trọng (%)</th></tr>
    <tr><th></th><th>Q2</th><th>Q3</th></tr>
  </thead>
  <tbody>
    <tr><td>MBB</td><td>21.094</td><td>20.598</td></tr>
    <tr><td>ACB</td><td>1,2</td><td>1,3</td></tr>
    <tr><td>Tổng</td><td></td><td></td></tr>
  </tbody>
</table>
"""


def make_doc() -> dict:
    table = parse_table_html(HTML, page=2, caption="Bảng 1: Danh mục",
                             table_id="t1")
    elements = [
        UElement("title", "Bảng 1: Danh mục", 2),
        UElement("table", table.to_text(), 2, table=table),
        UElement("narrative", "Đoạn một. Đoạn hai.", 3),
    ]
    return {"source_file": SRC, "elements": elements}


# ---------------------------------------------------------------------------
# HTML -> luoi.
# ---------------------------------------------------------------------------
def test_parse_colspan_header() -> None:
    table = parse_table_html(HTML, page=1, table_id="t")
    assert table.columns == ["Mã", "Tỷ trọng (%) / Q2", "Tỷ trọng (%) / Q3"]
    assert table.rows[0] == ["MBB", "21.094", "20.598"]
    assert table.n_rows == 3


def test_parse_rowspan() -> None:
    html = (
        "<table><thead><tr><th>Nhóm</th><th>Mã</th><th>KL</th></tr></thead>"
        "<tbody><tr><td rowspan='2'>A</td><td>MBB</td><td>1</td></tr>"
        "<tr><td>ACB</td><td>2</td></tr></tbody></table>"
    )
    table = parse_table_html(html, table_id="t")
    assert table.rows == [["A", "MBB", "1"], ["", "ACB", "2"]]


def test_parse_empty_html() -> None:
    table = parse_table_html("", table_id="t")
    assert table.rows == []
    assert table.n_cols == 0


# ---------------------------------------------------------------------------
# Chipper.
# ---------------------------------------------------------------------------
def test_chipper_table_is_whole_and_separate() -> None:
    doc = make_doc()
    chunks = chipper_chunks(doc, doc["elements"])
    tables = [c for c in chunks if c.kind == "chipper_table"]
    assert len(tables) == 1
    assert "MBB" in tables[0].text and "ACB" in tables[0].text
    assert "Bảng 1" in tables[0].text


def test_chipper_title_starts_new_chunk() -> None:
    doc = make_doc()
    elements = [
        UElement("narrative", "Trước tiêu đề.", 1),
        UElement("title", "Tiêu đề mới", 1),
        UElement("narrative", "Sau tiêu đề.", 1),
    ]
    chunks = chipper_chunks(doc, elements, MethodConfig(chipper_target=10_000))
    assert len(chunks) == 2
    assert chunks[0].text == "Trước tiêu đề."
    assert chunks[1].text.startswith("Tiêu đề mới")


def test_chipper_never_splits_element_and_respects_target() -> None:
    doc = make_doc()
    long_text = "x" * 900
    elements = [UElement("narrative", long_text, 1)] * 5
    chunks = chipper_chunks(doc, elements, MethodConfig(chipper_target=2048))
    # 2 phan tu = 1800 < 2048; phan tu thu 3 day qua nguong -> chunk moi.
    assert len(chunks) == 3
    assert all(long_text in c.text for c in chunks)


def test_chipper_prefix_uses_caption_for_table() -> None:
    doc = make_doc()
    chunks = chipper_chunks(doc, doc["elements"], add_prefix=True)
    table_chunk = next(c for c in chunks if c.kind == "chipper_table")
    assert table_chunk.text.splitlines()[0] == "Bảng 1: Danh mục"


# ---------------------------------------------------------------------------
# Dual granularity.
# ---------------------------------------------------------------------------
def test_dual_parent_child_links() -> None:
    doc = make_doc()
    small, large = dual_granularity_chunks(doc, doc["elements"])
    large_ids = {c.chunk_id for c in large}
    assert small and all(c.parent_id in large_ids for c in small)
    # Bang: moi dong co so -> 1 khoi nho; dong "Tổng" khong co so -> bo.
    rows = [c for c in small if c.kind == "dual_row"]
    assert len(rows) == 2


def test_dual_merges_sentences_with_embed() -> None:
    doc = make_doc()
    elements = [
        UElement("narrative", "Câu một. Câu hai. Câu ba.", 1),
    ]
    same = [[1.0, 0.0] for _ in range(3)]
    small, _ = dual_granularity_chunks(
        doc, elements, MethodConfig(small_theta=0.9), embed=lambda texts: same
    )
    assert len(small) == 1
    assert small[0].text == "Câu một. Câu hai. Câu ba."

    diff = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    small2, _ = dual_granularity_chunks(
        doc, elements, MethodConfig(small_theta=0.9), embed=lambda texts: diff
    )
    assert len(small2) == 3


def test_dual_doc_label_in_sparse_text() -> None:
    doc = make_doc()
    small, _ = dual_granularity_chunks(doc, doc["elements"])
    assert "q2" in small[0].sparse_text.lower()


# ---------------------------------------------------------------------------
# Row JSON.
# ---------------------------------------------------------------------------
def test_row_json_is_valid_and_self_contained() -> None:
    doc = make_doc()
    chunks = row_json_chunks(doc, doc["elements"])
    assert len(chunks) == 2
    payload = json.loads(chunks[0].text)
    assert payload["page"] == 2
    assert payload["table"].startswith("Bảng 1")
    assert payload["row"]["Mã"] == "MBB"
    assert payload["row"]["Tỷ trọng (%) / Q2"] == "21.094"


def test_row_json_skips_rows_without_numbers() -> None:
    doc = make_doc()
    texts = [json.loads(c.text)["row"]["Mã"] for c in row_json_chunks(doc, doc["elements"])]
    assert texts == ["MBB", "ACB"]


def test_table_to_text_has_header_and_rows() -> None:
    table: UTable = make_doc()["elements"][1].table
    text = table.to_text()
    assert "Mã | Tỷ trọng (%) / Q2" in text
    assert "MBB | 21.094 | 20.598" in text
