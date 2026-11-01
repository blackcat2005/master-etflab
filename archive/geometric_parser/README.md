# Archive — parser bảng bằng hình học

Thư mục này lưu **phiên bản cũ** của `etflab`: tách bảng không viền bằng toạ độ
chữ (PyMuPDF word boxes), dựng cột theo "thung lũng" x-coordinate, ghép header
nhiều tầng, rồi sinh chunk `table_row` / `table_card` / `prose`.

## Vì sao lưu trữ

Parser hình học chỉ đúng với **một họ layout** (bảng không viền, cột thẳng,
header 1–2 tầng). Nó không cover được:

* bảng có ô merge bất quy tắc, cột xiên, bảng lồng;
* trang scan / text layer lệch;
* layout thay đổi giữa các công ty chứng khoán.

Vì vậy nó **không còn nằm trong pipeline đánh giá**. Pipeline hiện tại dùng
`unstructured` (`hi_res` + Table Transformer) làm parser duy nhất; xem
`etflab/uelements.py` và `scripts/ingest/parse_unstructured.py`.

Kết quả cũ vẫn được giữ để đối chiếu lịch sử, nhưng **không còn được trích dẫn
như kết quả chính** trong báo cáo/tổng hợp. Gold QA 130 câu vẫn giữ nguyên
(người dùng chọn giữ để đo coverage); 4 câu mà `unstructured` làm mất giá trị
được báo cáo riêng trong `data/eval/extraction_gap.json`.

## Nội dung

```
etflab_geometric/       # package tự chứa (copy vnnum/docmeta để chạy độc lập)
  geometry.py           # Word/Line, column valleys, group cells
  tables.py             # TableExtractor, TickerExtractor, TableConfig
  chunking.py           # table_card / table_row / prose, build_chunks
  pipeline.py           # process_pdf, IngestConfig, DocumentResult
  cli.py, __main__.py   # CLI cũ (python -m etflab)
  vnnum.py, docmeta.py  # bản copy tại thời điểm lưu trữ
scripts/
  eval_chunking.py      # 4 chiến lược cũ (fixed/element/etflab)
  eval_row_only.py      # ablation chỉ table_row
  gen_gold_qa.py        # sinh gold QA từ ô bảng hình học
  colpali_retrieval.py  # bản ColPali còn baseline etflab
  search_demo.py        # demo PDF -> chunk -> hybrid search
tests/                  # test parser/chunking/pipeline cũ (đã đổi import)
```

`etflab_geometric` **không** được cài bởi `pip install -e .`; đây là mã tham
khảo. Nó cần `pymupdf` (không còn là dependency bắt buộc của package chính).

## Chạy lại (tuỳ chọn)

```bash
# trong venv có pymupdf + pytest
python -m pytest archive/geometric_parser/tests -q
python archive/geometric_parser/scripts/eval_chunking.py
python archive/geometric_parser/scripts/eval_row_only.py
```

Các script đọc dữ liệu đã parse trong `data/processed/` và gold trong
`data/eval/gold_qa.jsonl`, ghi kết quả vào `data/eval/` như cũ.

## Liên hệ với code hiện tại

* `etflab/chunk.py` giữ dataclass `Chunk` (dùng chung cho retrieval và
  `chunking_methods`) — phần này **không** thuộc parser hình học.
* `etflab/chunking_methods.py` (Chipper, dual-granularity, row-JSON) chạy trên
  element của `unstructured`, không import gì từ archive.
* `etflab/retrieval.py` chỉ phụ thuộc `Chunk`, `docmeta`, `vnnum`.
