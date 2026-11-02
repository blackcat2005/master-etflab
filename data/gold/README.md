# Gold QA (có version)

| File | Nội dung |
|---|---|
| `gold_qa_v2.jsonl` | 615 câu hỏi ô bảng (gồm 129 qid duy nhất của gold cũ (130 dòng, 1 qid trùng) + 486 câu mới) |
| `gold_qa_v2.stats.json` | Thống kê phân bố theo file, kỳ/chỉ số, mẫu câu |

Sinh bằng `scripts/eval/enrich_gold.py` (seed 42, `--per-table 60`). Gold cũ ở `data/eval/` bị `.gitignore`, nên bản v2 đặt ở đây để được version.

## Trường
`qid, query, source_file, table_id, page, row_index, col_index, entity, col_label, value_raw, value_norm, period, index` như gold cũ, thêm:

| Trường | Ý nghĩa |
|---|---|
| `source` | `orig` (có trong gold cũ) hoặc `new` |
| `variant` | `full`, `natural`, `no_index`, `hard` (mẫu câu hỏi) |
| `hard` | `true` nếu câu bỏ kỳ và chỉ số, mơ hồ; loại bằng `--exclude-hard` |
| `multi_period` | cùng (mã, cột, chỉ số) có ở kỳ khác, dùng đo lỗi trả nhầm kỳ |
| `gold_pages` | mọi trang PDF có cả mã và giá trị (text layer PyMuPDF) |
| `text_layer_ok` | mã và giá trị cùng nằm trên một trang |

## Giới hạn
- Giá trị lấy từ parser hình học cũ, chưa ai kiểm tay. `text_layer_ok` chỉ chứng minh mã và số cùng xuất hiện trên một trang, không chứng minh đúng hàng, đúng cột.
- Phân bố lệch: Q1 2026 VNDiamond chiếm 227/615 câu.
- Vẫn cùng 7 PDF, 10 bảng; chỉ tăng số câu và độ đa dạng câu hỏi, không tăng số tài liệu.
