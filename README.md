# etflab — PDF → JSON → chunks → hybrid retrieval

Pipeline xử lý báo cáo tài chính ETF Việt Nam: đọc PDF **không có đường viền
bảng**, dựng lại bảng thành JSON có cấu trúc, cắt thành đơn vị truy hồi, rồi
trả lời truy vấn bằng **BM25 + vector search** hợp nhất bằng RRF.

Mục tiêu: câu hỏi dạng *"tỷ trọng mới của BSR là bao nhiêu?"* phải trả lời
đúng **từng ô**, không phải đoạn văn bị cắt vụn.

---

## 1. Cài đặt

Yêu cầu: **Python ≥ 3.11**.

```bash
pip install -r requirements.txt
```

Hoặc cài dạng package (tạo lệnh `etflab`):

```bash
pip install -e .
```

Vector search cần thêm (mặc định **tắt**, BM25 vẫn chạy độc lập):

```bash
pip install -e ".[retrieval]"
```

> Windows: nếu PowerShell hiện lỗi Unicode khi in bảng tiếng Việt, chạy
> `chcp 65001` hoặc đặt `PYTHONIOENCODING=utf-8`.

---

## 2. Chạy thử

### 2.1 Ingest — PDF sang JSON

```bash
# Toàn bộ data/raw/ (mặc định)
python -m etflab

# Một file cụ thể
python -m etflab "data/raw/ssi_review_announcements/Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf"

# Ghi ra chỗ khác, không in chi tiết
python -m etflab data/raw -o /tmp/out --quiet

# Khi đã cài package
etflab data/raw
```

Kết quả in ra dạng:

```
[+] data
aw\ssi_review_announcements\Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf: 3 trang, 1 bang, 24 khoi van ban
      bao_cao_phan_tich_ssi_etf_vn30__t01  p[1, 2]  31x7
        cols=['Mã CK', 'Tỷ trọng mới', 'Tỷ trọng hiện tại', '% Thay đổi', ...]
```

JSON được ghi vào `data/processed/<tên file>.json`.

### 2.2 Truy vấn — demo hybrid search

```bash
# Bốn câu hỏi mẫu
python scripts/search_demo.py

# Câu hỏi riêng
python scripts/search_demo.py --query "số phiên giao dịch tương đương của MCH"

# Thêm vector search (lần đầu sẽ tải model, khá lâu)
python scripts/search_demo.py --dense
```

Đầu ra:

```
>>> tỷ trọng mới của BSR
  1. [table_row] score=0.0164  (trang 1 | bao_cao_phan_tich_ssi_etf_vn30__t01)
     ... Mã CK: BSR; Tỷ trọng mới: 0,87%; Tỷ trọng hiện tại: 0,00%; ...
     [cha] Bảng: Ước tính thay đổi về số lượng cổ phiếu mua bán ...
```

### 2.3 Test và lint

```bash
python -m pytest        # 187 test, tat ca pass
python -m ruff check .
```

---

## 3. Dùng như thư viện

```python
from pathlib import Path
from etflab import IngestConfig, build_chunks, process_pdf
from etflab.retrieval import HybridIndex, RetrievalConfig

# 1. PDF -> bang + van ban
result = process_pdf(Path("data/raw/ssi_review_announcements/Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf"),
                     IngestConfig())
table = result.tables[0]
print(table.n_rows, table.n_cols)          # 31 7
print(table.columns[1])                    # 'Tỷ trọng mới'
print(table.rows[0][4])                    # '3.274.147'

# 2. -> don vi truy hoi (moi dong = 1 chunk)
chunks = build_chunks(result)
print(len(chunks), "chunks")

# 3. -> hybrid search
index = HybridIndex(chunks, RetrievalConfig(top_k=3))   # BM25 (mac dinh)
# Them vector search: RetrievalConfig(top_k=3, use_dense=True)
for hit in index.search("tỷ trọng mới của BSR"):
    print(hit.chunk_id, hit.chunk.text[:80])
```

### Quy tắc số Việt Nam

`parse_number` xử lý đúng định dạng Việt Nam — dùng lại cho bước sinh SQL/PoT sau này:

| Chuỗi | Kết quả | Ghi chú |
|---|---|---|
| `2.976,9` | `2976.9` | dấu chấm = phân tách nghìn |
| `(69.559)` | `-69559.0` | ngoặc = số âm |
| `-1,3` | `-1.3` | |
| `0,87%` | `0.87`, `unit="percent"` | giữ nguyên độ lớn, đơn vị ở cờ riêng |
| `30/06/2026` | `None` | ngày tháng **không** phải số |

---

## 4. Cấu trúc project

```
.
├── README.md                  # file này
├── pyproject.toml             # cấu hình package + ruff + pytest
├── requirements.txt           # cài đặt nhanh
│
├── etflab/                    # mã nguồn
│   ├── __main__.py            # python -m etflab
│   ├── cli.py                 # CLI Phase 0
│   ├── vnnum.py               # Phase 0 — parse số kiểu Việt Nam
│   ├── geometry.py            # Phase 0 — gom dòng/ô, dựng ranh giới cột
│   ├── tables.py              # Phase 0 — phát hiện bảng, header nhiều tầng, tách bảng cạnh nhau
│   ├── pipeline.py            # Phase 0 — điều phối, caption, gộp bảng nhiều trang, cảnh báo OCR
│   ├── chunking.py            # Phase 1 — table_card / table_row / prose chunks
│   ├── docmeta.py             # Phase 1 — metadata tài liệu (loại, chỉ số, kỳ, ngày)
│   └── retrieval.py           # Phase 1 — BM25 + dense + RRF + lọc theo kỳ
│
├── scripts/
│   └── search_demo.py         # chạy thử hybrid search
│
├── tests/                     # 187 test, tat ca pass
│   ├── conftest.py
│   ├── test_vnnum.py
│   ├── test_tables.py
│   ├── test_pipeline.py       # golden test trên PDF thật
│   ├── test_corpus.py         # hồi quy toàn corpus + các lỗi đã sửa
│   ├── test_chunking.py
│   ├── test_docmeta.py        # metadata tài liệu + lọc theo kỳ
│   └── test_retrieval.py
│
├── data/
│   ├── raw/                          # PDF nguồn — 3 nhóm tài liệu (không commit)
│   │   ├── legal_charters/           # điều lệ, cáo bạch, thông báo HOSE (scan — cần OCR)
│   │   ├── ssi_review_announcements/  # thông báo rà soát / thay đổi chỉ số
│   │   └── broker_forecast_reports/  # báo cáo / bản tin dự báo của CTCK
│   ├── references/                   # bài báo khoa học tham khảo (không ingest)
│   └── processed/                    # JSON đầu ra (không commit)
│
└── docs/
    ├── RAG Bảng Biểu Tài Chính.md
    ├── Tong_quan_va_Phuong_phap_Dual_Granularity_RAG.md
    ├── De_cuong_Luan_van_Thac_si_RAG_Phai_sinh.md
    ├── Ky_thuat_da_su_dung.md
    ├── Ky_thuat_va_Tham_khao.md     # đối chiếu từng kỹ thuật với bài báo (A/B/C)
    └── Tai_lieu_tham_khao.md
```

---

## 5. Giai đoạn đã hoàn thành

### Phase 0 — PDF → JSON

Không dùng `pymupdf.find_tables()` vì bảng trong báo cáo tài chính **không có
đường viền**. Pipeline tự dựng lại:

- Gom từ thành dòng → ô, đo bề rộng ký tự để ước lượng ngưỡng;
- Dựng ranh giới cột bằng khe trống dọc, siết lại để header căn giữa khớp dữ liệu;
- Nhận diện vùng bảng bằng mật độ ô số, loại văn bản justify;
- **Giữ dòng có ô rỗng** — dòng thiếu số vẫn là dòng dữ liệu hợp lệ;
- Dựng header nhiều tầng, ghép tên cột qua các tầng (`header_levels`);
- Tách **bảng cạnh nhau** thành hai bảng riêng;
- Gộp bảng **nối trang** theo chữ ký cột + khoảng ngang;
- Bắt caption (`Bảng 2`, `Table 2`), loại running header/footer;
- Cảnh báo PDF scan thay vì trả về kết quả rỗng.

Đối chiếu với dữ liệu thật:

| PDF | Kỳ vọng | Kết quả |
|---|---|---|
| `ssi_review_announcements/Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf` | 1 bảng, trang 1–2, 31×7, dòng đầu `BSR`, cuối `DGC` | ✅ |
| `broker_forecast_reports/Du_bao_thay_doi_danh_muc_chi_so_VN30_VNFINLEAD_Q3_2026.pdf` | 2 bảng cạnh nhau, trang 2–3, 41×6 và 41×5 | ✅ |
| `broker_forecast_reports/Du_bao_thay_doi_danh_muc_chi_so_VN30_VNFINLEAD_Q1_2026.pdf` | 2 bảng cạnh nhau (khe chỉ 12pt), trang 3, 40×6 và 40×5 | ✅ |

### Phase 1 — chunks + hybrid retrieval

- **`table_row`** — mỗi dòng dữ liệu là một chunk, kèm nhãn dòng (`Mã CK = BSR`),
  giá trị đã chuẩn hóa và ô rỗng được giữ là `null` (không đoán bừa).
- **`table_card`** — chunk cha mô tả bảng, dùng làm context khi sinh câu trả lời
  và cho câu hỏi mức tổng quan.
- **`prose`** — văn bản ngoài bảng, tách theo câu có overlap.
- Mỗi chunk có **hai bản text**: `text` đầy đủ cho embedding, `sparse_text` rút
  gọn cho BM25 — cùng một chunk nhưng hai cách nhìn (dual granularity).
- **BM25 tự hiện thực**, tối ưu cho tiếng Việt tài chính: bỏ dấu, `đ→d`, và
  chuẩn hóa số sao cho hỏi `2976,9` vẫn khớp `2.976,9`.
- **RRF** hợp nhất BM25 + vector, không cần chuẩn hóa thang đo hai bên.

---

## 6. Giới hạn đã biết

| Vấn đề | Trạng thái |
|---|---|
| PDF scan (điều lệ quỹ, 95 trang BCB) | Chỉ phát cảnh báo **cần OCR**; chưa tích hợp OCR |
| Bài báo tiếng Anh | Tên cột bảng kém chất lượng, tách thành nhiều mảnh; ngoài corpus mục tiêu |
| Ô trộn nhiều tầng lền nhau | Giới hạn ở 2 bảng cạnh nhau (`max_cols_single_table = 8`) |
| Bảng có đường viền | Chưa tận dụng `find_tables()`; hiện vẫn parse hình học |
| Bảng cạnh nhau có khe hẹp hơn khe trong bảng | Đã sửa: dòng tiêu đề nhóm `Kịch bản 1/2` trong header làm bằng chứng, không phụ thuộc độ rộng khe |
| Hỏi bản Q2 lại ra bản Q3 | Đã sửa: chunk mang metadata kỳ, lọc cứng qua `infer_doc_filter` |
| Chưa đo Recall@k | Chưa có bộ câu hỏi vàng |

---

## 7. Định hướng tiếp theo

1. **OCR** cho PDF scan (điều lệ quỹ) — mở rộng corpus.
2. **PoT (Program-of-Thought)**: sinh SQL/Python trên `Chunk.cells` để trả lời
   câu hỏi so sánh/lọc (*"quỹ nào bán mạnh nhất?"*).
3. **Reranker** (cross-encoder) sau khi RRF.
4. **Đánh giá**: bộ câu hỏi vàng + đo Recall@k / MRR trên cả hai PDF mục tiêu.