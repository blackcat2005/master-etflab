# etflab — chunking & retrieval cho báo cáo tài chính ETF Việt Nam

Pipeline xử lý báo cáo/phân tích ETF tiếng Việt: parse PDF bằng **`unstructured`**
(`hi_res` + Table Transformer), cắt thành đơn vị truy hồi theo cấu trúc bảng, rồi
trả lời truy vấn bằng **BM25 + vector (RRF)**.

Mục tiêu: câu hỏi dạng *"tỷ trọng mới của BSR là bao nhiêu?"* phải trả lời đúng
**từng ô**, không phải đoạn văn bị cắt vụn.

> **Lưu trữ 2026-10:** parser bảng bằng hình học cũ (`geometry.py`, `tables.py`,
> `pipeline.py`, `chunking.py`, CLI) đã được chuyển vào
> [`archive/geometric_parser/`](archive/geometric_parser/README.md). Nó **không
> còn** nằm trong pipeline đánh giá; mọi kết quả dưới đây chạy trên `unstructured`.
> Gold QA 130 câu vẫn giữ nguyên để so sánh lịch sử; 4/130 giá trị bị
> `unstructured` làm mất được báo cáo riêng (coverage 96,9%).

---

## 1. Cài đặt

Yêu cầu: **Python ≥ 3.11**.

```bash
pip install -r requirements.txt      # numpy + dev tools
# hoặc
pip install -e .
```

Vector search cần thêm (mặc định **tắt**, BM25 vẫn chạy độc lập):

```bash
pip install -e ".[retrieval]"
```

Parser `unstructured` chạy trong **venv riêng** (torch + layout/table model,
~2–3 GB):

```bash
C:/_venvs/unstr/Scripts/python.exe -m pip install "unstructured[pdf]" torch
```

Đánh giá ColPali cần `pymupdf` (có trong `.[eval]`) và `colpali-engine`:

```bash
pip install -e ".[eval]"
pip install colpali-engine
```

> Windows: nếu PowerShell hiện lỗi Unicode khi in bảng tiếng Việt, chạy
> `chcp 65001` hoặc đặt `PYTHONIOENCODING=utf-8`.

---

## 2. Chạy thử

### 2.1 Parse PDF → element + bảng

```bash
# 7 PDF của bộ đánh giá; cache vào data/processed_unstructured/
C:/_venvs/unstr/Scripts/python.exe scripts/ingest/parse_unstructured.py --force
```

Mỗi file cache là JSON gồm element (title, narrative, list, table…) và bảng đã
chuẩn hoá từ `metadata.text_as_html` của Table Transformer. Không dùng toạ độ
chữ hay dò đường kẻ.

### 2.2 Chunking + đánh giá

```bash
# 6 chiến lược: chipper, chipper_prefix, row_json, dual, dual_small, dual_dense
# (dense cần torch; --no-dense chạy được bằng python thường)
C:/_venvs/colpali/Scripts/python.exe scripts/eval/eval_three_methods.py
C:/_venvs/colpali/Scripts/python.exe scripts/eval/eval_three_methods.py --no-dense

# Đo khoảng cách trích xuất của unstructured so với gold (4 câu bị mất giá trị)
C:/_venvs/colpali/Scripts/python.exe scripts/eval/diagnose_unstructured.py

# Trang vàng độc lập parser + ColPali/ColSmol (CPU)
C:/_venvs/colpali/Scripts/python.exe scripts/eval/build_gold_pages.py
C:/_venvs/colpali/Scripts/python.exe scripts/eval/colpali_retrieval.py
```

### 2.3 Test và lint

```bash
python -m pytest        # 107 test
python -m ruff check .
```

---

## 3. Kết quả (130 câu vàng / 7 PDF / 27 trang)

Cùng một gold, cùng BM25; chỉ đổi **đơn vị phân mảnh**. `coverage` = tỷ lệ câu
có ít nhất một chunk chứa đáp án (trần của recall, phản ánh parser/OCR).

### 3.1 Chunk-level (xếp hạng trong file)

| Chiến lược | #chunk | avg chars | coverage | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `row_json` (1 dòng = 1 JSON) | 332 | 491 | 93,8% | **81,5%** | **90,0%** | **90,0%** | **90,0%** | **0,850** |
| `dual_small` | 935 | 228 | 93,8% | 66,1% | 75,4% | 78,5% | 86,2% | 0,723 |
| `dual` | 935 | 228 | 93,8% | 50,0% | 66,9% | 68,5% | 68,5% | 0,583 |
| `chipper_prefix` | 136 | 662 | 93,8% | 36,9% | 67,7% | 77,7% | 89,2% | 0,535 |
| `chipper` | 136 | 557 | 93,8% | 27,7% | 63,8% | 76,9% | 89,2% | 0,475 |
| `dual_dense` (e5-small) | 935 | 228 | 93,8% | 31,5% | 53,1% | 63,8% | 70,8% | 0,441 |

### 3.2 Page-level (max-pool điểm chunk lên trang, xếp hạng 27 trang)

| Chiến lược | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| `row_json` | **90,0%** | **100%** | **100%** | **100%** | **0,950** |
| `dual_small` | 73,9% | 96,2% | 100% | 100% | 0,844 |
| `dual` | 56,1% | 94,6% | 100% | 100% | 0,751 |
| `chipper_prefix` | 45,4% | 89,2% | 99,2% | 100% | 0,665 |
| `chipper` | 36,1% | 88,5% | 99,2% | 100% | 0,618 |
| `dual_dense` | 44,6% | 94,6% | 100% | 100% | 0,700 |

### 3.3 So với ColPali / BM25 trang

| Hệ | R@1 corpus | R@3 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| `row_json` (page) | **90,0%** | **100%** | **100%** | **100%** | **0,950** |
| `dual_small` (page) | 73,9% | 96,2% | 100% | 100% | 0,844 |
| `chipper_prefix` (page) | 45,4% | 89,2% | 99,2% | 100% | 0,665 |
| `colpali:colSmol-256M` | 13,8% | 26,9% | 36,2% | 74,6% | 0,288 |
| `bm25_page` (unstructured text) | 4,6% | 22,3% | 41,5% | 75,4% | 0,227 |

ColSmol-256M là **surrogate CPU** của ColPali-3B, không phải kết luận về bản
gốc. Chi tiết: `docs/ColPali_Thuc_nghiem_Tieng_Viet.md`.

Kết quả thô: `data/eval/three_methods.json`, `data/eval/colpali_results.json`,
`data/eval/extraction_gap.json`, `data/eval/gold_pages.json`.
Tổng hợp + khuyến nghị: `docs/Tong_hop_3_phuong_phap_Chunking_va_ColPali.md`.

---

## 4. Dùng như thư viện

```python
from pathlib import Path

from etflab.chunking_methods import row_json_chunks
from etflab.retrieval import HybridIndex, RetrievalConfig
from etflab.uelements import load_unstructured_doc

# 1. Cache JSON do unstructured sinh ra
doc = load_unstructured_doc(
    Path("data/processed_unstructured/Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.json")
)
print(doc["source_file"], len(doc["elements"]))

# 2. Mỗi dòng bảng = 1 chunk JSON tự chứa (kèm metadata file/page/bảng)
chunks = row_json_chunks(doc, doc["elements"])
print(len(chunks), chunks[0].text[:120])

# 3. Hybrid search (BM25 mặc định; thêm dense: use_dense=True)
index = HybridIndex(chunks, RetrievalConfig(top_k=3))
for hit in index.search("tỷ trọng mới của BSR"):
    print(hit.chunk.chunk_id, hit.chunk.text[:80])
```

Các hàm chunking khác: `chipper_chunks`, `dual_granularity_chunks` (cùng file
`etflab/chunking_methods.py`). `Chunk` là dataclass parser-independent trong
`etflab/chunk.py`.

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

## 5. Cấu trúc project

```
.
├── README.md
├── pyproject.toml             # package + ruff + pytest
├── requirements.txt
│
├── etflab/                    # package hiện tại (không phụ thuộc parser hình học)
│   ├── __init__.py            # API công khai
│   ├── chunk.py               # dataclass Chunk (dùng chung)
│   ├── vnnum.py               # parse số kiểu Việt Nam
│   ├── docmeta.py             # metadata tài liệu (loại, chỉ số, kỳ, ngày)
│   ├── retrieval.py           # BM25 + dense + RRF + lọc theo kỳ
│   ├── uelements.py           # adapter output unstructured (HTML bảng -> lưới)
│   └── chunking_methods.py    # Chipper / dual-granularity / row-JSON
│
├── archive/geometric_parser/  # parser hình học cũ + script/test đi kèm
│   ├── etflab_geometric/      # package tự chứa (geometry/tables/pipeline/chunking/CLI)
│   ├── scripts/               # eval_chunking, eval_row_only, gen_gold_qa, ...
│   ├── tests/                 # test parser cũ (đã đổi import)
│   └── README.md              # vì sao lưu trữ + cách chạy lại
│
├── scripts/
│   ├── ingest/
│   │   └── parse_unstructured.py  # PDF -> element + bảng (hi_res), cache JSON
│   └── eval/                  # đánh giá chunking (không cần GPU/API)
│       ├── build_gold_pages.py     # trang vàng từ text layer PyMuPDF
│       ├── eval_three_methods.py   # Chipper / dual / row-JSON + Recall@k, MRR
│       ├── diagnose_unstructured.py # coverage: đáp án bị mất khi đổi parser
│       ├── metric_common.py        # token hoá + khớp số dùng chung
│       └── colpali_retrieval.py    # retrieval mức TRANG: BM25 vs ColPali/ColSmol
│
├── tests/                     # 107 test
│   ├── conftest.py
│   ├── test_vnnum.py
│   ├── test_docmeta.py        # metadata tài liệu + lọc theo kỳ
│   ├── test_retrieval.py      # BM25 + dense + RRF
│   └── test_metric_matching.py # số âm kế toán, đơn vị '000
│
├── data/
│   ├── raw/                          # PDF nguồn (không commit)
│   ├── references/                   # bài báo khoa học tham khảo
│   ├── eval/                         # gold QA + kết quả đánh giá (không commit)
│   ├── processed/                    # JSON của parser cũ (giữ làm dữ liệu lịch sử)
│   └── processed_unstructured/       # cache element + bảng của unstructured
│
├── docs/
│   ├── Tong_hop_3_phuong_phap_Chunking_va_ColPali.md  # kết quả + khuyến nghị
│   ├── ColPali_Thuc_nghiem_Tieng_Viet.md
│   ├── Co_che_Chipper_Financial_Report_Chunking.md
│   ├── Tong_quan_va_Phuong_phap_Dual_Granularity_RAG.md
│   ├── So_sanh_Chipper_va_cac_phuong_phap_Chunking_2025_2026.md
│   ├── De_cuong_Luan_van_Thac_si_RAG_Phai_sinh.md
│   ├── Huong_dan_ky_thuat.md      # (parser cũ — đã có banner lưu trữ)
│   └── Tham_khao_va_nguon.md
│
└── slides/
    ├── tong_quan_phuong_phap.tex       # slide Beamer (EN): methods overview + deep dive
    ├── dual_granularity_rag.tex        # slide Beamer: dual-granularity
    ├── chipper_element_chunking.tex    # slide Beamer: Chipper + thực nghiệm
    └── colpali_deep_dive.tex           # slide Beamer: ColPali + thực nghiệm
```

> **Biên dịch slide:** cần XeLaTeX (fontspec + font có dấu tiếng Việt).
> ```bash
> tectonic --reruns 2 slides/tong_quan_phuong_phap.tex
> ```

---

## 6. Phương pháp trong một đoạn

- **Parser**: `unstructured` `hi_res` (YOLOX layout + Table Transformer). Bảng
  lấy từ HTML do Table Transformer sinh, `colspan`/`rowspan` được trải thành
  lưới chữ nhật. Không dùng toạ độ chữ hay đường kẻ.
- **Chipper**: 4 quy tắc paper — Title mở chunk, Table giữ nguyên, gộp đoạn
  ngắn < 2048 ký tự, không cắt qua bảng.
- **Dual-granularity**: chunk lớn ~1024 ký tự + chunk nhỏ (câu) ~cửa sổ; truy
  hồi bằng chunk nhỏ, xếp hạng lại bằng trung bình small + large (bản `dual`),
  hoặc chỉ small (`dual_small`), hoặc dense rerank (`dual_dense`).
- **Row-JSON**: mỗi dòng bảng là một JSON tự chứa `{file, page, table, columns,
  row}` — mạnh nhất cho câu hỏi trích ô.
- **Retrieval**: BM25 (mặc định) hoặc hybrid BM25 + dense e5-small, gộp bằng
  RRF; lọc metadata theo kỳ/chỉ số.

---

## 7. Hạn chế đã biết

1. **Gold QA sinh từ parser cũ** (130 câu). `unstructured` giữ được 126/130
   giá trị (96,9%); 4 câu `value_missing` được ghi trong
   `data/eval/extraction_gap.json`, không bị loại khỏi mẫu.
2. **Corpus nhỏ**: 7 PDF / 27 trang / 10 bảng. Chỉ nên đọc như pilot.
3. **Row-JSON thắng lớn nhưng phụ thuộc parser**: nếu Table Transformer sai
   hàng/cột, JSON sai theo. Coverage 93,8% ở chunk-level là trần thực tế.
4. **Dual rerank bằng dense e5-small làm giảm R@1** (66,1% → 50,0%); giá trị
   của dual nằm ở ngữ cảnh small→large cho generation, không phải precision.
5. **ColPali mới chạy ColSmol-256M zero-shot trên CPU** — là sàn, không phải
   kết luận về ColPali-3B / fine-tune.
6. **Chưa đo generation**: mới đo retrieval (Recall@k / MRR), chưa đo câu trả
   lời end-to-end.

---

## 8. Định hướng

- Viết thêm ~20 câu vàng thủ công để giảm bias gold tự sinh.
- Đo generation (PoT/tool-calling) trên chunk row-JSON.
- Thử ColPali-v1.2 / ColQwen2 trên GPU, fine-tune LoRA tiếng Việt.
- Hybrid 2 tầng: visual lấy trang → row-JSON lấy ô.
