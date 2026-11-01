# ColPali trên corpus tiếng Việt — thiết kế & kết quả thực nghiệm

Tài liệu này ghi lại thực nghiệm **retrieval ở mức trang** để kiểm tra luận điểm
của paper ColPali (arXiv:2407.01449): *bỏ pipeline parsing, embed thẳng ảnh trang*.
Toàn bộ chạy **CPU-only**, không GPU, không API trả phí.

- Code: `scripts/eval/colpali_retrieval.py`
- Gold QA: `data/eval/gold_qa.jsonl` (130 câu, sinh từ bộ ô bảng cũ — xem hạn chế)
- Trang vàng độc lập parser: `data/eval/gold_pages.json` (sinh bởi
  `scripts/eval/build_gold_pages.py`)
- Kết quả thô: `data/eval/colpali_results.json`
- Slide: `slides/colpali_deep_dive.tex` (Phần III)

> **Cập nhật 2026-10:** parser bảng bằng hình học đã được lưu trữ tại
> `archive/geometric_parser/`. Mọi baseline text trong tài liệu này chạy trên
> element/bảng do `unstructured` (`hi_res` + Table Transformer) parse.

---

## 1. Câu hỏi nghiên cứu

> Với bảng không kẻ tiếng Việt, **visual retrieval zero-shot** (ColPali/ColSmol)
> có vượt được **text retrieval** trên cùng corpus không — và còn cách bao xa
> một pipeline **chunk theo cấu trúc bảng** (row-JSON)?

Đây là câu hỏi *tương đối*: không cần tái lập SOTA ViDoRe, chỉ cần so sánh
công bằng trên cùng corpus, cùng câu hỏi, cùng đơn vị retrieval.

---

## 2. Thiết kế

### 2.1. Đơn vị retrieval = TRANG

ColPali chỉ trả về trang (không chunk, không ô bảng). Ép mọi hệ về **page-level**
là cách duy nhất để so sánh công bằng.

- **Corpus-level** (chính): xếp hạng trên toàn bộ 27 trang của 7 tài liệu.
- **File-level** (phụ): chỉ xếp hạng trong file của câu hỏi — đo khả năng phân
  biệt trang *trong cùng tài liệu*, loại bỏ yếu tố "đoán đúng file".

Với hệ chunk (`row_json`, `chipper`, `dual`), điểm của trang = **max** điểm các
chunk thuộc trang đó (max-pool) — đúng protocol paper dùng khi so sánh với
pipeline parsing.

### 2.2. Dữ liệu

| Thành phần | Số lượng |
|---|---|
| PDF đã parse | 7 (3 broker forecast + 4 SSI review) |
| Trang | 27 |
| Bảng được sinh câu hỏi | 10 |
| Câu hỏi vàng | 130 |

Mỗi câu hỏi có `entity` (mã cổ phiếu), `value_norm` (số đã chuẩn hoá theo quy
tắc Việt Nam), `source_file`, và **tập trang đúng**.

Trang đúng không lấy trực tiếp từ `gold_qa.jsonl` (trường `page` ở đó là trang
*bắt đầu* của bảng, nên dòng nằm ở trang tiếp theo bị chấm sai). Thay vào đó,
`build_gold_pages.py` dùng **text layer của PyMuPDF** — không dùng parser bảng
nào — tìm mọi trang chứa cả mã CK và giá trị:

```bash
python scripts/eval/build_gold_pages.py   # -> data/eval/gold_pages.json
```

Kết quả: 111 câu có 1 trang đúng, 18 câu có 2 trang, 1 câu có 3 trang. Nhờ vậy
các bảng trải qua 2 trang không bị phạt oan.

> PDF thứ 8 (`..._VNDiamond_Q4_2024.pdf`) có trong `data/raw` nhưng chưa parse,
> nên không nằm trong corpus đánh giá.

### 2.3. Các hệ so sánh

| Hệ | Đầu vào | Cách chấm |
|---|---|---|
| `bm25_page` | text của toàn bộ element `unstructured` trên trang (bỏ header/footer) | BM25 trên từng trang |
| `row_json` | mỗi dòng bảng = 1 JSON tự chứa | BM25 trên chunk → max-pool lên trang |
| `chipper` / `chipper_prefix` | 4 quy tắc Chipper trên element `unstructured` | BM25 trên chunk → max-pool lên trang |
| `dual` / `dual_small` | hai tầng: chunk nhỏ + chunk lớn | BM25 trên chunk → max-pool lên trang |
| `colpali:*` | **ảnh trang** 150 DPI | multi-vector + MaxSim |

`bm25_page` là baseline text **công bằng**: nó cũng nhìn thấy nội dung bảng đã
parse, chỉ khác ColPali ở chỗ không dùng ảnh.

### 2.4. Metrics

Recall@{1,3,5,10} và MRR, tính trên **trang** (không phải chunk). Với câu hỏi
có nhiều trang đúng, lấy hạng của trang đúng đầu tiên.

---

## 3. Kết quả

Chạy trên 130 câu hỏi, 27 trang, 7 tài liệu; ảnh trang render 150 DPI;
ColSmol-256M zero-shot, CPU (8 threads).

### 3.1. Corpus-level — xếp hạng trên toàn bộ 27 trang

| Hệ | R@1 | R@3 | R@5 | R@10 | MRR | Thời gian |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_page` (text + bảng đã parse) | 4,6% | 22,3% | 41,5% | 75,4% | 0,227 | 0,1 s |
| `chipper` (cả bảng 1 chunk) | 36,1% | 88,5% | 99,2% | 100% | 0,618 | 1 s |
| `chipper_prefix` | 45,4% | 89,2% | 99,2% | 100% | 0,665 | 1 s |
| `dual_small` | 73,9% | 96,2% | 100% | 100% | 0,844 | 2 s |
| `row_json` (1 dòng = 1 chunk) | **90,0%** | **100%** | **100%** | **100%** | **0,950** | 2 s |
| `colpali:colSmol-256M` | 13,8% | 26,9% | 36,2% | 74,6% | 0,288 | ~25 s (đã cache ảnh) |

> Các số `row_json` / `chipper` là **page-level** lấy từ
> `data/eval/three_methods.json` (max-pool điểm chunk lên trang, cùng
> `gold_pages.json`).

### 3.2. File-level — chỉ xếp hạng trong file của câu hỏi

| Hệ | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| `bm25_page` | 17,7% | 90,8% | 99,2% | 100% | 0,528 |
| `colpali:colSmol-256M` | 20,8% | 88,5% | 100% | 100% | 0,547 |

> File-level chỉ chạy cho hai hệ page thuần. Các hệ chunk được báo ở
> corpus-level để tránh so sánh khập khiễng (chúng đã có metadata file/page
> trong chunk).

> Chi phí: lần đầu embed 27 trang ~6--7 phút trên CPU; các lần sau dùng
> cache (`data/eval/cache/*.pt`), chỉ còn ~25 s embed 130 câu + chấm điểm.

---

## 4. Phân tích

### 4.1. ColPali vượt BM25 trang, nhưng thua xa chunk có cấu trúc

ColSmol đạt **13,8% R@1**, cao hơn `bm25_page` (4,6%) — tức tín hiệu ảnh có
thật, không phải nhiễu. Nhưng so với `row_json` (90,0% R@1, MRR 0,950) khoảng
cách là **6,5 lần**. Ở MRR: 0,288 so với 0,227 (BM25) và 0,950 (row-JSON).

Điểm mấu chốt không phải "visual hay text", mà là **đơn vị truy hồi**:
`row_json` đưa ra 332 ứng viên, mỗi ứng viên là một dòng bảng tự chứa mã CK +
giá trị; ColPali chỉ có 27 ứng viên trang, và phải giải mã đúng ô trong ảnh.

### 4.2. Tín hiệu có thật nhưng không đủ để dùng

R@10 = 74,6% nghĩa là trang đúng thường nằm trong top 10/27 — mô hình không
"mù", nhưng tín hiệu yếu. File-level R@5 = 100% cho thấy nó **nhận ra tài
liệu**, nhưng vẫn yếu ở việc **chọn đúng trang** (R@1 20,8%). Với RAG, lấy 10
trang vào prompt là quá nhiều và không giải quyết được câu hỏi "ô nào trong bảng".

### 4.3. Vì sao yếu — 4 nguyên nhân cộng dồn

1. **Surrogate 256M**: ColPali-3B có ~12× tham số; R@1 của bản 256M không
   đại diện cho ColPali. Kết quả này là *sàn* của họ late-interaction, không
   phải kết luận về ColPali.
2. **Train 100% tiếng Anh**: chữ Việt có dấu, font nhỏ trong bảng; model
   chưa từng thấy phân phối này.
3. **27 trang gần như giống nhau**: cùng template bảng "Mã | Tỷ trọng |
   KL", khác nhau vài chữ số. Ở mức ảnh 150 DPI, khác biệt chủ yếu là **chữ
   nhỏ**; embedding patch 512×512 bị "hòa" giữa các trang.
4. **Không fine-tune, không query augmentation**: paper gốc dùng LoRA trên
   chính miền; zero-shot là điều kiện khắc nghiệt nhất.

### 4.4. Hệ quả cho đồ án

- Visual retrieval **không thay thế** được pipeline parse + chunk theo ô trên
  corpus này. Nó chỉ nên là **tầng 1** (chọn trang) sau khi fine-tune tiếng Việt.
- Nếu cần trả lời đúng ô, `row_json` là lựa chọn mạnh nhất trong các phương
  pháp không dùng parser hình học: 81,5% R@1 ở chunk-level, 90,0% ở page-level.
- Muốn đánh giá đúng ColPali phải chạy `vidore/colpali-v1.2` (3B) trên GPU,
  hoặc fine-tune LoRA; xem mục 7.

---

## 5. Hạn chế (phải nêu trong luận văn)

1. **Gold QA tự sinh** từ ô bảng của parser hình học cũ (đã lưu trữ). Giữ
   nguyên 130 câu để so sánh lịch sử, nhưng 4/130 giá trị **không còn** trong
   output `unstructured` (coverage 96,9%; xem `data/eval/extraction_gap.json`).
   Đây là thiên lệch thiết kế, không phải bằng chứng visual retrieval kém.
2. **Trang vàng** lấy từ text layer PyMuPDF (entity + value), không từ parser
   bảng; vẫn có thể trùng trang do giá trị xuất hiện ở nhiều bảng.
3. **Corpus nhỏ** (27 trang, 130 câu). Khoảng tin cậy rộng; chỉ nên đọc như pilot.
4. **ColSmol-256M ≠ ColPali-3B**. Đây là surrogate chạy được trên CPU; kết luận
   về ColPali bản gốc cần chạy thêm.
5. **Zero-shot tiếng Việt**: cả ColPali và ColSmol đều train 100% tiếng Anh.
6. **Page-level** không đo được khả năng trả về đúng *ô bảng* — điều mà
   `row_json` làm được và là lợi thế thật của pipeline hiện tại.
7. **Không đo generation**: chỉ đo retrieval, chưa đo end-to-end RAG.

---

## 6. Cách chạy

```bash
# 0) Trang vàng độc lập parser (cần pymupdf)
python scripts/eval/build_gold_pages.py

# 1) Baseline text (không cần model ảnh)
python scripts/eval/colpali_retrieval.py --no-colpali

# 2) ColSmol-256M (CPU, ~10-15 phút cho 27 trang lần đầu)
python scripts/eval/colpali_retrieval.py --model vidore/colSmol-256M --torch-threads 8

# 3) ColPali-v1.2 (nặng hơn nhiều; cần ~6 GB RAM/weights)
python scripts/eval/colpali_retrieval.py --model vidore/colpali-v1.2 --dtype bfloat16
```

Môi trường đã dùng: Python 3.11, `colpali-engine`, `pymupdf`, torch CPU.
Lưu ý: Hugging Face Xet tải bị treo trên máy này → đặt `HF_HUB_DISABLE_XET=1`.

---

## 7. Hướng tiếp

- Chạy ColPali-v1.2 / ColQwen2 trên toàn bộ 27 trang để kiểm tra kết luận.
- Fine-tune LoRA trên cặp (câu hỏi tiếng Việt, ảnh trang) — paper cho thấy chỉ
  1.5k mẫu đã cải thiện rõ.
- Ánh xạ patch → bounding box để trả về đúng bảng, thay vì cả trang.
- Hybrid 2 tầng: visual retrieval lấy trang → `row_json` / bảng đã parse lấy ô.
