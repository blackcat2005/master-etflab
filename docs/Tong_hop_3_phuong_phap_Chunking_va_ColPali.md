# TỔNG HỢP THỰC NGHIỆM: DUAL-GRANULARITY, CHIPPER VÀ COLPALI TRÊN BÁO CÁO TÀI CHÍNH VIỆT NAM

*Tài liệu tổng hợp cho luận văn. Các phương pháp được so sánh trên **cùng một bộ dữ liệu** (7 PDF / 27 trang / 130 câu hỏi vàng) và **cùng một parser** — `unstructured` (`hi_res` + Table Transformer). Parser bảng bằng hình học cũ đã được lưu trữ tại `archive/geometric_parser/` và **không còn** là baseline trong tài liệu này.*

> **Liên quan:**
> - [`Co_che_Chipper_Financial_Report_Chunking.md`](./Co_che_Chipper_Financial_Report_Chunking.md) — cơ chế paper Chipper.
> - [`Tong_quan_va_Phuong_phap_Dual_Granularity_RAG.md`](./Tong_quan_va_Phuong_phap_Dual_Granularity_RAG.md) — paper Peng et al.
> - [`ColPali_Thuc_nghiem_Tieng_Viet.md`](./ColPali_Thuc_nghiem_Tieng_Viet.md) — protocol và kết quả ColPali chi tiết.
> - [`So_sanh_Chipper_va_cac_phuong_phap_Chunking_2025_2026.md`](./So_sanh_Chipper_va_cac_phuong_phap_Chunking_2025_2026.md) — bối cảnh học thuật 2024–2026.
> - Kết quả thô: `data/eval/three_methods.json`, `data/eval/colpali_results.json`,
>   `data/eval/extraction_gap.json`, `data/eval/gold_pages.json`.

---

## 0. TL;DR

Trên 130 câu hỏi vàng (7 PDF, 27 trang, 10 bảng), **đơn vị truy hồi quyết định phần lớn kết quả**:

| Hệ | R@1 chunk | R@1 page | MRR chunk | MRR page |
|---|---:|---:|---:|---:|
| **`row_json`** (1 dòng = 1 JSON) | **81,5%** | **90,0%** | **0,850** | **0,950** |
| `dual_small` (câu + dòng) | 66,1% | 73,9% | 0,723 | 0,844 |
| `dual` (rerank + ngữ cảnh lớn) | 50,0% | 56,1% | 0,583 | 0,751 |
| `chipper_prefix` | 36,9% | 45,4% | 0,535 | 0,665 |
| `chipper` (cả bảng 1 chunk) | 27,7% | 36,1% | 0,475 | 0,618 |
| `dual_dense` (e5-small) | 31,5% | 44,6% | 0,441 | 0,700 |
| BM25 page (text `unstructured`) | — | 4,6% | — | 0,227 |
| ColSmol-256M (visual) | — | 13,8% | — | 0,288 |

Năm kết luận:

1. **Một dòng = một chunk là đơn vị đúng** cho câu hỏi ô bảng. `row_json` đạt 81,5% R@1 (chunk) và 90,0% (page), MRR 0,850/0,950 — gấp **~3 lần** Chipper ở R@1.
2. **Chipper không hợp bảng số**: R@1 27,7% dù R@10 89,2%; `chars@1` 2.299 so với 494 — chunk cả bảng chỉ "chứa", không "trả lời". Prefix caption chỉ +9,2 điểm.
3. **Dual-granularity là chiến lược cho generation, không phải precision**: `dual_small` (66,1%) thua `row_json` 15,4 điểm; rerank bằng khối lớn làm giảm thêm 16,1; dense e5-small còn 31,5%. Small2big hữu ích khi LLM cần ngữ cảnh, không giúp tìm đúng ô.
4. **ColPali zero-shot 256M vượt BM25 trang nhưng còn xa chunk có cấu trúc**: 13,8% R@1 page (BM25 4,6%), nhưng chỉ bằng **1/6,5** `row_json`. Visual là bộ chọn trang, chưa phải bộ trả lời ô.
5. **Giới hạn parser là thật và được định lượng**: `unstructured` giữ 126/130 giá trị (96,9%) ở mức element, 122/130 (93,8%) ở mức chunk. Bốn giá trị bị mất được báo cáo riêng, không bị loại khỏi mẫu.

---

## 1. Câu hỏi và phạm vi

Câu hỏi của luận văn: **với bảng số không viền trong báo cáo tài chính tiếng Việt, đơn vị truy hồi nào trả lời đúng câu hỏi "giá trị của mã X là bao nhiêu" với chi phí thấp nhất?**

Các ứng viên đại diện cho các triết lý khác nhau:

| Triết lý | Phương pháp | Đơn vị truy hồi |
|---|---|---|
| Chia theo **cấu trúc tài liệu** | Chipper (arXiv:2402.05131) | phần tử (title/table/prose) gộp tới 2.048 ký tự |
| Chia theo **hai mức ngữ nghĩa** | Dual-granularity (Peng et al., ESWA 2026) | câu / dòng (nhỏ) → đoạn (lớn) |
| **Không chia** — truy hồi trên ảnh | ColPali (arXiv:2407.01449) | cả trang PDF |
| **1 dòng = 1 object** (biến thể text) | Row-JSON | một dòng bảng tự chứa |

**Trong phạm vi:** 7 PDF (3 báo cáo broker + 4 bản tin SSI) đã có text layer, 27 trang, 10 bảng gold.
**Ngoài phạm vi:** PDF scan thuần ảnh, bảng có ô merge phức tạp, câu hỏi cần tổng hợp nhiều bảng.

> **Lưu trữ:** parser hình học cũ từng được dùng để tạo gold QA và làm baseline. Nó đã được chuyển vào `archive/geometric_parser/`; các số liệu của nó **không còn** xuất hiện như kết quả chính. Gold QA 130 câu được giữ nguyên để bảo toàn lịch sử so sánh.

---

## 2. Các phương pháp

### 2.1. Dual-granularity (Peng et al.)

**Ý tưởng:** tách tài liệu thành hai tầng song song.

- **Khối lớn** (large chunks): đoạn văn hoàn chỉnh, gộp các đoạn liền kề tới `γ = 1.024` ký tự; `Title` và `Table` là ranh giới cứng (bảng giữ nguyên nên có thể vượt `γ`, giống Chipper). Khối lớn là ngữ cảnh đưa cho LLM.
- **Khối nhỏ** (small chunks): câu (với văn bản) hoặc **từng dòng** (với bảng). Hai câu liền nhau được gộp nếu `cosine ≥ θ = 0.9`.
- **Truy hồi:** xếp hạng khối nhỏ; **rerank** bằng trung bình điểm khối nhỏ và điểm khối lớn chứa nó; cuối cùng trả khối lớn (small2big).

**Cách hiện thực trong đồ án** (`etflab/chunking_methods.py`):

| Thành phần | Paper | Đồ án | Lý do |
|---|---|---|---|
| Parser | PDF → paragraph (LLM phân loại) | `unstructured` hi_res + Table Transformer | không dùng parser hình học; tái sử dụng cho cả 3 phương pháp |
| Embed câu | `bge-large-zh-v1.5` | `intfloat/multilingual-e5-small` (384d, CPU) | đã có cache, hỗ trợ tiếng Việt, không cần GPU |
| Gộp đoạn lớn | LLM phân loại | liền kề + `γ = 1.024` + ranh giới Title/Table | bỏ LLM để tái lập được, giữ nguyên ngưỡng |
| Rerank | `½(cos nhỏ + cos lớn)` | `½(BM25 nhỏ + BM25 lớn)` (sparse) **và** `½(cos nhỏ + cos lớn)` (dense) | so sánh retriever, giữ nguyên công thức |
| Khi trả kết quả | khối lớn | khối lớn (`dual`) | đúng tinh thần small2big |

Vì gold QA hỏi **một ô**, đồ án đo thêm biến thể `dual_small`: xếp hạng và trả về chính khối nhỏ. Cặp `dual` / `dual_small` tách bạch "chất lượng truy hồi" và "chất lượng ngữ cảnh trả về".

### 2.2. Chipper (arXiv:2402.05131)

Bốn quy tắc, không tham số:

1. Gặp phần tử có tổng độ dài `< 2.048` ký tự → thử gộp với phần tử kế tiếp.
2. Lặp tới khi đạt ngưỡng; **không bao giờ cắt vỡ một phần tử**.
3. Gặp `Title` → **mở chunk mới** (title nằm trong chunk mới).
4. Gặp `Table` → **mở chunk mới và giữ nguyên toàn bộ bảng**.

Đồ án dùng đúng 4 quy tắc trên chuỗi phần tử của `unstructured`. Thêm biến thể `chipper_prefix` theo metadata *Prefix & Table Description* của paper: tiền tố là 2 câu đầu của chunk; với bảng là caption. Đây là biến thể tất định mạnh nhất trong paper (các biến thể còn lại cần GPT-4 để sinh keyword/summary).

> **Điểm cần nói rõ khi bảo vệ:** Chipper giữ nguyên bảng nên chunk có thể **vượt 2.048 ký tự** rất xa (bảng 45 dòng ≈ 4–6k ký tự). Đây là hệ quả trực tiếp của quy tắc 4, không phải lỗi cài đặt.

### 2.3. ColPali (arXiv:2407.01449)

- Render mỗi trang PDF thành ảnh (150 DPI), cắt thành các patch 512×512 (một trang A4 → 13 crop → 875 vector 128 chiều).
- Truy hồi bằng **late interaction / MaxSim**: mỗi token câu hỏi lấy max cosine với mọi patch, rồi cộng lại.
- **Không cần parser, không cần OCR, không cần chunk** — nhưng đơn vị truy hồi là **cả trang**.

Đồ án chạy **`vidore/colSmol-256M`** (surrogate 256M, CPU) vì ColPali-v1.2 (3B) cần GPU. Kết quả vì thế là **sàn** của họ late-interaction, không phải phán quyết về ColPali bản gốc.

### 2.4. Row-JSON (biến thể của nhánh text)

Mỗi dòng bảng → một object JSON:

```json
{"file": "Bản tin thay đổi thành phần chỉ số VNDIAMOND kỳ Q2/2026",
 "page": 2, "table": "Bảng 1: ...",
 "row": {"Mã CP": "MBB", "Vốn hóa tự do": "21.094", "Tỷ trọng mới": "7,18%"}}
```

Chunk tự chứa cả **ngữ cảnh** (file, trang, tên bảng) lẫn **dữ liệu** (tên cột → giá trị), nên không cần embed cả bảng. Đây là cách hiện thực hoá đề xuất "chuyển từng dòng thành JSON"; đồng thời là biến thể small-chunk của dual-granularity khi gặp bảng.

### 2.5. Bảng so sánh khái niệm

| | Dual-granularity | Chipper | ColPali | Row-JSON |
|---|---|---|---|---|
| Đơn vị truy hồi | câu / dòng | phần tử gộp 2k, cả bảng | ảnh trang | 1 dòng |
| Đơn vị trả về | đoạn lớn (small2big) | chunk 2k / cả bảng | cả trang | 1 dòng |
| Cần parser bảng | có | có | **không** | có |
| Cần model khi ingest | embed câu (nhỏ) | không | không | không |
| Cần GPU khi truy vấn | không | không | **có** (thực tế) | không |
| Chi phí mỗi câu hỏi | thấp | thấp | cao (MaxSim trên 875 patch/trang) | thấp |
| Trả lời câu hỏi "ô nào" | tốt nếu small = dòng | trung bình (chunk lớn) | **không trực tiếp** | tốt nhất |
| Ngữ cảnh cho LLM | 1.024 ký tự | 2.048+ ký tự | 1 trang ảnh | 1 dòng |
| Phụ thuộc chất lượng parser | cao | cao | thấp | cao |

---

## 3. Thiết lập thực nghiệm

### 3.1. Dữ liệu

| | |
|---|---|
| Tài liệu | 7 PDF: 3 báo cáo dự báo (broker) + 4 bản tin SSI |
| Trang | 27 |
| Bảng gold | 10 (cách gộp/tách bảng của `unstructured` khác parser cũ) |
| Câu hỏi | 130, sinh tự động từ ô bảng (mỗi câu = 1 mã CK + 1 giá trị) |
| Câu hỏi / file | 10–40 |

Gold QA sinh từ ô bảng của **parser hình học cũ** (nay lưu trữ), không phải từ `unstructured`. Đây là **thiên lệch có chủ ý**: nó cho phép giữ nguyên câu hỏi qua các parser và đo luôn phần dữ liệu bị mất khi đổi parser (mục 4.3).

### 3.2. Parser

Cả ba phương pháp text dùng **cùng một** đầu vào: `unstructured==0.27.34`, `strategy="hi_res"`, `infer_table_structure=True`, `languages=["vie"]`, `pdf_image_dpi` mặc định. Bảng lấy từ `metadata.text_as_html` của Table Transformer, parse bằng `html.parser` (stdlib) trong `etflab/uelements.py`; `colspan`/`rowspan` được trải thành lưới chữ nhật, header nhiều tầng gộp bằng `" / "`.

Không có bước nào dùng toạ độ chữ, đường kẻ hay ngưỡng hình học.

### 3.3. Gold và metric

**Chunk-level:**

- Xếp hạng trong phạm vi **một file**, BM25.
- Một chunk "đúng" nếu chứa **cả** mã CK **và** giá trị đã chuẩn hoá — không phụ thuộc `chunk_id`.
- `coverage` = tỉ lệ câu có ít nhất một chunk chứa đáp án. Đây là **trần** của recall: phần mất do parser/OCR, không phải do ranking.
- `chars@1st` = độ dài chunk đầu tiên chứa đáp án — đo **chi phí ngữ cảnh**.

**Page-level:**

- Điểm chunk được **max-pool lên trang**, xếp hạng **toàn bộ 27 trang**.
- Trang đúng lấy từ **text layer PyMuPDF** (`scripts/eval/build_gold_pages.py`): mọi trang PDF chứa cả mã CK và giá trị. Cách này độc lập parser và sửa lỗi "trang bắt đầu bảng" của gold gốc — 19/130 câu có nhiều trang đúng (bảng trải qua 2–3 trang).
- Với `dual`, điểm dùng để pool là điểm sau rerank của khối nhỏ; trang của khối nhỏ được dùng.

Metric: Recall@1/3/5/10, MRR.

**Chuẩn hoá số khi so khớp** (dễ sai, nên ghi rõ):

- `(36,2)` là **−36,2** (kế toán), không phải +36,2.
- Đơn vị trong header như `('000 cổ phiếu)` bị loại khỏi text trước khi trích số; nếu không, *mọi* chunk trong bảng đều "chứa" số 0 và các câu có đáp án 0 sẽ khớp giả tạo.
- Có test hồi quy `tests/test_metric_matching.py`.

### 3.4. Vì sao có hai mức đánh giá

ColPali chỉ có 27 ứng viên (trang), còn chunking có hàng trăm ứng viên (dòng/chunk). Đặt chung một bảng R@1 sẽ **thiên vị hệ ít ứng viên hơn**. Vì vậy:

- Bảng **chunk-level** trả lời: "trong nhánh text, cách chia nào tốt nhất?"
- Bảng **page-level** trả lời: "nếu mọi hệ đều chỉ được chọn 1 trang, hệ nào tốt nhất?"

Không có bảng nào "công bằng tuyệt đối"; hai bảng bổ sung nhau. Page-level dùng max-pool để mọi hệ chunk có cùng đơn vị so với ColPali.

---

## 4. Kết quả

### 4.1. Chunk-level (xếp hạng trong file, BM25)

| Phương pháp | #chunk | avg ký tự | coverage | R@1 | R@3 | R@5 | R@10 | MRR | chars@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`row_json`** (1 dòng = 1 JSON) | 332 | 491 | 93,8% | **81,5%** | **90,0%** | **90,0%** | **90,0%** | **0,850** | 494 |
| `dual_small` (câu + dòng) | 935 | 228 | 93,8% | 66,1% | 75,4% | 78,5% | 86,2% | 0,723 | 492 |
| `dual` (rerank + ngữ cảnh lớn) | 935 | 228 | 93,8% | 50,0% | 66,9% | 68,5% | 68,5% | 0,583 | 519 |
| `dual_dense` (e5-small) | 935 | 228 | 93,8% | 31,5% | 53,1% | 63,8% | 70,8% | 0,441 | 534 |
| `chipper_prefix` (4 quy tắc + caption) | 136 | 662 | 93,8% | 36,9% | 67,7% | 77,7% | 89,2% | 0,535 | 2430 |
| `chipper` (4 quy tắc) | 136 | 557 | 93,8% | 27,7% | 63,8% | 76,9% | 89,2% | 0,475 | 2299 |

Đọc bảng:

- **`row_json` thắng mọi chỉ số**: R@1 81,5%, MRR 0,850, chunk ~491 ký tự. Khi đã có đáp án trong chunk, nó nằm trong top-3 tới 90% (R@3 = R@10 = 90,0%).
- **Chipper tệ nhất về precision**: R@1 27,7% (bằng 1/3 `row_json`), trong khi `chars@1` gấp **4,7 lần**. R@10 89,2% gần bằng `row_json` — evidence rằng chunk lớn "chứa" đáp án nhưng không "trả lời".
- **`chipper_prefix` +9,2 điểm R@1** (27,7 → 36,9) nhưng vẫn kém `row_json` 44,6 điểm.
- **Dual càng phức tạp càng kém**: `dual_small` 66,1% → rerank bằng khối lớn còn 50,0% → dense e5-small 31,5%. Trên bài toán ô số, BM25 + dòng là đủ; thêm tầng chỉ thêm nhiễu.
- Ở bộ dữ liệu này, cả 6 hệ có **cùng coverage 93,8%** (122/130) vì cùng parser — khác biệt thuần là chunking/ranking.

### 4.2. Page-level (max-pool lên trang, xếp hạng 27 trang)

| Hệ | R@1 | R@3 | R@5 | R@10 | MRR | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| **`row_json`** | **90,0%** | **100%** | **100%** | **100%** | **0,950** | 1 dòng = 1 JSON |
| `dual_small` | 73,9% | 96,2% | 100% | 100% | 0,844 | chunk nhỏ |
| `dual` | 56,1% | 94,6% | 100% | 100% | 0,751 | rerank + ngữ cảnh lớn |
| `chipper_prefix` | 45,4% | 89,2% | 99,2% | 100% | 0,665 | 4 quy tắc + caption |
| `dual_dense` | 44,6% | 94,6% | 100% | 100% | 0,700 | e5-small dense |
| `chipper` | 36,1% | 88,5% | 99,2% | 100% | 0,618 | 4 quy tắc |
| `colSmol-256M` | 13,8% | 26,9% | 36,2% | 74,6% | 0,288 | visual, zero-shot |
| `bm25_page` (text `unstructured`) | 4,6% | 22,3% | 41,5% | 75,4% | 0,227 | baseline yếu |

Đọc bảng:

- **`row_json` bỏ xa phần còn lại**: 90,0% R@1, gần như bão hoà ở 27 trang. Đây là mốc đúng để so với ColPali.
- **Chipper tìm đúng trang trong top-3 (88,5%) nhưng hiếm khi top-1 (36,1%)**: chunk cả bảng là "máy dò trang", không phải "máy trả lời ô".
- **Mọi biến thể dual thấp hơn `row_json`**: càng thêm tầng (prose, rerank, dense) càng nhiễu trên bài toán ô số.
- **ColPali (13,8%) vượt BM25 trang (4,6%)** — tín hiệu ảnh có thật — nhưng còn cách `row_json` **6,5 lần** ở R@1. Visual là bộ chọn trang, chưa phải bộ trả lời ô.
- Page-level dùng **trang vàng từ text layer PDF**, không dùng trang bắt đầu bảng; nhờ vậy các bảng trải 2 trang không bị phạt oan. Số page-level vì thế khác bản cũ (dùng trang bắt đầu bảng).

### 4.3. Khoảng cách trích xuất: `unstructured` giữ được bao nhiêu đáp án?

Gold QA sinh từ ô bảng của parser cũ, nên trước khi so ranking phải đo xem
`unstructured` còn giữ được đáp án hay không. Đo trên **toàn bộ text element**
(không chỉ chunk) để tách lỗi parser khỏi lỗi chunking:

| Parser | Đơn vị | Giữ đáp án | Ghi chú |
|---|---|---:|---|
| `unstructured` | 809 element | **96,9%** (126/130) | trần lý thuyết của mọi cách chunking |
| `unstructured` | chunk thực tế | **93,8%** (122/130) | 4 mất hẳn + 4 tách entity/value giữa các chunk |

Gold gốc được sinh từ ô bảng của parser hình học cũ (đã lưu trữ); parser đó
**không còn** là baseline và không xuất hiện trong bảng kết quả.

4 đáp án bị mất, **tất cả đều là mất ở mức ô** (không phải mất cả dòng):

| File | Trang | Mã | Giá trị | Nguyên nhân |
|---|---:|---|---:|---|
| `..._VNDiamond_VN30_VNFIN_Q2_2026.pdf` | 2 | TCB | 8,75% | header nhiều tầng bị gộp, cột giá trị không còn |
| `..._VNDiamond_VN30_VNFIN_Q2_2026.pdf` | 2 | TPB | 3.729 | ô bị dồn/ghép cột khi Table Transformer đọc |
| `..._VNDiamond_VN30_VNFIN_Q2_2026.pdf` | 3 | MBB | 65.517 | ô rỗng (cột SSIAM VNFINLEAD) |
| `..._VNDiamond_VN30_VNFIN_Q1_2026.pdf` | 4 | MBB | 24,4 | dòng/cột lệch |

Lỗi chất lượng khác (không làm mất đáp án nhưng làm xấu key JSON):

- Header bị cắt: `Thay đổi số _=`, `lượng DCVFMVN Diamond ETF` (mất tiền tố).
- Cột STT bị OCR thành `` ` ``, `C3`, `+>`, `ƠI`.
- `unstructured` **gộp** bảng cạnh nhau (VNFINLEAD Q1) nhưng lại **tách** bảng VN30 Q2 thành 2 — hành vi không ổn định theo layout.

Điểm quan trọng: 96,9% là **trần**, không phải kết quả ranking. Mọi con số
của `unstructured` trong §4.1–4.2 là **cận dưới**; phần thiếu chủ yếu do parser,
không do cách chia chunk.

---

## 5. Ưu và nhược điểm

### 5.1. Chipper

**Ưu**

- **Không cần train, không cần model, không tham số** — đúng tinh thần "best chunk size without tuning".
- Giữ nguyên bảng nên **không bao giờ cắt đôi một dòng**; recall@10 89,2%.
- Là baseline học thuật chuẩn, dễ tái lập, dễ giải thích.
- Rẻ: vài giây cho 7 tài liệu (BM25).

**Nhược**

- **Chunk phình**: bảng 45 dòng ≈ 4–6k ký tự, gấp 2–3 lần ngưỡng 2.048; `chars@1` trung bình 2.299 (gấp 4,7 lần row-JSON).
- R@1 27,7%: một chunk chứa 45 mã CK thì tín hiệu của một mã bị pha loãng; BM25 không phân biệt được "chứa" và "trả lời".
- Recall@k cao **một phần vì chunk lớn chứa nhiều đáp án** — R@10 89,2% nhưng R@1 chỉ 27,7%; dùng làm context cho LLM thì tốn token và dễ nhiễu.
- Prefix caption chỉ cứu được một phần: R@1 36,9%.
- Không có cơ chế small2big: hoặc lấy cả bảng, hoặc không lấy gì.

### 5.2. Dual-granularity

**Ưu**

- **Tách vai trò**: khối nhỏ để tìm, khối lớn để sinh — đúng vấn đề "tìm đúng dòng nhưng cần ngữ cảnh bảng".
- Khi khối nhỏ = dòng bảng, nó **tốt cho câu hỏi ô**; khi khối nhỏ = câu, nó tốt cho câu hỏi định tính.
- Có rerank nên có thể sửa lỗi first-stage; `dual_small` là mốc so sánh trực tiếp với row-JSON.
- Không cần GPU; embed câu e5-small chạy CPU.

**Nhược**

- **Nhiều tham số hơn**: `γ`, `θ`, cách chọn đơn vị nhỏ cho bảng, cách chuẩn hoá điểm khi rerank.
- Phụ thuộc **chất lượng parser** gấp đôi: sai header → sai cả khối nhỏ (JSON key) lẫn khối lớn.
- Trong thực nghiệm này, rerank `½(nhỏ + lớn)` **làm giảm R@1 từ 66,1% xuống 50,0%** so với chỉ dùng khối nhỏ; khối lớn (cả bảng) thêm nhiễu nhiều hơn tín hiệu.
- Dense e5-small zero-shot chỉ đạt 31,5% R@1 — thấp hơn BM25 34,6 điểm; bài toán số/mã CK vẫn cần lexical.
- Chi phí ingest cao hơn: phải embed toàn bộ câu và dòng (935 chunk vs 332 của row-JSON).

### 5.3. ColPali

**Ưu**

- **Bỏ toàn bộ parser**: không cần tách bảng, không cần OCR, không sợ PDF lạ.
- Xử lý được layout phức tạp, hình ảnh, biểu đồ — thứ mà text pipeline bỏ qua.
- Là hướng "end-to-end" đang rất mạnh trong literature khi **có fine-tune**.

**Nhược**

- **Đơn vị truy hồi là trang**, không phải ô: trả lời "giá trị của MBB" vẫn cần bước thứ hai (đọc bảng trong trang).
- Zero-shot trên tiếng Việt **yếu**: ColSmol-256M 13,8% R@1 corpus; chỉ hơn BM25 trang (4,6%) nhưng kém `row_json` 6,5 lần.
- Cần GPU để chạy bản 3B; CPU 256M đã tốn ~25 s cho 130 câu (sau khi cache ảnh).
- Cần fine-tune theo miền mới mong vượt text pipeline — nằm ngoài phạm vi đồ án.

### 5.4. Row-JSON

**Ưu**

- **R@1 cao nhất trong nhóm dùng `unstructured`**: 81,5% chunk / 90,0% page, MRR 0,850/0,950; chunk nhỏ (~491 ký tự), tín hiệu tập trung.
- Tự chứa ngữ cảnh (file, trang, caption, tên cột) nên không cần embed cả bảng.
- Không cần model, không cần GPU; rẻ và dễ debug (đọc JSON là hiểu).
- Là đầu vào tốt cho LLM: một dòng = một sự kiện.

**Nhược**

- **Mất cấu trúc tổng thể**: câu hỏi kiểu "bảng này có bao nhiêu mã?" hoặc "so sánh hai cột" cần thêm chunk bảng/card.
- Phụ thuộc header do parser trả về; header sai (như `"lượng DCVFMVN Diamond ETF"` thiếu tiền tố) làm key JSON khó hiểu.
- Không giải quyết câu hỏi định tính trong phần prose — cần chunk prose song song (dual làm việc này).

### 5.5. Bài học cắt ngang: trộn nhiều loại chunk không miễn phí

So sánh trực tiếp trên cùng parser `unstructured`:

| Cấu hình index | #chunk | R@1 (chunk) | R@1 (page) |
|---|---:|---:|---:|
| `row_json` — chỉ dòng bảng | 332 | **81,5%** | **90,0%** |
| `dual` — dòng + câu prose + rerank | 935 | 50,0% | 56,1% |
| `chipper` — cả bảng + prose gộp | 136 | 27,7% | 36,1% |

Thêm prose và chunk lớn vào index làm giảm mạnh R@1. Lý do: BM25 không phân biệt chunk "chứa" và chunk "trả lời"; một đoạn prose nhắc mã CK có thể đứng trên dòng đúng. Ở mức trang, chunk nhiễu ở trang sai còn kéo cả trang sai lên qua max-pool.

Hệ quả thiết kế:

- **Row index** cho câu hỏi ô số (một dòng = một chunk).
- **Card/prose index riêng** cho câu hỏi định tính/tổng hợp.
- Nếu buộc một index: dùng **RRF có trọng số** hoặc metadata filter, không trộn thô rồi hy vọng BM25 tự chọn đúng.

---

## 6. Khuyến nghị

Kết quả **đảo ngược một giả định ban đầu** của đồ án: lợi thế lớn nhất không đến từ parser hình học, mà từ **đơn vị truy hồi "một dòng = một chunk"**.

1. **Đơn vị truy hồi thắng**: `row_json` đạt 81,5% R@1 (chunk) / 90,0% (page), gấp ~3 lần Chipper và hơn `dual_small` 15,4 điểm.
2. **Trộn chunk là bước lùi**: thêm prose/cả bảng vào cùng index làm R@1 giảm mạnh (`dual` 50,0%, `chipper` 27,7%) — xem §5.5.
3. **Chipper không phù hợp bảng số**: chunk cả bảng 2,3k ký tự, R@1 chỉ ~1/3 row-JSON; R@5 cao là "chứa" chứ chưa chắc "trả lời".
4. **Dual-granularity là chiến lược cho generation, không phải precision**: small2big hữu ích khi LLM cần ngữ cảnh, nhưng ở mức truy hồi ô, `dual_small` thua row-JSON và rerank bằng khối lớn còn làm giảm R@1.
5. **ColPali zero-shot 256M chưa thay được text pipeline**: 13,8% R@1 page, hơn BM25 trang (4,6%) nhưng cách xa `row_json` (90,0%). Cần fine-tune + GPU để đánh giá đúng.

Khuyến nghị theo mục tiêu:

| Mục tiêu | Chọn | Vì sao |
|---|---|---|
| Hỏi đáp ô số ("MBB bao nhiêu?") | **`row_json` + BM25**, metadata file/trang/caption | R@1 81,5% (chunk) / 90,0% (page), 491 ký tự/chunk |
| Hỏi đáp cần ngữ cảnh bảng | row (small) → bảng/card (large), **hai index** hoặc RRF | giữ small2big mà không pha loãng index số |
| Hỏi đáp định tính | prose index riêng | tránh chunk văn xuôi chen vào top-1 của câu hỏi số |
| Baseline học thuật | Chipper (+ prefix) | không tham số, tái lập được; chỉ dùng để so sánh |
| PDF scan / không text layer | ColPali sau fine-tune, có GPU | zero-shot 256M chưa đủ; 3B + fine-tune ngoài phạm vi |

**Thay đổi cụ thể cho code hiện tại:**

1. Dùng `row_json_chunks` làm index chính cho câu hỏi ô số.
2. Giữ `chipper_chunks` / `dual_granularity_chunks` cho câu hỏi cần ngữ cảnh, ở **index riêng**.
3. Log `coverage` của parser (96,9% ở đây) như một chỉ số chất lượng ingest.
4. Với ColPali: chỉ đầu tư nếu có GPU + dữ liệu fine-tune theo miền; nếu không, dùng nó như bộ lọc trang (R@3 26,9% — chưa đủ).

---

## 7. Hạn chế

1. **Gold do parser cũ sinh** → thiên lệch nhẹ về cách chia ô; `unstructured` chỉ khớp 96,9% ô gold (mục 4.3). Con số tuyệt đối là **cận dưới** cho `unstructured`, nhưng so sánh *giữa các cách chunking trên cùng parser* vẫn hợp lệ.
2. **Page-level dùng trang vàng từ text layer PDF**, không phải nhãn người; 19 câu có nhiều trang đúng. Đây là protocol hợp lý nhưng vẫn có thể trùng trang do giá trị xuất hiện ở nhiều bảng.
3. **ColPali-3B chưa chạy** (cần GPU). ColSmol-256M là surrogate; kết luận "visual thua text" chỉ đúng cho zero-shot 256M trên corpus này.
4. **27 trang, 7 tài liệu** — nhỏ. Chênh lệch 1–2 câu có thể đảo thứ hạng; nên đọc kèm MRR và coverage.
5. **Chưa có human-annotated gold** ở mức ô; giá trị `value_norm` do `vnnum` parse.
6. **Chưa đánh giá generation** (ROUGE/BLEU) — có chủ ý, vì mục tiêu là retrieval và chi phí.
7. **`unstructured` 0.27.34 + Table Transformer** có lỗi OCR/glyph trên corpus (cột STT thành `` ` ``, `C3`, `+>`; header bị cắt). Đây là hạn chế của parser phổ thông, không phải của chunking.

---

## 8. Tái lập

```bash
# 1) Parser: PDF -> element + bảng (venv riêng, ~65-130 s/tài liệu trên CPU)
C:/_venvs/unstr/Scripts/python.exe scripts/ingest/parse_unstructured.py --force

# 2) Các cách chunking + đo (cần torch cho dense; --no-dense chỉ cần stdlib)
C:/_venvs/colpali/Scripts/python.exe scripts/eval/eval_three_methods.py
C:/_venvs/colpali/Scripts/python.exe scripts/eval/eval_three_methods.py --no-dense

# 3) Khoảng cách trích xuất
C:/_venvs/colpali/Scripts/python.exe scripts/eval/diagnose_unstructured.py

# 4) Trang vàng + ColPali / ColSmol (page-level, CPU; cần torch + transformers)
C:/_venvs/colpali/Scripts/python.exe scripts/eval/build_gold_pages.py
C:/_venvs/colpali/Scripts/python.exe scripts/eval/colpali_retrieval.py --model vidore/colSmol-256M

# 5) Test
python -m pytest tests/test_metric_matching.py tests/test_retrieval.py
```

| File | Nội dung |
|---|---|
| `etflab/uelements.py` | adapter `unstructured` → `UElement`/`UTable`, parse HTML bảng |
| `etflab/chunking_methods.py` | `chipper_chunks`, `dual_granularity_chunks`, `row_json_chunks` |
| `scripts/ingest/parse_unstructured.py` | ingest + cache |
| `scripts/eval/eval_three_methods.py` | chunk-level + page-level |
| `scripts/eval/diagnose_unstructured.py` | coverage / mất ô |
| `scripts/eval/build_gold_pages.py` | trang vàng độc lập parser (PyMuPDF) |
| `scripts/eval/colpali_retrieval.py` | BM25 page vs ColPali/ColSmol |
| `tests/test_metric_matching.py` | hồi quy cách so khớp số (ngoặc, đơn vị) |
| `data/eval/three_methods.json` | kết quả thô chunk/page |
| `data/eval/colpali_results.json` | kết quả thô ColPali/BM25 page |
| `data/eval/gold_pages.json` | tập trang đúng cho từng câu |
| `data/eval/extraction_gap.json` | danh sách câu bị mất + lý do |
