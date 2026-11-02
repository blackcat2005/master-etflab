# CƠ CHẾ PAPER CHIPPER: ELEMENT-TYPE CHUNKING CHO RAG BÁO CÁO TÀI CHÍNH

> **Cập nhật 2026-10:** parser hình học từng được nêu là "lấp khoảng trống" của
> Chipper đã được lưu trữ; thực nghiệm hiện tại chạy Chipper trên element của
> `unstructured`, và so sánh với row-JSON / dual-granularity trên cùng đầu vào.

*Dựa trên công trình: "Financial Report Chunking for Effective Retrieval Augmented Generation" — Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebastian Laverde, Renyu Li (Unstructured Technologies), arXiv:2402.05131 (v1 05/02/2024, v3 16/03/2024).*

> **Phân biệt với các file khác trong `docs/`:**
> - File này trả lời **"paper Chipper làm gì, cơ chế ra sao"** — dùng khi viết phần "phương pháp" và để **chọn baseline đối chiếu** cho tầng chunking.
> - [`So_sanh_Chipper_va_cac_phuong_phap_Chunking_2025_2026.md`](./So_sanh_Chipper_va_cac_phuong_phap_Chunking_2025_2026.md) — các công trình **sau** Chipper, bằng chứng "mạnh hơn/yếu hơn", và đề xuất ablation.
> - [`Tong_quan_va_Phuong_phap_Dual_Granularity_RAG.md`](./Tong_quan_va_Phuong_phap_Dual_Granularity_RAG.md) — paper Peng et al. (chunking hai cấp theo LLM).
> - [`Tong_hop_Phuong_phap_Chunking.md`](./Tong_hop_Phuong_phap_Chunking.md) — kết quả thực nghiệm trên corpus này.
> - [`Tham_khao_va_nguon.md`](./Tham_khao_va_nguon.md) — đối chiếu kỹ thuật ↔ nguồn học thuật.

---

## 0. Thông tin paper

| Trường | Giá trị |
|---|---|
| **Tên** | Financial Report Chunking for Effective Retrieval Augmented Generation |
| **Tác giả** | Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebastian Laverde, Renyu Li |
| **Đơn vị** | Unstructured Technologies (Sacramento, CA, USA) — tác giả của thư viện `unstructured` |
| **Nguồn** | arXiv:2402.05131, DOI `10.48550/arXiv.2402.05131` |
| **Giấy phép** | CC BY-NC-ND 4.0 |
| **Trạng thái** | Preprint (không thấy venue hội nghị/tạp chí chính thức trong bản arXiv) → **mức B** |
| **Điểm cốt lõi** | Chunk theo **loại phần tử cấu trúc** (Title / NarrativeText / Table / ListItem…) thay vì theo số token cố định |
| **Model nhận diện phần tử** | **Chipper** — vision encoder-decoder lấy cảm hứng từ Donut, xuất JSON theo trang |
| **Dataset đánh giá** | FinanceBench (141 câu hỏi / 80 tài liệu 10-K, 10-Q, 8-K) |
| **Kết quả chính** | Q&A manual **53,19%** — SOTA trên FinanceBench tại thời điểm công bố |

> ⚠️ **Lưu ý về tên:** "Chipper" trong paper là **tên model nhận diện bố cục tài liệu** (doc understanding model) của Unstructured, **không phải tên của phương pháp chunking**. Phương pháp được gọi là **element-based chunking** (chunk theo phần tử). Trong doc này dùng "Chipper" như cách gọi tắt của cả pipeline.

> 🔍 **Ba tầng cần phân biệt — Chipper chỉ làm 2 tầng đầu:**
>
> | Tầng | Chipper | Bằng chứng (§3.4, tr. 7) |
> |---|---|---|
> | Layout analysis — nhận diện phần tử | ✅ | JSON phần tử theo trang, mỗi phần tử có `type` + `bounding box` + `text` |
> | Table detection — khoanh vùng bảng | ✅ | `Table` là 1 trong 11 loại phần tử; chỉ có bbox, không có cấu trúc bên trong |
> | **Table structure recognition** — hàng/cột/ô | ❌ | *"if a table element is found, a new chunk is started, **preserving the entire table**"* |
>
> Hệ quả: bảng trong Chipper là **hộp đen nguyên khối**. Không thể tạo row-JSON, không có metric cấu trúc (TEDS/GriTS), và câu hỏi tra một ô cụ thể vẫn khó. Đây chính là khoảng trống mà **row-JSON trên bảng do `unstructured` parse** lấp vào — **không phải tái tạo Chipper, mà là năng lực Chipper không có**.
>
> *Phân biệt:* thư viện `unstructured` có module table extraction riêng (`hi_res` + Table Transformer), nhưng **paper không dùng và không đánh giá** module đó. "Chipper + TSR" là một hệ khác, không phải hệ được công bố trong arXiv:2402.05131.

---

## 1. Hệ thống thuật ngữ

1. **Element-based chunking (chunk theo phần tử):** Phân mảnh tài liệu dựa trên **đơn vị cấu trúc gốc** của văn bản (tiêu đề, đoạn văn, bảng, danh sách…) thay vì số token.
2. **Chipper:** Model vision encoder-decoder (Donut-style) đọc ảnh trang PDF và xuất **JSON mô tả danh sách phần tử** — mỗi phần tử có `type` (loại), `bounding box` (hộp bao), `text` (nội dung).
3. **Composite chunk (khối hợp thành):** Chunk được tạo bằng cách **gộp nhiều phần tử** lại cho đủ dài, chứ không cắt giữa một phần tử.
4. **Element type (loại phần tử):** 11 loại Unstructured dùng: `NarrativeText`, `Title`, `ListItem`, `Table`, `Header`, `Footer`, `FigureCaption`, `Formula`, `Address`, `Image`, `UncategorizedText`.
5. **Aggregation (hợp nhất nhiều cấu hình):** Chạy nhiều chiến lược chunking song song rồi gộp **kết quả truy xuất** để tăng recall.
6. **Naive representation (biểu diễn ngây thơ):** Metadata lấy **2 câu đầu** của khối làm tiền tố; với bảng thì lấy **caption**.

---

## 2. Vấn đề paper giải quyết

RAG truyền thống chia tài liệu theo **độ dài cố định** (fixed-size) hoặc **theo đoạn văn** (paragraph-level). Paper chỉ ra ba hạn chế:

| Vấn đề | Hệ quả |
|---|---|
| Coi mọi văn bản **bình đẳng** | Bỏ qua thông tin nằm trong **cấu trúc tài liệu** (tiêu đề, bảng) |
| Không quan tâm **bảng biểu** | Bảng bị cắt ngang, mất liên kết hàng–cột |
| Phải **tinh chỉnh chunk size** thủ công | Kích thước tối ưu thay đổi theo từng loại dữ liệu → không tổng quát |

Điểm khác biệt then chốt: paper không tìm cách *cải tiến fixed-size*, mà **đổi đơn vị phân mảnh** — từ "số token" sang "phần tử cấu trúc". Hệ quả là khối có **kích thước tự thích ứng** mà *không cần tuning*.

---

## 3. Kiến trúc RAG tổng thể

Paper giữ **nguyên hệ thống RAG** và chỉ thay đổi duy nhất bước **chunking**:

```
Tài liệu (PDF SEC 10-K/10-Q/8-K)
        │
        ▼
[1] CHUNKING  ← điểm duy nhất paper thay đổi
        │
        ▼
[2] INDEXING  — encoder → vector DB (Weaviate)
        │
        ▼
[3] RETRIEVAL — top-10 chunk theo cosine similarity
        │
        ▼
[4] GENERATION — GPT-4 sinh câu trả lời từ top-10
```

| Thành phần | Lựa chọn của paper |
|---|---|
| **Vector DB** | Weaviate (ANN / HNSW) |
| **Encoder** | `sentence-transformers/multi-qa-mpnet-base-dot-v1` (train trên >256M cặp QA) |
| **Top-k** | `k = 10` |
| **Generator** | GPT-4 |
| **Prompt sinh** | Nếu không trả lời được → trả về `No answer` |

Vì mọi thành phần khác **giữ cố định**, chênh lệch kết quả quy được trực tiếp về chiến lược chunking. Đây là **thiết kế thực nghiệm sạch** (single-variable).

---

## 4. CƠ CHẾ CHUNKING — trái tim của paper

### 4.1 Baseline

- **Fixed-size:** cắt theo `n` token, `n ∈ {128, 256, 512}`.
- **Aggregation:** chạy song song nhiều cấu hình rồi gộp kết quả truy xuất.

### 4.2 Element-based (quy trình 4 bước)

Tài liệu được Chipper phân tích → JSON danh sách phần tử (mỗi phần tử có `type`, `bbox`, `text`). Từ danh sách phần tử, khối được sinh theo thuật toán:

```
1. Nếu độ dài text của phần tử < 2.048 ký tự
      → thử GỘP với phần tử kế tiếp
2. Lặp lại bước 1 cho tới khi đạt độ dài mong muốn
      → NHƯNG KHÔNG BAO GIỜ cắt vỡ một phần tử
3. Nếu gặp phần tử TITLE
      → BẮT ĐẦU chunk mới
4. Nếu gặp phần tử TABLE
      → BẮT ĐẦU chunk mới và GIỮ NGUYÊN TOÀN BỘ BẢNG
```

**Bốn quy tắc này là toàn bộ "thuật toán" — không có tham số cần tinh chỉnh.** Đây chính là luận điểm "best chunk size *without tuning*" trong abstract.

> **Điểm mấu chốt cần nhớ khi viết luận văn:**
> - `2.048 ký tự` là ngưỡng **độ dài tối thiểu để coi là chunk**, không phải độ dài cố định.
> - **Title** và **Table** là hai loại phần tử được đối xử đặc biệt: chúng **buộc mở chunk mới** → ranh giới chunk trùng ranh giới ngữ nghĩa.
> - Bảng được giữ **nguyên vẹn** → khắc phục "đứt gãy bảng danh mục".

### 4.3 Ba loại metadata làm giàu khối

Sau khi có chunk, paper sinh thêm metadata để hỗ trợ truy xuất:

| # | Loại metadata | Cách sinh | Ghi chú |
|---|---|---|---|
| 1 | **Keywords** | GPT-4, prompt template → tối đa **6 từ khóa** đại diện | Giúp khớp thuật ngữ |
| 2 | **Summary** | GPT-4, prompt template → đoạn tóm tắt | Ngữ cảnh cô đọng |
| 3 | **Prefix & Table Description** | **2 câu đầu** của khối (tiền tố); với bảng → **mô tả/caption của bảng** | Không cần LLM, tất định |

Ba loại metadata này tương ứng **ba biến thể thực nghiệm** trong bảng kết quả (`Keywords Chipper`, `Summary Chipper`, `Prefix & Table Description Chipper`).

> **Nhận xét thiết kế:** loại #3 (Prefix & Table Description) là loại **rẻ nhất và tất định nhất** — chỉ lấy 2 câu đầu + caption bảng. Nó cũng cho **BLEU cao nhất (0.400)** trong ba biến thể đơn lẻ. Đây là gợi ý trực tiếp cho dự án: **caption bảng là tín hiệu truy xuất mạnh.**

---

## 5. Dữ liệu và thiết lập đánh giá

| Trường | Giá trị |
|---|---|
| **Dataset** | **FinanceBench** (Islam et al., 2023) — open-book QA trên báo cáo SEC |
| **Quy mô gốc** | 150 câu hỏi / 84 báo cáo |
| **Quy mô dùng được** | **141 câu hỏi / 80 tài liệu** (mất 4 báo cáo không tải được) |
| **Số trang** | tổng **11.787** trang; trung bình **147,34** trang/tài liệu (min 4, max 549) |
| **Token trung bình** | **102.444** token/tài liệu (std ≈ 61.979) |
| **Trang chứa bằng chứng** | trung bình trang **54,58** (std 43,66; trải từ trang 1 đến trang 304) |
| **Loại câu hỏi** | có câu hỏi **đòi suy luận tính toán** (computational reasoning) |

Phân bố phần tử Chipper dự đoán trên corpus:

| Loại phần tử | Số lượng |
|---|---|
| NarrativeText | 61.780 |
| ListItem | 33.054 |
| Title | 29.664 |
| UncategorizedText | 9.400 |
| **Table** | **7.700** |
| Header | 3.959 |
| Footer | 1.026 |
| Address | 229 |
| FigureCaption | 54 |
| Formula | 29 |
| Image | 26 |
| **Tổng** | **146.921** |

---

## 6. Kết quả

### 6.1 Hiệu quả phân mảnh (số lượng chunk)

| Chiến lược | Tổng chunk | Chunk/tài liệu (std) |
|---|---|---|
| Base 128 | 64.058 | 800,73 (484,11) |
| Base 256 | 32.051 | 400,64 (242,04) |
| Base 512 | 16.046 | 200,58 (121,01) |
| **Chipper** | **20.843** | **260,57 (145,80)** — bảng: 96,20 (57,53) |

Kích thước chunk của Chipper gần với **Base 512** → tổng số chunk tương đương, nhưng **ranh giới chunk có ý nghĩa ngữ nghĩa**.

### 6.2 Độ chính xác truy xuất

| Chiến lược | Tổng chunk | Page Acc | ROUGE | BLEU |
|---|---|---|---|---|
| Base 128 | 64.058 | 72,34 | 0,383 | 0,181 |
| Base 256 | 32.051 | 73,05 | 0,433 | 0,231 |
| Base 512 | 16.046 | 68,09 | 0,455 | 0,250 |
| Base Aggregation | 112.155 | 83,69 | 0,536 | 0,277 |
| Keywords Chipper | 20.843 | 46,10 | 0,444 | 0,315 |
| Summary Chipper | 20.843 | 62,41 | 0,473 | 0,350 |
| Prefix & Table Desc. Chipper | 20.843 | 67,38 | 0,514 | 0,400 |
| **Chipper Aggregation** | **62.529** | **84,40** | **0,568** | **0,452** |

**Đọc bảng này:** fixed-size thắng ở **page-level** (72–73%) nhưng thua ở **paragraph-level** (ROUGE/BLEU) → tức là "trúng trang nhưng lệch đoạn". Chipper **nhất quán** giữa hai mức. `Chipper Aggregation` đạt đỉnh cả hai (84,40 / 0,568 / 0,452) **nhưng chỉ cần 62.529 chunk — bằng ~một nửa Base Aggregation (112.155)**.

### 6.3 Độ chính xác Q&A

| Chiến lược | No answer (%) | GPT-4 (%) | **Manual (%)** |
|---|---|---|---|
| Base 128 | 35,46 | 29,08 | 35,46 |
| Base 256 | 25,53 | 32,62 | 36,88 |
| Base 512 | 24,82 | 41,84 | 48,23 |
| **Keywords Chipper** | **22,70** | 43,97 | **53,19** |
| Summary Chipper | **17,73** | 43,97 | 51,77 |
| Prefix & Table Desc. Chipper | 20,57 | 41,13 | 53,19 |

- **Tỉ lệ `No answer` thấp nhất (17,73%)** thuộc về Summary Chipper → chunk theo phần tử **giữ được ngữ cảnh đủ để trả lời**.
- **Q&A manual cao nhất (53,19%)** ở Keywords và Prefix Chipper — paper tuyên bố **SOTA trên FinanceBench** (so sánh với ~50% của baseline).
- **Chú ý phương pháp luận:** có **chênh lệch hệ thống giữa GPT-4 (tự động) và người chấm**, đặc biệt khi câu trả lời dài kèm giải thích tính toán. Ví dụ paper nêu: đáp án đúng `39,7%` nhưng model trả `39,73%` kèm lời giải → GPT-4 judge đánh sai.

---

## 7. Phát hiện & thảo luận

1. **Element-based cải thiện rõ rệt** Q&A tài chính và **nhất quán** giữa page-level và paragraph-level, trong khi fixed-size không.
2. **Không cần tinh chỉnh chunk size** → tổng quát hóa tốt hơn sang loại tài liệu mới.
3. **Base 512 gần nhất** với element-based (do độ dài tương đương), nhưng **mất mạch ngữ cảnh** trong một số trường hợp → ROUGE/BLEU thấp hơn.
4. **Aggregation cải thiện truy xuất nhưng hại Q&A**: khi gộp nhiều cấu hình, prompt **vượt giới hạn token của GPT-4** → không hiệu quả cho sinh câu trả lời.
5. **Prompt engineering có ảnh hưởng thật:** đổi động từ `referencing` → `using` làm **giảm chất lượng** câu trả lời; đổi thứ tự context/question thì **không khác biệt có ý nghĩa**.
6. **Chi phí giảm:** đạt điểm truy xuất cao nhất với **một nửa số vector** → giảm chi phí index và độ trễ truy vấn.

---

## 8. Điểm yếu & phản biện (dùng khi viết phần "hạn chế")

| Điểm yếu | Chi tiết |
|---|---|
| **Phụ thuộc model nhận diện phần tử** | Toàn bộ chất lượng chunk phụ thuộc **Chipper** (vision encoder-decoder). Chưa có → không chạy được |
| **Miền Anh/PDF có text layer** | Corpus là SEC 10-K/10-Q tiếng Anh; không kiểm chứng trên bảng **không viền tiếng Việt** |
| **Ngưỡng 2.048 ký tự** | Vẫn là một ngưỡng "cứng" — paper mô tả là "desired length" nhưng không phân tích độ nhạy của ngưỡng này |
| **Không đánh giá cấu trúc bảng** | Giữ nguyên cả bảng, nhưng **không đo TEDS** hay chất lượng khôi phục cấu trúc — nên không biết bảng có bị méo không |
| **Không có venue chính thức** | Bản arXiv; chưa rõ phản biện kín → độ tin cậy **mức B** |
| **Q&A accuracy ~53%** | Vẫn còn một nửa sai; câu hỏi suy luận tính toán là điểm yếu (đây là lý do cần lớp tool-calling/PoT) |
| **Chênh lệch GPT-4 vs manual** | Đo tự động chưa đáng tin với câu trả lời kèm tính toán → phải chấm tay |

---

## 9. Đối chiếu với row-JSON và paper Peng

### 9.1 Ba chiến lược chunking cạnh tranh

| | **Chipper (paper này)** | **Peng et al. (dual-granularity)** | **row-JSON (hiện tại)** |
|---|---|---|---|
| Đơn vị phân mảnh | Phần tử bố cục (element type) | Đoạn văn (LLM gộp) | Một dòng bảng |
| Sinh chunk lớn | Gộp phần tử theo quy tắc | LLM few-shot phân loại gộp | (không có 2 cấp; ngữ cảnh nằm trong JSON) |
| Sinh chunk nhỏ | (không có 2 cấp) | Embedding similarity ghép câu | Mỗi dòng = 1 JSON tự chứa |
| Cần model? | **Có** (Chipper) | **Có** (LLM) | **Không** — chỉ cần parser + rule |
| Cần tuning? | Ngưỡng 2.048 ký tự | γ=1024, θ=0.9 | Không |
| Bảng | Giữ **nguyên cả bảng** | Cắt theo câu | Header lặp lại trong **từng dòng** |
| Metadata | Keywords / Summary / Prefix | — | file + page + caption + tên cột |

### 9.2 Chìa khóa khác biệt

Cả ba đều đi đến kết luận **"chunk theo cấu trúc tốt hơn chunk theo token"**, nhưng khác ở **nguồn của cấu trúc**:

- **Chipper:** cấu trúc *nhận diện bằng model bố cục* (học từ ảnh trang).
- **Peng:** cấu trúc *suy ra bằng LLM* (gộp đoạn theo chủ đề).
- **row-JSON:** cấu trúc bảng *khôi phục từ HTML của `unstructured`* (Table Transformer), tất định sau bước parse.

→ Đây là điểm để **định vị đóng góp**: row-JSON là biến thể **tất định, kiểm thử được** của cùng nguyên lý "structure-aware chunking", áp cho **bảng không viền tiếng Việt** — thứ cả Chipper lẫn Peng đều chưa chạm tới.

### 9.3 Cái nên học từ Chipper cho pipeline hiện tại

| Ý tưởng Chipper | Áp vào pipeline | Trạng thái |
|---|---|---|
| Title buộc mở chunk mới | `chipper_chunks` mở chunk tại `Title` | ✅ đã cài |
| Table giữ nguyên cả bảng | `chipper_chunks` giữ nguyên `Table` | ✅ đã cài |
| **Caption bảng = metadata truy xuất** | row-JSON lặp caption trong mọi dòng; `chipper_prefix` thêm caption | ✅ đã cài |
| **Prefix 2 câu đầu** | `chipper_prefix` | ✅ đã đo |
| Element type (title/text/table/list) | `uelements.py` map `Title`/`NarrativeText`/`Table`/`ListItem` | ✅ đã có |
| Đo **consistency page-level vs paragraph-level** | `eval_three_methods.py` báo cả chunk-level và page-level | ✅ đã có |

### 9.4 Cái KHÔNG nên bê về

- **Chipper model:** phụ thuộc checkpoint ảnh + huấn luyện trên tiếng Anh; rule-based + row-JSON của đồ án đạt R@1 cao hơn trên corpus này (mục 4.1, `Tong_hop_Phuong_phap_Chunking.md`).
- **Aggregation:** paper tự thừa nhận **hại Q&A** do tràn ngữ cảnh → không dùng.
- **Keywords/Summary bằng GPT-4:** tốn chi phí và không tất định; Prefix & Table Description đã đủ hiệu quả (BLEU cao nhất).

---

## 10. Downstream: dùng Chipper như thế nào trong luận văn

1. **Làm baseline đối chiếu tầng chunking.** Paper cung cấp số liệu sẵn trên FinanceBench → so sánh trực tiếp định lượng với row-JSON là hợp lý nếu chạy trên bộ QA riêng.
2. **Bộ ba so sánh đủ mạnh:** **Chipper** (element-type) → **Peng** (LLM dual-granularity) → **row-JSON** (structural, borderless-VN). Mỗi bước là một mức "cấu trúc hóa".
3. **Đo lường:** theo đúng paper, đo **đồng thời** page-level accuracy *và* paragraph-level (ROUGE/BLEU **chỉ để đối chiếu**, không làm chỉ số chính — xem cảnh báo `Tham_khao_va_nguon.md` mục 13).
4. **Đóng góp gốc:** chứng minh **row-JSON tất định ≥ element-type/CV-based** trên bảng **không viền tiếng Việt** — điều Chipper không làm được.
5. **Lớp phía sau:** câu hỏi tính toán vẫn sai ở cả ba (~53%) → khẳng định luận điểm cần **PoT/tool-calling** cho Premium/Discount và mô phỏng cơ cấu rổ.

---

## 11. Trích dẫn

**APA:**
Jimeno Yepes, A., You, Y., Milczek, J., Laverde, S., & Li, R. (2024). *Financial Report Chunking for Effective Retrieval Augmented Generation.* arXiv:2402.05131.

**BibTeX:**
```bibtex
@misc{jimenoyepes2024chipper,
  title  = {Financial Report Chunking for Effective Retrieval Augmented Generation},
  author = {Jimeno Yepes, Antonio and You, Yao and Milczek, Jan and
            Laverde, Sebastian and Li, Renyu},
  year   = {2024},
  eprint = {2402.05131},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi    = {10.48550/arXiv.2402.05131}
}
```

**Tài liệu liên quan:**
- Islam, P., et al. (2023). *FinanceBench: A New Benchmark for Financial Question Answering.* arXiv:2311.11944.
- Kim, G., et al. (2021). *Donut: Document understanding transformer without OCR.* arXiv:2111.15664. — kiến trúc nền của Chipper.
- Peng, Q., Luo, X., et al. (2027). *Dual-granularity chunking and dynamic context augmentation.* Expert Systems with Applications, 332, 133546.

---

*Mức tin cậy nguồn: **B** — đọc trực tiếp bản HTML arXiv (v2), có bảng số liệu; chưa tra được venue hội nghị/tạp chí chính thức. Tra lại trước khi trích chính thức.*
