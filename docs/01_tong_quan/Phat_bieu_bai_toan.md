# PHÁT BIỂU BÀI TOÁN XUYÊN SUỐT: HYBRID RAG CHO PHÂN TÍCH QUỸ ETF

*Dùng cho chương "Phát biểu bài toán" của luận văn. Tài liệu mô tả **toàn bộ bài toán RAG từ PDF đến câu trả lời**, rồi chỉ ra phần nào đã hiện thực và đánh giá. Số liệu thực nghiệm lấy từ [`Tong_hop_Phuong_phap_Chunking.md`](../03_thuc_nghiem/Tong_hop_Phuong_phap_Chunking.md); đề cương gốc ở [`De_cuong_Luan_van_Thac_si_RAG_Phai_sinh.md`](De_cuong_Luan_van_Thac_si_RAG_Phai_sinh.md).*

> **Quy ước trạng thái:** ✅ đã hiện thực và đo · 🟡 có code, chưa đo riêng · ⬜ định hướng, chưa làm.

---

## 1. Bài toán một câu

> **Cho** kho tài liệu tài chính tiếng Việt về quỹ ETF (PDF văn bản + bảng) và dữ liệu chuỗi thời gian (giá, NAV, dòng tiền), **cho** một câu hỏi bằng ngôn ngữ tự nhiên, **hệ thống phải trả về câu trả lời đúng về số liệu, kèm bằng chứng truy vết được và mức độ tin cậy.**

Ba từ khoá của bài toán: **đúng số liệu** (không ảo giác), **truy vết được** (tới tài liệu, trang, bảng, hàng, cột), **biết mình chắc đến đâu** (confidence).

---

## 2. Vì sao RAG chuẩn không đủ

| # | Khó khăn | Biểu hiện | Hệ quả |
|---|---|---|---|
| K1 | **Đứt gãy bảng** | cắt theo độ dài cố định làm mất liên kết nhãn hàng, tiêu đề cột và ô | truy hồi sai ô, hoặc đúng bảng nhưng sai hàng |
| K2 | **Bảng không viền, định dạng Việt** | không có đường kẻ; `2.976,9`, `(69.559)` = số âm, `0,87%`, header nhiều tầng | parser sai cấu trúc, số bị hiểu sai |
| K3 | **Thiếu ngữ cảnh kỳ/chỉ số** | cùng mã BSR có ở nhiều kỳ, nhiều chỉ số; chunk dữ liệu không mang nhãn kỳ | trả nhầm kỳ |
| K4 | **Ảo giác khi tính toán** | LLM tự cộng, trừ, nhân số trong prompt | sai số mà không biết |
| K5 | **Tách rời văn bản và chuỗi số** | điều khoản nằm trong PDF, giá/NAV nằm trong chuỗi thời gian | không trả lời được câu hỏi cần cả hai |
| K6 | **Không biết độ chắc chắn** | câu trả lời sai và đúng trông như nhau | khó dùng để ra quyết định |

---

## 3. Phát biểu hình thức

### 3.1. Đối tượng

- **Kho văn bản** $\mathcal{D}$: tập PDF; mỗi tài liệu $d$ gồm trang, đoạn văn $p$ và bảng $T$. Bảng $T$ có hàng $r$ (nhãn thực thể $e$, ví dụ mã chứng khoán), cột $j$ (nhãn $h_j$, có thể nhiều tầng) và ô $c_{r,j}$.
- **Kho chuỗi thời gian** $\mathcal{S}$: giá OHLCV, NAV/CCQ, dòng tiền ngoại theo ngày cho từng mã ETF và cổ phiếu thành phần.
- **Câu hỏi** $q$: tiếng Việt, thuộc một trong các nhóm tác vụ ở mục 4.
- **Ngữ cảnh** $\pi = (\text{kỳ}, \text{chỉ số})$: suy ra từ $q$ hoặc từ metadata tài liệu.

### 3.2. Hàm cần học

Hệ thống là hàm

$$\mathcal{F}(q;\ \mathcal{D}, \mathcal{S}) \;\longmapsto\; (a,\ \mathcal{E},\ \kappa)$$

- $a$: câu trả lời (số kèm đơn vị, hoặc đoạn văn ngắn);
- $\mathcal{E}$: tập bằng chứng, mỗi phần tử là bộ *(file, trang, bảng, hàng, cột)* hoặc *(mã, khoảng thời gian)* cho chuỗi số;
- $\kappa \in [0,1]$: độ tin cậy.

Tiêu chí đúng: $a$ khớp đáp án vàng $a^*$ (số: sau chuẩn hoá `vnnum`, có dung sai làm tròn), $\mathcal{E}$ chứa bằng chứng vàng, và $\kappa$ hiệu chuẩn (độ tin cậy cao khi đúng, thấp khi sai).

---

## 4. Các nhóm tác vụ

| Nhóm | Ví dụ câu hỏi | Dữ liệu cần | Cách giải |
|---|---|---|---|
| **T1. Tra cứu ô bảng** | "Vốn hoá tự do của MBB kỳ Q2/2026, chỉ số VNDiamond?" | bảng trong PDF | truy hồi hàng ✅ |
| **T2. Tra cứu điều khoản** | "Chu kỳ xem xét lại cơ cấu danh mục? Phí quản lý quỹ?" | văn bản điều lệ, cáo bạch | truy hồi đoạn 🟡 |
| **T3. Chênh lệch giá/NAV** | "Premium/Discount của E1VFVN30 phiên gần nhất?" | chuỗi giá + NAV | tool-calling ⬜ |
| **T4. Mô phỏng cơ cấu rổ** | "Quỹ phải mua/bán bao nhiêu cổ phiếu BSR ở kỳ này?" | bảng tỷ trọng mới + cũ, NAV, giá | truy hồi + Program-of-Thought ⬜ |

T1–T2 chỉ cần **truy hồi**. T3–T4 cần **truy hồi rồi tính toán**: số lấy từ bằng chứng, phép tính do mã Python thực thi chứ không do LLM tự tính.

---

## 5. Pipeline xuyên suốt: input và output từng tầng

```
 PDF ──► [L1 Parse] ──► [L2 Chunk + Metadata] ──► [L3 Truy hồi] ──┐
                                                                    ├─► [L5 Sinh đáp án] ──► (a, E, κ)
 Chuỗi giá/NAV ──────► [L4 Công cụ / PoT] ─────────────────────────┘
                                 ▲
 Câu hỏi q ──► [L0 Router] ──────┘
```

### L0. Router *(⬜)*

| | |
|---|---|
| **Input** | câu hỏi $q$ |
| **Output** | nhãn tác vụ ∈ {T1, T2, T3, T4}; bộ lọc `DocFilter(kỳ, chỉ số)` |
| Hiện có | `infer_doc_filter` (`src/retrieval.py`) suy ra kỳ và chỉ số từ câu hỏi ✅; phân loại tác vụ chưa làm ⬜ |

### L1. Parse — PDF → element và bảng ✅

| | |
|---|---|
| **Input** | PDF có text layer, bảng không viền |
| **Output** | danh sách element (title, narrative, list, table…) + bảng chuẩn hoá thành lưới chữ nhật (`colspan`/`rowspan` đã trải), cache JSON |
| Công cụ | `unstructured` `hi_res` (YOLOX + Table Transformer); `scripts/ingest/parse_unstructured.py` |
| Chất lượng | giữ 126/130 giá trị gold (96,9%) |

### L2. Chunk và metadata ✅

| | |
|---|---|
| **Input** | element + bảng (L1) |
| **Output** | `list[Chunk]`; mỗi `Chunk` có `text`, `sparse_text`, `source_file`, `page`, `table_id`, `row_index`, `cells`, `parent_id`, `doc` (loại, chỉ số, kỳ, năm) |
| Chiến lược | `row_json` (1 hàng = 1 JSON), `chipper`, `chipper_prefix`, `dual`, `dual_small`, `dual_dense` |
| Giải quyết | K1 (giữ liên kết hàng–cột), K3 (metadata kỳ/chỉ số đẩy vào `sparse_text`) |

### L3. Truy hồi ✅

| | |
|---|---|
| **Input** | $q$, `list[Chunk]`, `DocFilter` (tuỳ chọn), `top_k` |
| **Output** | `list[Hit]` top-k theo điểm giảm dần |
| Cách làm | lọc cứng theo kỳ/chỉ số → BM25 (tokenizer chuẩn hoá số) → dense e5-small (tuỳ chọn) → hợp nhất RRF |
| Đã đo | Recall@k và MRR, mức chunk và mức trang (mục 6) |

### L4. Công cụ tính toán / Program-of-Thought *(⬜)*

| | |
|---|---|
| **Input** | hàng bằng chứng từ L3, chuỗi giá/NAV từ $\mathcal{S}$, nhãn tác vụ |
| **Output** | giá trị số do mã Python thực thi trong sandbox, kèm đơn vị |
| Ví dụ | Premium/Discount $= (\text{giá thị trường} - \text{NAV})/\text{NAV}$; lượng cổ phiếu cần mua/bán $= (w_{\text{mới}} - w_{\text{cũ}}) \times \text{NAV quỹ} / \text{giá}$ |
| Hiện có | `vnnum.parse_number` chuẩn hoá số Việt Nam làm nền ✅ |
| Giải quyết | K4, K5 |

### L5. Sinh đáp án và độ tin cậy *(⬜)*

| | |
|---|---|
| **Input** | $q$, bằng chứng (L3), kết quả tính toán (L4) |
| **Output** | $(a, \mathcal{E}, \kappa)$ |
| Hướng làm | LLM chỉ diễn đạt từ bằng chứng; $\kappa$ từ độ lệch điểm truy hồi, độ nhất quán giữa các bằng chứng, hoặc suy luận Bayes |
| Giải quyết | K6 |

---

## 6. Phần đã đánh giá: L1–L3 trên tác vụ T1

Đây là phần thực nghiệm hiện có của luận văn.

**Câu hỏi nghiên cứu.** Với bảng số không viền trong báo cáo tài chính tiếng Việt, **đơn vị truy hồi nào** trả lời đúng câu hỏi "giá trị của mã X là bao nhiêu" với chi phí thấp nhất?

**Input/output của thí nghiệm.**

| | |
|---|---|
| Input | 7 PDF, 27 trang, 10 bảng (thông báo SSI, báo cáo broker, điều lệ) + 130 câu hỏi vàng |
| Output | xếp hạng chunk/trang cho từng câu; Recall@{1,3,5,10}, MRR, coverage |
| Cố định | parser (`unstructured`), retriever (BM25, dense tuỳ chọn, RRF) |
| Thay đổi | **đơn vị phân mảnh** (L2) |

Một mẫu gold (`data/eval/gold_qa.jsonl`):

```json
{ "query": "Vốn hóa tự do của MBB kỳ Q2 2026 chỉ số VNDIAMOND",
  "source_file": "Thay_doi_thanh_phan_chi_so_VNDiamond_VN30_VNFIN_Q2_2026.pdf",
  "page": 2, "row_index": 4, "col_index": 2,
  "entity": "MBB", "col_label": "Vốn hóa tự do",
  "value_raw": "21.094", "value_norm": 21094.0,
  "period": "Q2 2026", "index": "VNDIAMOND" }
```

**Metric.** *Chunk-level*: xếp hạng chunk trong file; đúng nếu chunk chứa giá trị vàng. *Page-level*: max-pool điểm chunk lên trang, xếp hạng 27 trang. *Coverage*: tỷ lệ câu mà ít nhất một chunk chứa đáp án (trần của recall, phản ánh parser).

**Kết quả.**

| Phương pháp | Đơn vị truy hồi | R@1 chunk | R@1 trang | MRR trang |
|---|---|---:|---:|---:|
| **`row_json`** | một hàng bảng | **81,5%** | **90,0%** | **0,950** |
| `dual_small` | câu hoặc hàng | 66,1% | 73,9% | 0,844 |
| `chipper` | cả bảng + prose, ≤ 2.048 ký tự | 27,7% | 36,1% | 0,618 |
| `bm25_page` | văn bản cả trang | — | 4,6% | 0,227 |

**Kết luận.** Đơn vị truy hồi quyết định kết quả hơn thuật toán chunking; trộn nhiều loại chunk vào một index làm R@1 giảm mạnh; dual-granularity phục vụ ngữ cảnh cho sinh đáp án chứ không phục vụ độ chính xác truy hồi.

---

## 7. Bảng tổng hợp phạm vi

| Tầng | Vai trò trong bài toán xuyên suốt | Trạng thái | Bằng chứng |
|---|---|---|---|
| L0 Router | chọn tác vụ và bộ lọc | một phần (lọc kỳ/chỉ số) | `infer_doc_filter`, `tests/test_retrieval.py` |
| L1 Parse | PDF → bảng | ✅ | `scripts/ingest/parse_unstructured.py`, coverage 96,9% |
| L2 Chunk | đơn vị truy hồi | ✅ | `src/chunking_methods.py`, 6 chiến lược |
| L3 Truy hồi | tìm bằng chứng | ✅ | `src/retrieval.py`, mục 6 |
| L4 Công cụ / PoT | tính toán chính xác | ⬜ | — |
| L5 Sinh đáp án + $\kappa$ | trả lời có độ tin cậy | ⬜ | — |

---

## 8. Đóng góp

1. **Phát biểu bài toán RAG tài chính ETF Việt Nam** dưới dạng pipeline L0–L5, tách rõ phần truy hồi (cần đúng bằng chứng) khỏi phần tính toán (cần đúng số).
2. **Bằng chứng thực nghiệm cho L2**: một hàng = một chunk đạt R@1 81,5%, gấp khoảng 3 lần Chipper; trộn chunk làm giảm precision; dual-granularity hợp cho generation.
3. **Định lượng giới hạn của L1**: `unstructured` giữ 96,9% giá trị, 4 giá trị mất được liệt kê.
4. **Nền cho L4–L5**: chuẩn hoá số Việt Nam (`vnnum`), metadata kỳ/chỉ số, chunk tự chứa có truy vết.

---

## 9. Giới hạn

1. **Mới đánh giá L1–L3 trên T1.** T2 có chunk nhưng chưa có gold; T3, T4 và L4–L5 chưa có code lẫn kết quả. Không được nêu như kết quả đã đo.
2. **Gold QA do parser hình học cũ sinh** (`archive/geometric_parser/`), hơi thiên lệch. So sánh giữa các cách chunking trên cùng parser vẫn hợp lệ; số tuyệt đối của `unstructured` là cận dưới.
3. **Corpus nhỏ**: 7 tài liệu, 27 trang, 10 bảng. Chênh lệch 1–2 câu có thể đảo thứ hạng.
4. **Chưa có gold gán nhãn thủ công** ở mức ô.
5. **Chưa có dữ liệu chuỗi thời gian** (OHLCV, NAV, dòng tiền ngoại) trong repo, dù đề cương yêu cầu.
6. **Phụ thuộc parser**: Table Transformer sai hàng/cột thì JSON sai theo (header nhiều tầng bị gộp, ô rỗng, cột STT bị OCR sai).
7. **Phạm vi PDF**: chỉ PDF có text layer; PDF scan, ô merge phức tạp và câu hỏi tổng hợp nhiều bảng nằm ngoài phạm vi.
