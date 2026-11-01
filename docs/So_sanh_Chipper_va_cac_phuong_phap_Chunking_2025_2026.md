# SO SÁNH CHIPPER VỚI CÁC CÔNG TRÌNH CHUNKING MỚI HƠN (2024–2026)

*Câu hỏi: "Có paper nào mạnh hơn Chipper không?" — khảo sát ngày 26/10/2026.*

> **Phân biệt với các file khác trong `docs/`:**
> - [`Co_che_Chipper_Financial_Report_Chunking.md`](./Co_che_Chipper_Financial_Report_Chunking.md) — cơ chế paper Chipper (baseline C của luận văn).
> - **File này** — khảo sát các công trình *sau* Chipper, đối chiếu bằng chứng, và đề xuất strategy bổ sung cho pipeline hiện tại (`unstructured` + row-JSON).
> - [`Tong_quan_va_Phuong_phap_Dual_Granularity_RAG.md`](./Tong_quan_va_Phuong_phap_Dual_Granularity_RAG.md) — hướng chunking hai cấp của Peng et al.
> - [`Tong_hop_3_phuong_phap_Chunking_va_ColPali.md`](./Tong_hop_3_phuong_phap_Chunking_va_ColPali.md) — kết quả thực nghiệm trên corpus này.

---

## 0. Kết luận nhanh

**Có, nhưng không có paper nào "đè" Chipper một cách trực diện.** Tính đến 10/2026, Chipper mới có **17 trích dẫn** (OpenAlex) và phần lớn là survey hoặc ứng dụng RAG, không có nghiên cứu nào chạy lại đúng protocol của Chipper trên corpus mới rồi báo "thắng". Vì vậy "mạnh hơn" ở đây là **suy luận từ bằng chứng gián tiếp**, không phải kết luận head-to-head.

Ba ứng viên đáng chú ý nhất, theo ba nghĩa khác nhau:

| Hạng | Công trình | Vì sao đáng chú ý | Ràng buộc của luận văn |
|---|---|---|---|
| **1** | **STC** — *Structure-Aware Chunking for Tabular Data* (arXiv 2605.00318, 05/2026) | Gần bài toán nhất (bảng biểu), **chỉ dùng BM25**, Recall@1 **0,366 → 0,754** | ✅ Chạy được, không GPU/API |
| **2** | **Paragraph Group Chunking (PGC)** — *A Systematic Investigation of Document Chunking Strategies* (arXiv 2603.06976, 03/2026) | Thực nghiệm lớn nhất: **36 strategy × 6 domain × 5 embedder** | ⚠️ Cần dense retrieval |
| **3** | **Document-level metadata** — *Long-Context Isn't All You Need* (Snowflake, 03/2025) | Mức tăng lớn nhất với chi phí **gần bằng 0**: QA **50–60% → 72–75%** | ✅ Metadata đã có sẵn trong pipeline |

**Điểm quan trọng nhất về mặt học thuật:** các công trình 2025–2026 cho thấy lợi ích của Chipper **không đến từ taxonomy 11 loại phần tử**, mà từ hai yếu tố đơn giản hơn:

1. **Không cắt ngang qua bảng/biên cấu trúc** (boundary alignment).
2. **Mang ngữ cảnh của tài liệu/tiêu đề vào chunk** (context prefix).

Đây chính là khe hở để luận văn đóng góp: **phân rã (ablate) lợi ích của Chipper** thay vì chỉ so sánh tổng.

---

## 1. Phạm vi khảo sát

- 17 công trình trích dẫn Chipper (OpenAlex, `cites:W4391709255`).
- Tìm kiếm có hệ thống trên arXiv + ACM DL + ACL Anthology cho: tabular chunking, financial report chunking, structure-aware retrieval, Vietnamese financial QA (2024–10/2026).
- Ưu tiên công trình **có số liệu retrieval** (Recall@k / MRR / nDCG / P@1) và **có thể tái lập trên CPU**.

Không khảo sát: các blog tổng hợp benchmark (Firecrawl, Prem AI) trừ khi dẫn lại nguồn gốc; các phương pháp chỉ báo QA accuracy do LLM chấm mà không công bố retrieval.

---

## 2. Bảng tổng hợp bằng chứng

> ⚠️ **Các con số dưới đây đến từ corpus khác nhau.** Không được cộng/so sánh trực tiếp với kết quả nội bộ ở dòng cuối. Cột "Corpus" ghi rõ để tránh so sánh khập khiễng.

| # | Công trình | Năm / Venue | Ý tưởng cốt lõi | Corpus | Metric & kết quả | Chi phí |
|---|---|---|---|---|---|---|
| 1 | **STC** (Guttal et al.) | 2026, arXiv WIP (5 trang) | Row Tree: mỗi hàng = block key–value; split theo token budget đúng biên cấu trúc; greedy merge không overlap | MAUD (CSV/Excel) | BM25-only **R@1 0,366 → 0,754**; hybrid **MRR 0,3576 → 0,5945**; giảm 40–56% số chunk | CPU, O(n) |
| 2 | **PGC** (Shaukat et al.) | 2026, arXiv | Gộp các đoạn văn liền kề theo nhóm cấu trúc thay vì cắt theo token | 6 domain, 5 embedder | **nDCG@5 0,459; P@1 24%; H@5 59%** (cao nhất); fixed-size: nDCG@5 < 0,244; P@1 2–3% | CPU + embedder |
| 3 | **Snowflake** (Tarakad et al.) | 2025, blog kỹ thuật | Gắn metadata cấp tài liệu (ticker, ngày, loại form) vào **mọi** chunk; ưu tiên chunk ~1.800 ký tự; retrieve top-50 | 23k SEC 10-K/10-Q | QA **50–60% → 72–75%**; chunk-level context **−5,8 / −0,4 điểm**; chunk 14.400 ký tự **−10…−20%**; markdown-aware **+5…10 pp** | Gần 0 nếu metadata có sẵn |
| 4 | **ICSE-SEIP '26** (El Bachyr et al.) | 2026, ICSE-SEIP + arXiv 2604.12047 | Khảo sát parser × chunker × overlap cho PDF tài chính; công bố benchmark **TableQuest** | FinanceBench + TableQuest | pdfplumber **MRR 0,807** (table) / 0,634 (text); 512 token + 25% overlap; **parser & retriever quyết định hơn chunker** | CPU |
| 5 | **HDRR** (Cheng et al.) | 2026, arXiv v3 | Định tuyến query → tài liệu (LLM structured output) rồi mới retrieve chunk trong phạm vi tài liệu đó | FinDER (1.500 query) | điểm TB **6,02 → 7,54**; failure **22,5% → 6,4%**; correctness **+18,7 pp**; token 5–15K (SFR: 50–200K) | Cần LLM |
| 6 | **TableRAG** (Yu et al.) | 2025, **EMNLP** | Không flatten bảng: query decomposition → text retrieval → **sinh & chạy SQL** → hợp thành đáp án | HeteQA + public | SOTA heterogeneous doc QA | Cần LLM + DB |
| 7 | **Late Chunking** (Günther et al.) | 2024/2025, arXiv | Embed **toàn văn** trước, cắt chunk **sau** transformer, trước mean-pooling | BeIR | Tốt hơn fixed-size chunk embeddings; không cần train | Cần long-context embedder |
| 8 | **Upgrade the Retriever First** (Mangla) | 2026, arXiv v4 (pre-registered) | Đo đúng giá trị của "1 LLM call/chunk" so với **cấu trúc miễn phí + retriever tốt hơn** | Qasper, FreshStack | 7B enrichment ≈ **section-path prefix miễn phí** (+5,5 Qasper); CR tốn **5,8–8,7× token** mà tương đương; **BM25: enrichment +8,7**; late chunking **giảm** recall Qasper | Cần LLM để so sánh |
| 9 | **Mix-of-Overlap** (Ben Ameur et al.) | 2025, **IEEE BigData** | Multi-overlap chunking, định vị trực tiếp so với chunking kiểu Chipper | — | Chưa truy cập được full text → **chưa xác minh số** | CPU |
| 10 | **Pipeline hiện tại** (`unstructured`) | — | row-JSON / Chipper / dual | 130 câu, 7 báo cáo VN | **row-JSON R@1 81,5% / MRR 0,850**; Chipper 27,7% / 0,475; `dual_small` 66,1% / 0,723 | CPU, BM25 |

**Đọc bảng này đúng cách:** dòng 1–9 là bằng chứng *bên ngoài*; dòng 10 là *của bạn*. Chênh lệch R@1 giữa Chipper (27,7%) và row-JSON (81,5%) **lớn hơn** mọi mức cải thiện mà các paper ngoài kia báo cáo — nhưng một phần do gold set auto-generate từ parser cũ (xem §7).

---

## 3. Phân tích theo nhóm

### 3.1. Nhóm thay thế trực tiếp: STC và Mix-of-Overlap

**STC là công trình gần bài toán nhất.** Nó không đề xuất "loại phần tử" mới, mà đề xuất **đơn vị hàng (row) là nguyên tử**, rồi:

1. Mỗi hàng → một block `key: value` (giữ liên kết cột–giá trị).
2. Xây **Row Tree** phân cấp (bảng → nhóm hàng → hàng → ô).
3. Split theo **token budget nhưng tôn trọng biên cấu trúc**.
4. **Greedy merge không overlap** để chunk dày, không lặp.

Điểm đáng chú ý: **trên BM25-only, R@1 tăng 38,8 điểm phần trăm**. Đây là mức tăng cùng bậc với row-JSON của đồ án (81,5% R@1, không cần model). Với bảng không viền tiếng Việt — nơi cấu trúc ô đã được `unstructured` dựng sẵn — đây là bước tiến tự nhiên: row-JSON hiện tại gần với "row tree" nhưng **thiếu liên kết cha–con tường minh và thiếu greedy packing theo budget**.

Hạn chế phải nói rõ khi trích dẫn:

- Dataset là **CSV/Excel**, không phải PDF; không có OCR noise, không có bảng nhiều tầng header như báo cáo tài chính Việt Nam.
- **Work in progress, 5 trang, chưa peer-review** → mức tin cậy B/C.
- Không báo Recall@5/10, không báo MRR cho BM25-only (chỉ R@1).

**Mix-of-Overlap** là paper duy nhất tôi tìm thấy *tự định vị* là kế thừa Chipper (IEEE BigData 2025). Nhưng bản full text bị chặn (ResearchGate 403, IEEE không trả nội dung), nên **chưa xác minh được con số**. Chỉ nên nhắc như "hướng tiếp cận song song", đừng trích số.

### 3.2. Nhóm "đổi ngữ cảnh, không đổi biên": PGC và document metadata

Đây là nhóm **rẻ nhất và có mức tăng ổn định nhất**.

**PGC** thắng trong thực nghiệm 36 strategy: gộp **nhóm đoạn văn** theo cấu trúc, thay vì cắt theo token. Với tài liệu tài chính, "nhóm đoạn văn" tương đương "một mục + bảng + diễn giải của mục đó". Điều này gợi ý strategy mà pipeline hiện tại chưa có: **section chunk** = tiêu đề mục + toàn bộ bảng + đoạn văn diễn giải trong mục.

**Snowflake** đưa ra phát hiện phản trực giác và rất hữu ích:

| Cách thêm ngữ cảnh | Tác động |
|---|---|
| Metadata cấp **tài liệu** (ticker, ngày, form, kỳ) | **+12…15 pp** |
| Context cấp **chunk** do LLM sinh (kiểu Anthropic Contextual Retrieval) | **−5,8 pp** (Llama) / **−0,4 pp** (Claude) |
| Chunk quá lớn (14.400 ký tự) | **−10…−20%** |
| Markdown/header-aware (khi không có metadata) | **+5…10 pp** |

Với pipeline hiện tại, metadata cấp tài liệu **đã có sẵn trong tên file** (`broker_forecast_reports/<ticker>_<period>...`, `ssi_review_announcements/...`). Nghĩa là có thể thêm tiền tố `"<ticker> — <loại báo cáo> — <kỳ>"` vào mọi chunk **mà không cần LLM, không cần GPU**. Đây là thí nghiệm có tỉ lệ đóng góp/chi phí cao nhất.

### 3.3. Nhóm mạnh hơn nhưng vi phạm ràng buộc

| Công trình | Vì sao mạnh | Vì sao không dùng được ngay |
|---|---|---|
| **TableRAG** (EMNLP 2025) | Không flatten bảng; SQL trên bảng giữ được truy vấn đa bước | Cần LLM sinh SQL + execute; corpus phải được chuẩn hoá thành bảng có schema |
| **HDRR** (2026) | +18,7 pp correctness, failure 6,4% | Cần LLM định tuyến query → tài liệu |
| **Contextual Retrieval** (Anthropic 2024) | Giảm 35–49% retrieval failure | 1 LLM call / chunk; Snowflake cho thấy **không đáng** so với metadata tài liệu |
| **Late Chunking** (2024) | Chunk giữ ngữ cảnh toàn văn | Cần long-context embedder; "Upgrade the Retriever First" báo **giảm** recall trên Qasper |
| **RAPTOR / GraphRAG / StructRAG** | Truy vấn đa bước, tổng hợp toàn cục | Cần LLM tóm tắt/xây đồ thị |

Các công trình này nên xuất hiện trong **Related Work** như "upper bound" và hướng phát triển, không phải baseline.

### 3.4. Nhóm phản chứng: Chipper không phải lúc nào cũng thắng

Đây là phần quan trọng nhất để luận văn **không bị bắt bẻ**.

**ICSE-SEIP '26** — khảo sát đúng câu hỏi của Chipper (parser × chunker × overlap trên PDF tài chính) và kết luận:

> *"retriever and parser choices strongly affect performance, with lightweight combinations often matching more complex ones"* — lựa chọn parser và retriever ảnh hưởng mạnh hơn lựa chọn chunker; tổ hợp nhẹ thường **ngang** tổ hợp phức tạp.

Họ so 6 chunker (token, sentence, semantic, recursive, SDPM, neural) ở 512 token + 25% overlap. **Element-based không phải người thắng.** Đây là bằng chứng mạnh nhất rằng "chunk theo phần tử" không phải chân lý phổ quát.

**Upgrade the Retriever First** (pre-registered, v4 10/2026) — kết luận thậm chí sắc hơn:

> *"Upgrade the retriever first, chunk at headers, and enrich mainly for lexical retrieval."*

Cụ thể: enrichment bằng LLM **tương đương** một tiền tố section-path miễn phí; dưới BM25 thì enrichment được **+8,7 recall** (đáng kể cho pipeline lexical hiện tại); late chunking **làm giảm** recall.

**"Is Semantic Chunking Worth the Computational Cost?"** (Qu et al., 2024) — semantic chunking không vượt fixed-size một cách hệ thống trên văn bản đồng nhất. Củng cố luận điểm: **thắng lợi nằm ở cấu trúc, không ở "thông minh" của bộ cắt**.

---

## 4. Vậy Chipper "mạnh" ở đâu, và yếu ở đâu?

Tổng hợp lại, bức tranh nhất quán qua 5 nguồn độc lập:

| Thành phần của Chipper | Có bằng chứng ủng hộ? | Ghi chú |
|---|---|---|
| **Cắt theo biên phần tử** (không cắt giữa bảng) | ✅ Mạnh | Nhất quán với PGC, Snowflake (markdown-aware), ICSE (recursive tốt) |
| **11 loại phần tử** (Title/NarrativeText/Table/…) | ⚠️ Yếu | Không paper nào cô lập được lợi ích của taxonomy; ICSE cho thấy chunker đơn giản ngang ngửa |
| **Composite chunk + aggregation** | ⚠️ Trung bình | Aggregation tăng recall nhưng tăng chi phí index; STC cho thấy merge thông minh tốt hơn |
| **Ngữ cảnh (caption, 2 câu đầu)** | ✅ Mạnh | Chính là "context prefix"; Snowflake làm mạnh hơn ở cấp tài liệu |
| **Không cần tuning chunk size** | ✅ | Vẫn đúng; nhưng STC cũng đạt được điều này bằng budget + biên cấu trúc |

**Kết luận tái định khung:** Chipper đúng, nhưng đúng vì **biên + ngữ cảnh**, không vì **nhãn loại**. Đây là luận điểm có thể bảo vệ trong luận văn và lý giải tại sao row-JSON (không dùng model Chipper) vẫn thắng Chipper trong thí nghiệm của bạn.

---

## 5. Benchmark & dataset tiếng Việt (dùng để tăng tính ngoại lai)

Điểm yếu lớn nhất của eval hiện tại là **gold set tự sinh từ ô bảng của parser hình học cũ** (đã lưu trữ; xem §7). Có ba nguồn có thể bổ sung:

| Dataset | Nội dung | Quy mô | Phù hợp thế nào |
|---|---|---|---|
| **ViFinQA** (HF: `HuuDong03uet/ViFinQA`) | Câu hỏi số học trên BCTC tiếng Việt, corpus-level | **1.012 câu / 1.973 báo cáo / 100 mã / 2015–2025** (bản đầy đủ: 143.815 bảng chuẩn hoá) | Dùng làm **tập kiểm tra ngoại lai**: chỉ cần trả lời "chunk đúng có nằm trong top-k không" |
| **ViNumQA** (VLSP 2025, ACL Anthology 2025.vlsp-1.25) | Numerical reasoning | **2.069 câu Việt + 2.005 FinQA-Vi** | Đánh giá khả năng giữ số khi chunk |
| **Open-ViTabQA** (2025) | Table QA tiếng Việt | Wikipedia table | Không cùng domain, nhưng kiểm tra khả năng đọc bảng tiếng Việt |

**Khuyến nghị:** giữ gold set 130 câu làm *dev set* nội bộ; lấy **100–200 câu ViFinQA** (lọc theo mã có trong corpus của bạn, hoặc tải báo cáo tương ứng) làm *test set ngoại lai*. Chỉ cần retrieval, không cần generation → vẫn nhẹ.

Lưu ý: ViFinQA bản public **không có đáp án/evidence vàng**, chỉ có câu hỏi + text báo cáo. Nên dùng để đo **Recall@k theo tài liệu** (retriever có tìm đúng báo cáo/mã/kỳ không) thay vì cell-level.

---

## 6. Đề xuất cho luận văn

### 6.1. Ba strategy bổ sung (thay vì đi tìm "paper mạnh hơn")

Thay vì thêm một baseline nữa, hãy **phân rã** cơ chế của Chipper. Đây là đóng góp mới, không trùng paper nào.

| Mã | Strategy | Công thức | Câu hỏi nó trả lời |
|---|---|---|---|
| **E** | **STC-style row tree** | Hàng = block `key: value`; split theo budget đúng biên; greedy merge; giữ `parent_table_id` | Row-tree có tốt hơn row-JSON không? |
| **F** | **C + doc preamble** | Y hệt C, chỉ thêm tiền tố `<ticker> — <report_type> — <period> — <section>` | Lợi ích của Chipper đến từ **ngữ cảnh** hay từ **biên**? |
| **G** | **Page-level** | Mỗi trang = 1 chunk (atom), không cắt | Biên "trang" so với biên "phần tử" thì sao? |

Và một biến thể kiểm tra tính bền:

| **H** | **D + doc preamble** | row-JSON + tiền tố metadata | Metadata có giúp cả baseline mạnh nhất không? |

### 6.2. Ma trận ablation (phần "đóng góp khoa học")

```
                 biên cấu trúc
                 fixed ── page ── element ── row-tree
ngữ cảnh  không   A/B     G        C          D/E
          có      —       G+pre    F          H
```

- **C vs F** → tác động *thuần* của context prefix (giữ nguyên biên).
- **C vs G** → tác động *thuần* của loại biên (element vs page).
- **D vs E** → row-tree/greedy packing có hơn row-JSON.
- **D vs H** → metadata có cộng thêm trên baseline mạnh nhất.

Nếu F > C đáng kể, luận văn có thể phát biểu: *"phần lớn lợi ích của element-based chunking trên báo cáo tài chính Việt Nam đến từ ngữ cảnh tài liệu, không từ taxonomy phần tử"* — một kết luận có giá trị và **trái với cách đọc thông thường paper Chipper**.

### 6.3. Metric bổ sung (rẻ, tăng độ tin)

Ngoài Recall@k/MRR hiện có:

- **Table-level recall**: chunk top-k có trỏ đúng `table_id` vàng không.
- **Cell coverage**: tỉ lệ ô vàng xuất hiện trong top-k (đo mức "nguyên vẹn bảng").
- **Doc-level recall**: đúng mã + đúng kỳ (quan trọng với báo cáo tài chính, nơi cùng một chỉ tiêu lặp qua nhiều năm).
- **Token efficiency**: số chunk × độ dài trung bình (STC báo giảm 40–56% chunk — đây là lợi thế vận hành).

---

## 7. Caveat phải giữ trong mọi bài viết

1. **Không có head-to-head thật.** Chưa ai chạy lại protocol Chipper trên corpus mới. Mọi so sánh là gián tiếp.
2. **Gold set tự sinh** từ ô bảng của parser hình học cũ → R@1 81,5% của row-JSON bị **lạc quan hoá**; `unstructured` chỉ giữ 96,9% ô gold (93,8% ở mức chunk). Cần ViFinQA hoặc câu hỏi viết tay để kiểm chứng.
3. **Khác corpus.** STC (CSV/Excel), PGC (6 domain tiếng Anh), Snowflake (SEC 10-K), ICSE (FinanceBench). Không cái nào là bảng không viền tiếng Việt.
4. **Khác retriever.** STC dùng BM25 (tốt, so sánh được), PGC/Late Chunking dùng dense (không so sánh được trực tiếp).
5. **Chất lượng nguồn khác nhau.** EMNLP (TableRAG) > ICSE-SEIP (El Bachyr) > arXiv WIP (STC 5 trang) ≈ blog kỹ thuật (Snowflake) > self-published (Mangla, dù có pre-registration).
6. **Chưa đo generation.** Toàn bộ là retrieval. Không được suy ra "trả lời tốt hơn".

---

## 8. Tài liệu tham khảo (BibTeX)

```bibtex
@article{guttal2026stc,
  title   = {Structure-Aware Chunking for Tabular Data in Retrieval-Augmented Generation},
  author  = {Guttal, Pooja and Magotra, Varun and Mahavishnu, Vasudeva and
             Chanto, Natasha and Sivaprasad, Sidharth and Gaur, Manas},
  journal = {arXiv preprint arXiv:2605.00318},
  year    = {2026},
  note    = {Work in progress, 5 pages}
}

@article{shaukat2026chunking,
  title   = {A Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity},
  author  = {Shaukat, Muhammad Arslan and Adnan, Muntasir and Kuhn, Carlos C. N.},
  journal = {arXiv preprint arXiv:2603.06976},
  year    = {2026}
}

@article{elbachyr2026pdf,
  title   = {Empirical Evaluation of PDF Parsing and Chunking for Financial
             Question Answering with RAG},
  author  = {El Bachyr, Omar and Song, Yewei and Ezzini, Saad and Klein, Jacques and
             Bissyand{\'e}, Tegawend{\'e} F. and Zilali, Anas and Ble, Ulrick and
             Goujon, Anne},
  journal = {arXiv preprint arXiv:2604.12047},
  year    = {2026},
  note    = {ICSE-SEIP 2026; DOI 10.1145/3786583.3786911}
}

@article{cheng2026hdrr,
  title   = {Sustainable Hybrid Document-Routed Retrieval for Financial RAG:
             Resolving the Robustness-Precision Trade-off},
  author  = {Cheng, Zhiyuan and Lai, Longying and Liu, Yue},
  journal = {arXiv preprint arXiv:2603.26815},
  year    = {2026}
}

@article{yu2025tablerag,
  title   = {TableRAG: A Retrieval Augmented Generation Framework for
             Heterogeneous Document Reasoning},
  author  = {Yu, Xiaohan and Jian, Pu and Chen, Chong},
  journal = {Proceedings of EMNLP},
  year    = {2025},
  note    = {arXiv:2506.10380}
}

@article{mangla2026upgrade,
  title   = {Upgrade the Retriever First: What an {LLM} Call per Chunk Buys in {RAG}},
  author  = {Mangla, Bhavik},
  journal = {arXiv preprint arXiv:2603.23533},
  year    = {2026},
  note    = {v4, pre-registered}
}

@misc{tarakad2025snowflake,
  title        = {Long-Context Isn't All You Need: Impact of Retrieval and Chunking on Finance RAG},
  author       = {Tarakad, Nitya and Yu, Puxuan and Samdani, Rajhans and Merrick, Luke},
  howpublished = {Snowflake Engineering Blog},
  year         = {2025},
  month        = mar,
  url          = {https://www.snowflake.com/en/blog/engineering/impact-retrieval-chunking-finance-rag/}
}

@inproceedings{benameur2025moo,
  title     = {Mix-Of-Overlap: Enhancing Retrieval-Augmented Generation with
               Multi-Overlap Chunking},
  author    = {Ben Ameur, Mahdi and Fathallah, Mohamed Ali and Abdellatif, Takoua},
  booktitle = {IEEE International Conference on Big Data (BigData)},
  year      = {2025},
  doi       = {10.1109/BigData66926.2025.11402402}
}

@article{gunther2024late,
  title   = {Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models},
  author  = {G{\"u}nther, Michael and Mohr, Isabelle and Williams, Daniel James and
             Wang, Bo and Xiao, Han},
  journal = {arXiv preprint arXiv:2409.04701},
  year    = {2024}
}

@article{qu2024semantic,
  title   = {Is Semantic Chunking Worth the Computational Cost?},
  author  = {Qu, Renyi and Tu, Ruixuan and Bao, Forrest},
  journal = {arXiv preprint arXiv:2410.13070},
  year    = {2024}
}

@inproceedings{toan2025vlsp,
  title     = {{VLSP} 2025 Challenge: Numerical Reasoning Question and Answer},
  author    = {Toan, Le Ngoc and Linh, Ha My and Duc, Pham Thi and Quyen, Ngo The and
               Huyen, Nguyen Thi Minh},
  booktitle = {Proceedings of the 11th International Workshop on Vietnamese Language
               and Speech Processing (VLSP)},
  pages     = {187--196},
  year      = {2025},
  address   = {Hanoi, Vietnam}
}

@misc{vifinqa2026,
  title        = {{ViFinQA}: A Comprehensive and Challenging Benchmark for
                  End-to-End Vietnamese Financial Reasoning},
  author       = {{DSKT-NOWJ}},
  howpublished = {Hugging Face Datasets},
  year         = {2026},
  url          = {https://huggingface.co/datasets/HuuDong03uet/ViFinQA}
}
```

---

## 9. Nguồn đã kiểm tra

| Nguồn | URL | Trạng thái |
|---|---|---|
| STC abstract + metadata | https://arxiv.org/abs/2605.00318 | ✅ đọc trực tiếp |
| Systematic investigation | https://arxiv.org/abs/2603.06976 | ✅ đọc trực tiếp |
| ICSE-SEIP '26 (ACM DL) | https://dl.acm.org/doi/10.1145/3786583.3786911 | ✅ abstract + §5 + §8 |
| HDRR | https://arxiv.org/abs/2603.26815 | ✅ đọc trực tiếp |
| Upgrade the Retriever First | https://arxiv.org/abs/2603.23533 | ✅ đọc trực tiếp |
| Snowflake blog | https://www.snowflake.com/en/blog/engineering/impact-retrieval-chunking-finance-rag/ | ✅ đọc toàn văn |
| TableRAG (EMNLP 2025) | https://arxiv.org/abs/2506.10380 | ✅ abstract |
| Late Chunking | https://arxiv.org/abs/2409.04701 | ✅ abstract |
| ViFinQA dataset card | https://huggingface.co/datasets/HuuDong03uet/ViFinQA | ✅ đọc toàn văn |
| ViNumQA / VLSP 2025 | https://aclanthology.org/2025.vlsp-1.25/ | ✅ metadata |
| Chipper citations | OpenAlex `W4391709255` (17 citations) | ✅ |
| Mix-of-Overlap | IEEE DOI 10.1109/BigData66926.2025.11402402 | ❌ **403/không truy cập được full text** |
