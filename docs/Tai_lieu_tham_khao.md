# Tài liệu tham khảo cho các kỹ thuật trong `etflab`

Danh mục này **phân hai mức độ tin cậy**, để bạn không phải tin vào trí nhớ của tôi:

| Mức | Ý nghĩa |
|---|---|
| **A — Đã kiểm chứng** | Tôi trích trực tiếp từ danh mục tham khảo của 2 file PDF có sẵn trong `data/references/`. Tên, năm, tạp chí, DOI lấy nguyên văn từ file. |
| **B — Cần đối chiếu** | Từ kiến thức chung. Tác giả/năm/số trang tôi khá chắc, nhưng **chưa kiểm chứng trong repo**. Trước khi trích vào luận văn, hãy tra arXiv/ACM/ACL Anthology. |

> File này là **danh mục rộng** (kể cả các hướng chưa làm). Bản đối chiếu **từng kỹ thuật đã
> hiện thực** với nguồn gốc nằm ở [`Ky_thuat_va_Tham_khao.md`](./Ky_thuat_va_Tham_khao.md).

Ngày trích xuất: 2026-10-03.

---

## 0. Hai bài báo nền đã có trong repo

Đây là hai bài đã tải về `data/references/`, và là nguồn cho toàn bộ mục A.

**A1. Peng, Q., Luo, X., et al. — *Phân đoạn hai cấp độ và mở rộng ngữ cảnh động: Một phương pháp tối ưu hóa cho việc tạo nội dung được hỗ trợ bởi tìm kiếm* (Dual-granularity chunking and dynamic context augmentation). Hệ thống chuyên gia và các ứng dụng, 332 (2027), 133546.**
- Bản PDF trong repo là bản **tiếng Việt** (journal EFA, ở `Expert Systems with Applications`).
- Cơ sở lý thuyết trực tiếp cho phần chunk hai cấp của dự án: chunk nhỏ để truy xuất, chunk lớn để giữ ngữ cảnh.
- Đáng chú ý: bài báo này **cũng nói rõ điểm yếu của chính phương pháp nó đề xuất** — phân đoạn dựa trên đánh giá bằng LLM thì "ảo giác", không ổn định; và đánh giá bằng ROUGE/BLEU không đo được trung thực. Đây chính là lý do không nên sao chép nguyên xi prompt-based chunking mà bản này chunk theo **cấu trúc bảng** (deterministic, kiểm thử được).

**A2. Cheng, H., Wang, K., Wang, Q., Liu, T., Sheng, K. — *A Chinese financial event knowledge graph-based retrieval-augmented generation framework for financial question answering* (CFinRAG). Engineering Applications of Artificial Intelligence, 175 (2026), 114670.**
- `data/references/1-s2.0-S0952197626009528-main.pdf`
- Pipeline **textualize → knowledge graph → RL query reconstruction → dual-level retrieval**. Bảng benchmark của bài này là baseline so sánh của dự án (`NaiveRAG`, `Self-RAG`, `CRAG`, `HippoRAG`, `CFinRAG`).
- Hạn chế được chính tác giả nêu: đánh giá trên tập dữ liệu tự xây, LLM-as-judge có thiên lệch, chưa có đối chiếu chuyên gia.

> **Cảnh báo quan trọng.** Trong bài A1, trích dẫn "(Grattafiori và cộng sự, 2024)" trông như một bài về chunking, nhưng khi tra danh mục thì nó là **báo cáo kỹ thuật Llama 3** (arXiv:2407.21783), dùng làm baseline **"không có truy xuất"**. Đừng mất công truy nó như một bài về RAG.

---

## 1. Nền tảng RAG

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS 33, 9459–9474. | Bài RAG gốc. Mô hình "retrieve rồi generate". |
| **A** | Gao, Y., Xiong, Y., Gao, X., et al. (2023). *Retrieval-augmented generation for large language models: A survey.* arXiv:2312.10997. | Khảo sát toàn diện, mục 1.3 "sai lầm trong RAG". Nền cho phần đánh giá. |
| **A** | Fan, W., Ding, Y., Ning, L., et al. (2024). *A survey on RAG meeting LLMs.* KDD '24, 6491–6501. | Khảo sát khác, góc nhìn hệ thống. |
| **A** | Zhu, Y., et al. (2023). *Retrieval-augmented language models: A survey.* (trích trong A2) | Bổ sung góc nhìn lịch sử. |

## 2. Chunking và ngữ cảnh

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Peng, Q., Luo, X. (2027). *Dual-granularity chunking and dynamic context augmentation.* ESWA 332, 133546. | **Nền tảng nhất** cho `table_row` + `table_card`. Xem A1. |
| **A** | Liu, Z., Simon, C., Caspani, F. (2025). *Phân đoạn đoạn văn trong tài liệu cho hệ thống trả lời câu hỏi dựa trên trích xuất* (Document paragraph segmentation for extraction-based QA). arXiv:2501.09940. | Trực tiếp về tách đoạn trong tài liệu — cùng bài toán với T15. |
| **A** | Li, Y., Dong, B., Guerin, F., Lin, C. (2023). *Nén ngữ cảnh để nâng cao hiệu quả suy luận của các mô hình ngôn ngữ lớn* (LLMLingua). EMNLP '23, 6342–6353. | Nén prompt, giảm token — hữu ích khi context bảng lớn. |
| **A** | Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A. (2023). *Kết hợp truy xuất thông tin với suy luận theo chuỗi ý tưởng* (IRCoT). ACL '23, 10014–10037. | Suy luận nhiều bước trên văn bản đã truy xuất. |
| **B** | Sarthi, R., et al. (2024). *RAPTOR: Recursive abstractive processing for tree-organized retrieval.* ICLR 2024. | Chunk đệ quy thành cây. Cùng tinh thần đa tầng với `table_row`/`table_card`. |

## 3. Truy xuất: BM25, dense, hợp nhất

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **B** | Robertson, S., Zaragoza, H. (2009). *The probabilistic relevance framework: BM25 and beyond.* Foundations and Trends in IR, 3(4), 333–389. | **Nguồn gốc công thức BM25** mà `BM25` (`retrieval.py:133`) hiện thực. Phải tra trước khi trích. |
| **B** | Cormack, G. V., Clarke, C. L. A., Buettcher, M. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR '09. | **Nguồn gốc RRF** (`retrieval.py:281`). K=60 là hằng số chuẩn. Phải tra. |
| **A** | Karpukhin, V., et al. (2020). *Dense passage retrieval for open-domain question answering.* EMNLP 2020. | Mẫu bi-encoder (mã hoá truy vấn và đoạn văn độc lập) — cùng kiến trúc với `HybridIndex`. |
| **A** | Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* EMNLP-IJCNLP, 3982–3992. | Nền cho dense retrieval; giải mạnh của BERT tower. |
| **A** | Koroteev, M. V. (2021). *BERT: A review of applications in NLP and understanding.* arXiv:2103.11943. | Tổng quan nền tảng. |
| **B** | Wang, L., Yang, N., Huang, X., et al. (2024). *Multilingual E5 text embeddings.* arXiv:2402.05672. | **Model đang dùng**: `intfloat/multilingual-e5-small`. Xác nhận hỗ trợ tiếng Việt (100+ ngôn ngữ). Phải tra. |
| **A** | Ma, X., Gong, Y., He, P., et al. (2023). *Viết lại truy vấn cho các mô hình ngôn ngữ lớn được tăng cường bằng tìm kiếm* (Query rewriting). EMNLP '23, 5303–5315. | Phase sau: chuẩn hóa câu hỏi tài chính trước khi truy xuất. |
| **A** | Jiang, Z., Xu, F. F., Gao, L., et al. (2023). *Tạo nội dung được tăng cường bằng truy xuất chủ động* (FLARE). EMNLP '23, 7969–7982. | Truy xuất có điều kiện trong quá trình sinh. |

## 4. Trích xuất bảng và cấu trúc tài liệu

Các mục này **chưa có trong repo**, và đây là phần kỹ thuật mà `etflab` tự hiện thực — nên là
chỗ đáng đầu tư đọc trước khi viết phần "kỹ thuật trích xuất bảng" trong luận văn.

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **B** | Zhong, X., et al. (2020). *PubTabNet: Largest realistic dataset for table structure recognition.* ECCV 2020. | Bộ dữ liệu chuẩn cho nhận dạng cấu trúc bảng. |
| **B** | Poddar, S., et al. (2022). *FinTabNet: Towards improved table understanding from financial PDFs.* WACV 2022. | **Sát nhất với dự án** — bảng tài chính từ PDF. Bài này chỉ ra tài liệu tài chính vốn khó hơn tổng quát. |
| **B** | Lysak, N., et al. (2023). *TableFormer: Table structure understanding with transformers.* ICDAR. | Mô hình cấu trúc bảng, dùng trong IBM Docling. Là hướng thay thế rule-based. |
| **B** | Chen, W., et al. (2020). *TAT-QA: A hybrid approach to real-world table and text question answering.* ACL 2020. | **Bộ dữ liệu QA tài chính với bảng + văn bản**. Rất phù hợp để dựng câu hỏi vàng cho dự án. |
| **B** | Zhou, T., et al. (2021). *FinQA: A dataset of numerical reasoning over financial data.* EMNLP 2021. | QA tài chính + **chương trình tính toán** — liên quan trực tiếp phase PoT sắp tới. |

## 5. Trích xuất bảng không viền — môi trường và thư viện

Đây là phần mà project **không dùng**, nhưng nên đọc để biết vì sao tự viết:

| Mức | Tài liệu | Ghi chú |
|---|---|---|
| **B** | McKinney, V. (2017). *Camelot: Python library to extract tables from PDFs.* | Camelot có hai chế độ: `lattice` (có đường viền) và `stream` (khoảng trắng). `column_valleys()` (`geometry.py:252`) về bản chất là cùng ý tưởng với `stream`. |
| **B** | Smith, M. (2007). *Tesseract — an open source OCR engine.* | Lựa chọn OCR mặc định khi mở rộng sang PDF scan. |

## 6. PoT / suy luận số — cho phase kế tiếp

Chưa cài đặt. Đây là nền tảng nếu muốn trả lời "cổ phiếu nào tăng mạnh nhất tháng 6".

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **B** | Liu, T., et al. (2022). *Program of Thoughts prompts*. arXiv:2211.10435. | LLM sinh **chương trình** thay vì sinh đáp án từng bước. |
| **B** | Gao, L., et al. (2022). *PAL: Program-aided language models.* arXiv:2211.11435. | Sinh chương trình rồi chạy bằng interpreter. |
| **B** | Chen, W., et al. (2020). TAT-QA (xem mục 4) | Bộ dữ liệu có chứa program để đối chiếu. |

## 7. Đánh giá RAG

Chưa cài đặt. **Nên làm trước khi tối ưu thêm bất kỳ thứ gì.**

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Salemi, A., Zamani, H. (2024). *Evaluating retrieval quality in retrieval-augmented generation.* SIGIR '24, 2395–2400. | Đo **chất lượng truy xuất** tách riêng — đúng thứ `etflab` cần. |
| **A** | Es, S., James, J., Anke, L. E., Schockaert, S. (2024). *RAGAS: Automated evaluation of RAG.* EACL System Demonstrations, 150–158. | Bộ chỉ số tự động. A1 cũng nhận diện là hướng tương lai của chính nó. |
| **A** | Sudhi, V., et al. (2024). *RAG-Ex: A generic framework for explaining retrieval augmented generation.* SIGIR '24, 2776–2780. | Giải thích vì sao một đoạn được truy xuất. |
| **A** | Wang, J., et al. (2023). *Is ChatGPT a good NLG evaluator? A preliminary study.* arXiv:2303.04048. | Nền cho LLM-as-judge. |
| **A** | Zheng, L., Chiang, W.-L., et al. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.* NeurIPS 36, 46595–46603. | MT-Bench — chuẩn đánh giá dùng LLM làm giám khảo. |
| **B** | Robertson, S. et al. — chỉ số MRR, NDCG | Định nghĩa `Recall@k` và `MRR` cho phase đánh giá. |

## 8. RAG nâng cao và truy xuất theo đồ thị

Chưa cài đặt. Hữu ích nếu muốn nối bảng với tri thức tài chính khác.

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Jimenez Gutierrez, B., et al. (2024). *HippoRAG: Neurobiologically inspired long-term memory for LLMs.* NeurIPS 37, 59532–59569. | Baseline so sánh trong A2. |
| **A** | Edge, D., et al. (2024). *From local to global: A Graph RAG approach to query-focused summarization.* arXiv:2404.16130. | Microsoft GraphRAG — tổng hợp cộng đồng. |
| **A** | Peng, B., Zhu, Y., Liu, Y., et al. (2025). *Graph retrieval-augmented generation: A survey.* ACM TOIS 44(2). | Khảo sát nhánh đồ thị. |
| **A** | Guo, Z., Xia, L., Yu, Y., et al. (2024). *LightRAG: Simple and fast RAG.* arXiv:2410.05779. | RAG nhẹ, dễ học. |
| **A** | Zhang, X., Wang, M., Yang, X., et al. (2024). *Hierarchical RAG model with rethink for multi-hop QA.* arXiv:2408.11875. | Truy xuất phân cấp + suy nghĩ lại — gần ý "truy xuất dòng nhỏ, mở rộng chunk lớn". |
| **A** | Yu, Y., Ping, W., Liu, Z., et al. (2024). *RankRAG: Unifying context ranking with RAG.* NeurIPS 37, 121156–121184. | Reranking context. |
| **A** | Yan, S.-Q., Gu, J.-C., Zhu, Y., Ling, Z.-H. (2024). *Corrective retrieval augmented generation* (CRAG). arXiv:2401.15884. | Baseline; tự sửa khi truy xuất sai. |
| **A** | Asai, A., Wu, Z., Wang, Y., et al. (2023). *Self-RAG: Learning to retrieve, generate, and critique through self-reflection.* ICLR 2024. | Baseline; phản chiếu điển hình. |
| **A** | Jeong, S., et al. (2024). *Adaptive-RAG: Learning to adapt RAG through question complexity.* NAACL 2024, 7029–7043. | Chọn số lần truy xuất theo độ khó câu. |
| **A** | Ram, O., et al. (2023). *In-context retrieval-augmented language models.* TACL 11, 1316–1331. | Nền học thuyết. |

## 9. Tài chính và LLM

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Zhao, H., Liu, Z., Wu, Z., et al. (2024). *Revolutionizing finance with LLMs: An overview of applications and insights.* arXiv:2401.11641. | Bản đồ ứng dụng LLM trong tài chính. |
| **A** | Li, Y., Wang, S., Ding, H., Chen, X. (2023). *Large language models in finance: A survey.* ACM ICAIF '23, 374–382. | Khảo sát riêng miền tài chính. |
| **A** | Shah, S., Ryali, S., Venkatesh, R. (2024). *Multi-document financial question answering using LLMs.* arXiv:2411.07264. | QA tài chính đa tài liệu — rất gần bài toán của dự án. |
| **A** | Li, J., Bian, Y., Wang, G., et al. (2023). *CFGPT: Chinese financial assistant with LLM.* arXiv:2309.10654. | Trợ lý tài chính. |
| **A** | Yuan, Z., et al. (2024). *FinLLMs: A framework for financial reasoning dataset generation.* IEEE Trans. Big Data. | **Có thể dùng để sinh câu hỏi vàng** cho phase đánh giá. |
| **A** | Kertkeidkachorn, N., Nararatwong, R., Xu, Z., Ichise, R. (2023). *FinKG: A core financial knowledge graph for financial analysis.* IEEE ICSC, 90–93. | Đồ thị tri thức tài chính. |
| **A** | Cheng, H., Wang, K., Tan, X. (2024). *A link prediction method for Chinese financial event knowledge graph based on GAT and CNN.* EAAI 138, 109361. | Bài tiền thân của A2. |

## 10. Chi tiết huấn luyện LLM (tham khảo)

Bài A2 trích dẫn nhóm này để bối cảnh; không dùng trực tiếp trong `etflab`.

- **A** Wei, J., et al. (2022). *Chain-of-thought prompting elicits reasoning in LLMs.* NeurIPS 35, 24824–24837.
- **A** Achiam, J., et al. (2023). *GPT-4 technical report.* arXiv:2303.08774.
- **A** Grattafiori, A., et al. (2024). *The Llama 3 herd of models.* arXiv:2407.21783. ← xem cảnh báo mục 0.
- **A** Hu, E. J., et al. (2022). *LoRA: Low-rank adaptation of LLMs.* ICLR 2022.
- **A** Chung, H. W., et al. (2023). *PaLM: Scaling language modeling with pathways.* JMLR 24(240), 1–113.

---

## Danh sách đọc theo thứ tự

Nếu chỉ có thời gian đọc 5 mục:

1. **A1** — dual-granularity chunking. Lý thuyết nền của thiết kế chunk trong dự án.
2. **B** Robertson & Zaragoza 2009 — BM25. Vì `retrieval.py` hiện thực đúng công thức này.
3. **B** Cormack et al. 2009 — RRF. Vì `retrieval.py` dùng k=60.
4. **B** Chen et al. 2020 — TAT-QA. Để biết bộ dữ liệu QA bảng + văn bản tài chính chuẩn trông như thế nào.
5. **A** Poddar et al. 2022 — FinTabNet. Vì `tables.py` tự viết đúng việc mà bài này huấn luyện mô hình.

## Việc cần làm với danh mục này

1. Tra mục **B** trước khi trích vào luận văn — tôi không thể kiểm chứng chúng trong repo.
2. Bổ sung mục **B** vào `data/references/` để pipeline test được trên chúng.
3. Xác minh DOI của A1: năm 2027 trong PDF có vẻ là lỗi đánh số của nhà xuất bản, hãy lấy DOI từ trang nhà xuất bản.