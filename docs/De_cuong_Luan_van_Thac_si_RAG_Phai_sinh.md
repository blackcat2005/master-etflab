# **HỆ THỐNG HYBRID RAG CHO PHÂN TÍCH QUỸ ETF**

---

1. ## PHẠM VI NGHIÊN CỨU CHI TIẾT (RESEARCH SCOPE)

### 1.1 Phạm vi Quỹ ETF và Chỉ số tham chiếu

* **Đối tượng nghiên cứu chính:** Tập trung vào **Top 3 quỹ ETF chỉ số có quy mô tài sản và thanh khoản lớn nhất** trên Sở Giao dịch Chứng khoán TP.HCM (HOSE):  
  1. **FUEVFVND** (Quỹ ETF DCVFMVN DIAMOND \- mô phỏng chỉ số VNDiamond).  
  2. **E1VFVN30** (Quỹ ETF DCVFMVN30 \- mô phỏng chỉ số VN30).  
  3. **FUESSVFL** (Quỹ ETF SSIAM VNFIN LEAD \- mô phỏng nhóm cổ phiếu tài chính đầu ngành).  
* **Chỉ số tham chiếu liên quan:** VN30, VNDiamond, VNFinLead và rổ cổ phiếu cơ sở thành phần.

### 1.2 Phạm vi Nghiệp vụ & Kịch bản câu hỏi (Task Scope)

Hệ thống tập trung giải quyết 3 nhóm tác vụ cốt lõi:

1. **Tra cứu điều lệ và tiêu chí kỹ thuật (Textual Terms Lookup):** Trích xuất quy định về chu kỳ xem xét lại cơ cấu danh mục định kỳ (tháng 1, 4, 7, 10), giới hạn tỷ trọng cổ phiếu đơn lẻ, phí quản lý quỹ từ văn bản pháp lý.  
2. **Định giá & Phân tích chênh lệch thời gian thực (Real-time Metric Tracking):** Tính toán độ lệch Premium/Discount giữa giá thị trường và NAV, hỗ trợ phát hiện các cơ hội kinh doanh chênh lệch giá.  
3. **Phân tích kịch bản kỳ cơ cấu rổ (Rebalancing Impact Simulator):** Trích xuất thông báo dự kiến cơ cấu rổ mới (thay đổi tỷ trọng, cổ phiếu thêm mới/loại ra), từ đó tự động tính toán khối lượng và giá trị giải ngân/bán ra ước tính của quỹ.

---

## 2\. NGUỒN DỮ LIỆU THU THẬP CHI TIẾT (DATA SOURCES)

### 2.1. Dữ liệu văn bản phi cấu trúc (Unstructured Text Data \- Nạp vào RAG)

* **Bản cáo bạch & Điều lệ quỹ:** Tải từ website chính thức của các công ty quản lý quỹ (Dragon Capital, SSIAM) hoặc cổng thông tin HOSE. Số lượng: 3 bản điều lệ và 3 bản cáo bạch của 3 quỹ mục tiêu.  
* **Thông báo cơ cấu định kỳ rổ chỉ số (Review Announcements):** Văn bản PDF do HOSE công bố định kỳ về việc thay đổi tỷ trọng, hệ số free-float, giới hạn vốn hóa của các cổ phiếu trong rổ VN30, VNDiamond. Số lượng: 8 – 12 thông báo (2–3 năm gần nhất).  
* **Báo cáo phân tích kỳ cơ cấu ETF:** Báo cáo tổng hợp dự báo của các CTCK lớn (SSI Research, VNDirect Research, HSC) công bố trước mỗi kỳ review danh mục. Số lượng: 6 – 8 báo cáo phân tích.

### 2.2. Dữ liệu chuỗi thời gian & Số liệu có cấu trúc (Time-Series Data \- Nạp vào CSDL)

* **Dữ liệu giao dịch khớp lệnh lịch sử (OHLCV):** Lấy dữ liệu giá đóng cửa, khối lượng giao dịch hàng ngày của 3 chứng chỉ quỹ (FUEVFVND, E1VFVN30, FUESSVFL) và các cổ phiếu thành phần trong rổ (khoảng thời gian 2 năm gần nhất).  
  * *Công cụ thu thập:* Thư viện Python mã nguồn mở vnstock / vnstock3 hoặc API CafeF/Vietstock.  
* **Chuỗi giá trị tài sản ròng (NAV/CCQ):** Thu thập chuỗi số liệu NAV hàng ngày từ báo cáo thống kê của quỹ Dragon Capital và SSIAM (2 năm gần nhất).  
* **Dữ liệu dòng tiền ngoại:** Khối lượng mua/bán ròng của khối ngoại đối với từng mã ETF theo từng phiên giao dịch.

---

## 3\. TỔNG QUAN KHÓ KHĂN HIỆN TẠI VÀ HƯỚNG MỚI ĐỀ XUẤT

### 3.1. Điểm nghẽn kỹ thuật hiện tại

1. **Ảo giác số liệu khi tính toán**  
2. **Đứt gãy bảng danh mục**  
3. **Thiếu khả năng kết hợp dữ liệu tĩnh (văn bản) và dữ liệu động (chuỗi giá/NAV)**

---

## 4\. LỘ TRÌNH TIẾP CẬN 4 BƯỚC & TÀI LIỆU THAM KHẢO Q1 LIÊN QUAN

\[BƯỚC 1: Xử lý Chunking Bảng danh mục & Metadata\]  
         ↓  
\[BƯỚC 2: Tích hợp Tool-Calling\]  
         ↓  
\[BƯỚC 3: Kiểm soát Độ bất định & Hỗ trợ ra quyết định\]  
         ↓  
\[BƯỚC 4: Thiết lập Benchmark Đánh giá & Thực nghiệm Đối sánh\]

### Bước 1: Tiền xử lý phân mảnh tài liệu quỹ (Chunking & Structured Indexing)

* **Mục tiêu kỹ thuật:** Xử lý PDF thông báo cơ cấu rổ và bản cáo bạch, giữ nguyên cấu trúc bảng tỷ trọng cổ phiếu.  
* **Tạp chí mục tiêu:** Expert Systems with Applications (ESWA), Engineering Applications of Artificial Intelligence (EAAI).  
* **Bài báo nghiên cứu gợi ý:**  
  1. *Dual-granularity chunking and dynamic context augmentation: An optimization method for retrieval-augmented generation*  
     * Tạp chí: Expert Systems with Applications (ESWA).  
     * Trọng tâm: Phân mảnh 2 tầng (chunk nhỏ để vector search, chunk lớn để nạp ngữ cảnh trọn vẹn bảng biểu cho LLM).  
     * Link gốc ScienceDirect (DOI): [https://doi.org/10.1016/j.eswa.2026.133546](https://doi.org/10.1016/j.eswa.2026.133546) (Bật extension Unpaywall để tải toàn văn)  
  2. *CFinRAG: Structured Index Construction with Knowledge Graphs and Rule Verification*  
     * Phân nhánh: EAAI Track / Financial NLP.  
     * Link mã nguồn & tài liệu: [https://github.com/chtkg/CFinRAG](https://github.com/chtkg/CFinRAG)

### Bước 2: Tích hợp công cụ định lượng ETF (Tool-Calling / PoT) & Chuỗi thời gian NAV

* **Mục tiêu kỹ thuật:** Thiết kế Semantic Router kích hoạt các module Python tính toán chuyên biệt cho ETF:  
* **Tạp chí mục tiêu:** Expert Systems with Applications (ESWA), Engineering Applications of Artificial Intelligence (EAAI).  
* **Bài báo nghiên cứu gợi ý:**  
  * *FinAgent-RAG: Agentic Retrieval-Augmented Generation for Financial Document Question Answering*  
    * Tạp chí: ESWA Track / arXiv.  
    * Trọng tâm: Kỹ thuật Program-of-Thought (PoT) sinh mã Python thực thi trong sandbox để tính toán số liệu tài chính.  
    * Link bài báo: [https://arxiv.org/abs/2605.05409](https://arxiv.org/abs/2605.05409) | GitHub: [https://github.com/cheer932041235/FinAgent-RAG](https://github.com/cheer932041235/FinAgent-RAG)  
  * *TCNAttention-RAG: Stock Prediction Framework Based on Financial Report Analysis and Time-Series Integration*  
    * Phân nhánh: Engineering Applications Track.  
    * Link gốc Preprints: [https://www.preprints.org/manuscript/202504.1921](https://www.preprints.org/manuscript/202504.1921)

### Bước 3: Kiểm soát độ bất định và Hỗ trợ ra quyết định (Decision Support)

* **Mục tiêu kỹ thuật:** Đảm bảo hệ thống không đưa ra dự báo giải ngân sai lệch; gắn nhãn mức độ tự tin cho phân tích.  
* **Tạp chí mục tiêu:** Decision Support Systems (DSS).  
* **Bài báo nghiên cứu gợi ý:**  
  * *Bayesian RAG: Uncertainty-Aware Retrieval for Reliable Financial Decision Support*  
    * Tạp chí: Decision Support Systems Track / PMC.  
    * Trọng tâm: Sử dụng suy luận Bayes để lượng hóa độ bất định, gán nhãn Confidence Score cho câu trả lời.  
    * Link gốc PMC: [https://pmc.ncbi.nlm.nih.gov/articles/PMC12886353/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12886353/)

### Bước 4: Phương pháp luận thực nghiệm và Bộ tiêu chí đánh giá (Evaluation)

* **Mục tiêu kỹ thuật:** Thiết lập benchmark kiểm thử chuẩn hóa và đo lường độ chính xác trên 3 tầng (Retrieval, Generation, Numerical Exact Match).  
* **Bài báo nghiên cứu gợi ý:**  
  1. *FinanceBench: A New Benchmark for Evaluating Large Language Models on Financial Reasoning*  
     * Link gốc arXiv: [https://arxiv.org/abs/2311.11944](https://arxiv.org/abs/2311.11944)  
  2. *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation*  
     * Link gốc arXiv: [https://arxiv.org/abs/2408.08067](https://arxiv.org/abs/2408.08067)