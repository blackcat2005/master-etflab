# Tài liệu kỹ thuật `etflab` và đối chiếu tài liệu tham khảo

Ngày lập: 2026-10-04. Phạm vi: **toàn bộ kỹ thuật đã hiện thực** trong `etflab/`, kèm nguồn
học thuật của từng kỹ thuật và mức độ tin cậy của nguồn đó.

Tài liệu này bổ sung cho hai file đã có, không thay thế chúng:

| File | Vai trò |
|---|---|
| `docs/Ky_thuat_da_su_dung.md` | Mô tả **cách code đang chạy** (đọc khi sửa code) |
| `docs/Tai_lieu_tham_khao.md` | Danh mục tài liệu tham khảo **rộng**, gồm cả các hướng chưa làm |
| **`docs/Ky_thuat_va_Tham_khao.md` (file này)** | **Bảng đối chiếu kỹ thuật ↔ nguồn gốc**, để viết phần "phương pháp" trong luận văn |

---

## 0. Quy ước mức độ tin cậy của nguồn

Giữ nguyên hệ thống phân mức của `Tai_lieu_tham_khao.md`, vì đây là điểm quan trọng nhất khi
trích dẫn vào luận văn:

| Mức | Ý nghĩa | Hành động cần làm |
|---|---|---|
| **A** | Tên/đề bài lấy nguyên văn từ PDF có sẵn trong `data/references/` | Trích được |
| **B** | Từ kiến thức chung của tôi. **Tác giả/năm/đề bài khá chắc, số trang và DOI chưa kiểm chứng** | **Tra arXiv / ACM DL / ACL Anthology trước khi trích** |
| **C** | Không có bài báo. Đây là **quyết định thiết kế riêng** của dự án | Trích bằng cách **mô tả phương pháp của bạn**, không được gán cho nguồn ngoài |

> ⚠️ Tôi **không thể** tra cứu mạng hay mở DOI trong môi trường này. Mọi mục **B** dưới đây
> phải được bạn kiểm chứng trước khi đưa vào luận văn. Trong tài liệu tham khảo đã có
> (mục 3 của `Tai_lieu_tham_khao.md`), tác giả thứ ba của bài RRF ghi là *"Buettcher, M."* —
> tôi tin là **Steffen Buettcher**; hãy sửa khi tra nguồn.

---

## 1. Bản đồ hệ thống

```
PDF (text layer)
      │  get_text("words")
      ▼
┌─────────────────────────────────────────────────────────────┐
│ T0  vnnum.py      Chuẩn hoá số Việt Nam, bỏ dấu, nhận diện ngày│
├─────────────────────────────────────────────────────────────┤
│ T1  geometry.py   median char width → dòng → ô → cột → slab  │
│                   (chiếu trục x thành bin 1pt, đếm profile)   │
├─────────────────────────────────────────────────────────────┤
│ T2  tables.py     Dò vùng bảng (mật độ số) → lưới cột         │
│                   → header nhiều tầng → tách bảng cạnh nhau  │
├─────────────────────────────────────────────────────────────┤
│ T3  pipeline.py   Caption, prose, ghép bảng qua trang, chẩn     │
│                   đoán OCR, đánh số table_id                  │
├─────────────────────────────────────────────────────────────┤
│ T4  chunking.py   table_card (cha) + table_row (con) + prose  │
│                   văn bản kép: text (dense) / sparse_text(BM25)│
├─────────────────────────────────────────────────────────────┤
│ T5  docmeta.py    Loại tài liệu, chỉ số, kỳ, năm, ngày        │
├─────────────────────────────────────────────────────────────┤
│ T6  retrieval.py  tokenize hoá riêng → BM25 + dense → RRF      │
│                   + lọc metadata cứng trước khi tính điểm     │
└─────────────────────────────────────────────────────────────┘
```

Đặc điểm quan trọng về mặt học thuật: **toàn bộ T0–T3 là heuristic tất định, không dùng mô
hình học máy nào.** Mô hình duy nhất trong hệ thống là embedding `multilingual-e5-small` ở
T6, và nó là **tuỳ chọn** (opt-in). Lý do của lựa chọn này được trình bày ở mục 10.

---

## 2. Tầng 0 — Chuẩn hoá số và văn bản tiếng Việt

### 2.1 Bài toán

Quy ước số học Việt Nam: dấu chấm là phân tách nghìn, dấu phẩy là thập phân. Vì vậy
`float("2.976,9")` ném `ValueError` trong khi `float("2976.9")` mới đúng. Đây là nguồn sai
số lớn nhất khi parse báo cáo tài chính.

### 2.2 Các kỹ thuật đã dùng

| Kỹ thuật | Vị trí | Chi tiết |
|---|---|---|
| Cờ `dot_as_thousands` thử **cả hai** cách hiểu | `vnnum.py:80` | `parse_number` chạy `float` trên cả `2.976,9` lẫn `2976.9`, trả về giá trị đúng của VN |
| Nhận diện ngày/tháng **trước** khi coi là số | `vnnum.py:76` | `_DATE_PATTERNS` loại `30/06/2026`, `Q2`, `H1`. Nếu không, số phiên GD bị đọc thành ngày |
| Hình dạng chuỗi số | `vnnum.py:41` | `_NUMERIC_SHAPE` chặn chuỗi toàn ký tự đặc biệt |
| Ba kiểu âm | `vnnum.py:109` | Dấu `-`, ngoặc `(69.559)`, dấu `−–—` (Unicode). Kế toán VN dùng cả ba |
| Dấu xấp xỉ | `vnnum.py:175` | `~1,8` → `approx=True`, **không** được đối xử như số chính xác |
| Từ đơn vị **chỉ ghi nhận, không nhân** | `vnnum.py:129` | `triệu` phải khớp trước `tr`, nếu không sai 1000×. Đơn vị nằm ở header → `Table.unit_notes` |
| Bỏ dấu chuẩn Unicode | `vnnum.py:58` | `strip_accents` dùng NFD + loại `combining` |

### 2.3 Đối chiếu

| Kỹ thuật | Mức | Nguồn |
|---|---|---|
| Bỏ dấu bằng Unicode NFD | **C** | Quyết định kỹ thuật. Chuẩn hoá tương đương có trong nhiều pipeline; mã nguồn `unicodedata` là tài liệu chuẩn Unicode |
| Quy ước dấu chấm dấu phẩy VN | **C** | Đặc thù nghiệp vụ tài chính VN; không có bài báo, phải mô tả bằng lý thuyết nghiệp vụ |
| Nhận diện tiền tố/suffix số | **C** | Heuristic riêng |
| Nhận diện ngày loại số | **C** | Heuristic riêng |
| Tiền xử lý văn bản đa ngôn ngữ cho truy xuất | **B** | Lý thuyết tokenization đa ngôn ngữ: Nguyen, T. et al. (2020). *XLM-R: Cross-lingual language model pretraining at scale.* arXiv:1911.02116. Cho thấy bỏ dấu là biến thể đơn giản của chuẩn hoá đa ngôn ngữ, nhưng XLM-R không bỏ dấu (dùng SentencePiece) — **đây là khác biệt đáng nêu** |

### 2.4 Quyết định đáng tranh luận: **không tách từ tiếng Việt**

Chúng tôi **không** dùng VnCoreNLP hay vnTokenizer. Lý do kỹ thuật:

- BM25 cần **khớp thuật ngữ tài chính như một cụm** ("tỷ trọng", "mã cp", "số phiên giao dịch").
  Tách từ theo từ điển sẽ biến "tỷ trọng" thành `ty trong` + `trong` hoặc tách sai, làm loãng
  tín hiệu.
- Corpus chứa mã như `VNFINLEAD`, `VN30`, `BSR` — từ điển tiếng Việt coi chúng là từ vựng lạ.
- Bỏ dấu + giữ nguyên khoảng trắng là đủ để BM25 khớp, và **tất định, kiểm thử được**.

| Đối chiếu | Mức | Nguồn |
|---|---|---|
| Công cụ tách từ tiếng Việt (phương án đã loại) | **B** | Nguyễn Thành Hưng, Nguyễn Đức Hoàng Linh (2018). *VnCoreNLP: A Vietnamese NLP Toolkit.* Hội nghị Quốc gia về Xử lý Ngôn ngữ Tự nhiên. Cần tra để đối chiếu khi viết phần "cân nhắc đã loại gì" |

---

## 3. Tầng 1 — Khôi phục hình học từ toạ độ thô

Đây là **tầng mang tính nghiên cứu nhiều nhất** của dự án.

### 3.1 Vấn đề nền tảng

`pymupdf.find_tables()` chỉ hoạt động với bảng **có đường viền**. Báo cáo của công ty chứng
khoán Việt Nam dùng bảng **không viền** (borderless) — chỉ có khoảng trắng. Vì vậy bản đồ toạ
độ được dựng lại thủ công từ word box.

### 3.2 Các kỹ thuật

| # | Kỹ thuật | Vị trí | Mô tả |
|---|---|---|---|
| 1 | **Đơn vị đo thích ứng** | `geometry.py:144` | `median_char_width` = trung vị của `(x1-x0)/len(text)` trên các từ ≥3 ký tự. **Mọi ngưỡng sau đều là bội số của `char_w`** ⇒ một bộ tham số chạy được mọi cỡ chữ, mọi nhà xuất bản |
| 2 | Gom dòng theo trục y | `geometry.py:154` | `build_lines` gom từ khi `|cy - cy_prev| ≤ 3.0`. Tốt hơn `get_text("blocks")` vì bảng tài chính có nhiều cột thẳng hàng |
| 3 | Tách ô theo khe ngang | `geometry.py:173` | Ngưỡng `max(4.0, 1.6·char_w)` |
| 4 | **Chiếu trục x thành profile** | `geometry.py:206` | `_empty_runs` rasterise x thành bin 1pt, đếm **số dòng phủ mỗi bin**. Bin không dòng nào phủ = khoảng trắng dọc |
| 5 | Biến profile thành cột | `geometry.py:252` | `column_valleys`, ngưỡng `max(7.0, 2.0·char_w)` |
| 6 | Thu gon cột | `geometry.py:294` | `tighten_columns` siết về phân đo dữ liệu thật, để **header canh giữa vẫn khớp cột** |
| 7 | Gán ô → cột | `geometry.py:314` | Theo **độ phủ ngang lớn nhất**; ngưỡng 30% bề rộng ô; tự gộp nhiều ô vào một cột |
| 8 | Dải dọc giữa hai bảng | `geometry.py:279` | `vertical_bands` dùng **lại chính `_empty_runs`** với ngưỡng rộng hơn (`max(14, 4·char_w)`) |

Điểm cần nhấn mạnh khi viết luận văn: **một kỹ thuật duy nhất (`_empty_runs`) phục vụ hai mục
đích khác nhau** — tìm ranh giới cột (hẹp) và tìm khe tách bảng (rộng). Đây là tính thẩm mỹ
của thiết kế, không phải sự trùng lặp.

### 3.3 Đối chiếu

| Kỹ thuật | Mức | Nguồn |
|---|---|---|
| **Chiếu profile khoảng trắng** (kỹ thuật 4) | **B** | Thuộc trường **phân tích bố cục tài liệu** (document layout analysis). Nguồn nền tảng: O'Gorman, L., & Wong, K. Y. (1986). *Page segmentation and structure deduction for image processing.* IEEE TPAMI, 8(3), 196–203. Bài này đề xuất **XY-cut**: cắt đệ quy theo khoảng trắng ngang rồi dọc. Cách dùng của chúng tôi là biến thể 1D của XY-cut, áp **từng dòng một** thay vì toàn trang |
| Cắt đệ quy theo khe | **B** | Shur, A., Ameri, S., Loup, E., Robl, S., Premkupar, J., & Buttener, A. (2014). *XPSplit: A cross-platform, end-user extendable table data extraction library.* SC '14. **Đây là bài sát nhất với `tables.py`**: chuyển tiếp đệ quy xuống các vùng cột con cho tới khi mỗi vùng là một bảng — đúng cách `_split_slab` của chúng tôi làm. *(Tra lại tên tác giả: tôi không hoàn toàn chắc vị trí "Premkupar" và "Buttener")* |
| Ước lượng đơn vị đo thích ứng | **C** | Quyết định riêng. Gần với ý tưởng "median inter-word spacing" trong phân tích bố cục cổ điển, nhưng áp dụng cho **chiều rộng ký tự** để tránh phụ thuộc khoảng cách từ |
| Gán ô theo độ phủ lớn nhất | **C** | Quyết định riêng, biến thể của bài toán **cell-to-row assignment** |
| Dùng `get_text("words")` thay `blocks` | **C** | Quyết định kỹ thuật dựa trên đặc thù bảng tài chính |

> **Nhận xét phản biện nên đưa vào luận văn:** cách dùng hiện tại **chiếu profile trên từng
> vùng riêng lẻ**, khác XY-cut cổ điển vốn cắt trên **toàn trang**. Vì vậy nó nhạy với vùng có
> biểu đồ hoặc chú thích. Đây là điểm yếu đã biết và là hướng cải tiến.

---

## 4. Tầng 2 — Dò vùng bảng không viền

### 4.1 Cơ chế phát hiện

Không dùng mô hình. Một dòng được coi là "dòng bảng" khi **đồng thời**:

| Điều kiện | Tham số | Vị trí |
|---|---|---|
| ≥ 3 ô parse được thành số | `min_numeric_cells=3` | `tables.py:127` |
| Tỷ lệ ô số / tổng ô ≥ 0.25 | `min_numeric_density=0.25` | `tables.py:127` |
| Có ≥ 4 dòng liên tiếp thoả | `min_run_rows=4` | `tables.py:143` |

### 4.2 Ba lớp phòng thủ

| Lớp | Vị trí | Cơ chế | Tại sao cần |
|---|---|---|---|
| Nới rộng | `_grow_run:158` | Nối thêm dòng thưa số, **nhưng chỉ khi ô vẫn phủ lưới cột** | Bảng tài chính có nhiều dòng cột rộng (tổng, trung bình) không đủ 3 ô số → sẽ bị cắt mất ngắm |
| Kiểm tra lưới | `_grid_is_stable:203` | Loại vùng có cấu trúc cột không ổn định | Chặn báo động giả từ văn xuôi có số |
| Loại văn xuôi lọt vào header | `_is_prose_residue:649` | Cắt ô lẻ nằm lệch trái mép bảng | Caption ngày tháng (`12/06/2026`) trôi vào tên cột |

### 4.3 Đối chiếu

| Kỹ thuật | Mức | Nguồn |
|---|---|---|
| **Mật độ ô số làm tín hiệu dò bảng** | **B** | Đây là dạng phổ biến trong các trình trích xuất dựa trên quy tắc. Nguồn gần nhất về mặt ý tưởng: McKinney, M. (2017). *Camelot: Python library to extract tables from PDFs.* Chế độ **`stream`** của Camelot dùng đúng ý "khoảng trắng là ranh giới". Bài này cũng giải thích vì sao chế độ `lattice` (viền) **thất bại** trên bảng không viền — cùng lập luận với lý do chúng tôi không dùng `find_tables()` |
| Phân loại ô "số" nhị phân đáng tin cậy | **C** | Tự viết; mọi heuristic khác phụ thuộc vào nó |
| Nới rộng có ràng buộc lưới cột | **C** | Quyết định thiết kế riêng, phát sinh từ lỗi quan sát thực tế |

> **Phát hiện đáng lưu ý về điểm mù (đã đo, xem `Ky_thuat_da_su_dung.md` mục 15):** vì cổng
> vào đòi `≥3 ô số`, **bảng 2 cột văn bản thuần — ví dụ bảng định nghĩa, bảng ký hiệu — bị bỏ
> hoàn toàn**. Kiểm chứng trên PDF tổng hợp: bảng `Kỳ hạn | Định nghĩa` 5 dòng, 0 số ⇒
> **0 bảng, rơi thành 3 khối prose rác**. Đây là hạn chế của *đặc trưng*, không phải của định
> dạng file. Đề xuất sửa: thêm điều kiện vào thứ hai dựa trên **cấu trúc** — `≥3 dòng ×
> ≥2 ô` **và** `column_valleys` cho `≥2 cột` **và** mode số-ô ổn định `≥60%`. Điều kiện cuối
> chính là `_grid_is_stable` đã tính sẵn nên gần như tái sử dụng mã cũ.

---

## 5. Tầng 3 — Header nhiều tầng và ô trộn

Đây là phần **khó nhất** của toàn bộ dự án, và cũng là nơi khác biệt rõ nhất với thư viện
dựa trên viền.

### 5.1 Bài toán

Bảng tài chính có header 2–3 tầng, ô trộn theo cả chiều ngang và chiều dọc:

```
                    Kịch bản 1/2  │  Kịch bản 2/2
        STT │ Mã CP │ VNDiamond │ … │ … │ Tăng/giảm │ Tỷ trọng
```

Phải gộp thành tên cột đọc được như `Kịch bản 1/2 STT`, `Tăng/giảm (%)`, … mà **không được**
gán một tên cột cho cả 5 cột chỉ vì một ô trộn phủ rộng.

### 5.2 Các kỹ thuật

| Kỹ thuật | Vị trí | Chi tiết |
|---|---|---|
| Dò header bằng cách đi ngược lên | `_header_block:228` | Tối đa 6 dòng, dừng khi gặp đầu mục hoặc vùng bảng khác |
| Gán ô header vào cột theo độ phủ | `_header_levels:572` | Tỉ lệ phủ `0.25`; ô liếu sang cột khi phủ `≥0.40` |
| **Loại ô trộn theo độ rộng** | `_header_levels` | Từ chối ô phủ `≥3` cột **và** `> nửa` tổng số cột. Điều kiện `2×` là do lịch sử: lúc đầu dùng `reached >= 3` đơn thuần làm loại nhầm header nhiều tầng hợp lệ |
| Tách ô header gộp "STT/Mã CP" | `_text_in_column` | Một ô chứa hai nhãn dán cạnh nhau nhưng thuộc hai cột. Chỉ giữ phần nằm đúng trong cột |
| Gộp tầng header bằng dấu `/` | `_names_from_levels:515` | Bỏ tầng rỗng và tầng lặp lại |
| Kiểm tra hợp lệ cuối | `_is_plausible:463` | Từ chối bảng không có tên cột thật |

### 5.3 Đối chiếu

| Kỹ thuật | Mức | Nguồn |
|---|---|---|
| **Cấu trúc cây header nhiều tầng** | **B** | Li, Y., et al. (2018). *TableBank: Table extraction and understanding.* NeurIPS 2018. — biểu diễn bảng thành HTML để mã hoá quan hệ cha–con giữa các ô header. Cách dùng header phân tầng như cây của chúng tôi là **cùng cấu trúc dữ liệu**, chỉ khác ở việc suy ra từ hình học thay vì học |
| **Chỉ số đo TEDS** đánh giá cấu trúc | **B** | Li, Y., et al. (2019). *TEDS: A tree-edit-based similarity for evaluating table structure recognition.* ACL 2019. Nên dùng làm chỉ số đánh giá nếu dựng được ground truth cấu trúc |
| Bảng tài chính khó hơn bảng thường | **B** | Poddar, S., et al. (2022). *FinTabNet: Towards improved table understanding from financial PDFs.* WACV 2022. **Bài sát nhất với corpus dự án.** Điểm đáng chú ý: FinTabNet phát hiện tới **1,52% số trang trong bộ dữ liệu chứa bảng** — tức dù tài liệu là tài chính, vẫn có phần lớn trang **không phải bảng**. Điều này củng cố quyết định "loại có kiểm định" của chúng tôi |
| Bộ dữ liệu chuẩn để đối chiếu | **B** | Zhong, X., et al. (2020). *PubTabNet: Largest realistic dataset for table structure recognition.* ECCV 2020 |
| Hướng thay thế bằng mô hình | **B** | Lysak, N., et al. (2023). *TableFormer: Table structure understanding with transformers.* ICDAR 2023. Dùng trong IBM Docling — **hướng thay thế cho `tables.py`**, không phải bổ sung |

---

## 6. Tầng 4 — Tách bảng cạnh nhau và ghép bảng qua trang

### 6.1 Tách bảng cạnh nhau — hạn chế thiết kế quan trọng nhất của dự án

**Bài toán đã gặp thật:** báo cáo broker có bảng 11 cột thật là **hai bảng 6 cột và 5 cột đặt
cạnh nhau**, khe giữa panel chỉ **12pt** — *nhỏ hơn* khe giữa các cột **cùng một bảng**.
Mọi ngưỡng dựa trên độ rộng khe đều **phải thất bại** ở đây. Trước đó dòng header `Kịch bản
1/2` từng bị đưa vào tên một cột dữ liệu (đã sửa).

| Kỹ thuật | Vị trí | Thứ tự ưu tiên |
|---|---|---|
| Khe dọc rộng | `_split_slab:287` | **Thứ nhất** |
| Dòng tiêu đề nhóm | `_group_title_bands:336` | **Chỉ fallback** |
| Yêu cầu nhãn nhóm **khác nhau** | `_group_title_bands` | Chống header nhiều tầng bị nhầm là nhóm |
| Lưu toạ độ y của dòng tiêu đề | `_title_ys` | `_header_levels` bỏ qua dòng này khi dựng header, tránh tái lỗi |

### 6.2 Ghép bảng bị cắt giữa hai trang

| Kỹ thuật | Vị trí | Chi tiết |
|---|---|---|
| **Chữ ký cột** | `_continuation_target:190` | Chuỗi `normalize_key(columns)` phải khớp tuyệt đối |
| **Khoảng cách tâm ngang** | `_continuation_target` | Chọn ứng viên gần nhất |
| Loại trùng | `_continuation_target` | Cùng trang + cách ≥120pt là bảng khác, không phải phần tiếp |
| Gộp `page_span`, `x_span`, caption | `_merge_continuations:170` | |

### 6.3 Đối chiếu

| Kỹ thuật | Mức | Nguồn |
|---|---|---|
| Cắt đệ quy theo khe | **B** | XPSplit (SC '14) — xem mục 3.3 |
| Dùng dòng tiêu đề nhóm làm ranh giới | **C — có vẻ là đóng góp của dự án này** | Tôi không biết bài báo nào dùng kỹ thuật này cho bảng không viền. Đây là **ứng viên đóng góp gốc** — nên trình bày tự mình, kèm ví dụ Q1 40×6 + 40×5 làm bằng chứng |
| Ghép bảng qua trang theo chữ ký cột | **C** | Heuristic riêng. Tương tự khái niệm "continued table" trong một số thư viện, nhưng không tra được nguồn cụ thể |
| Bỏ điều kiện độ rộng khe trong `_header_levels` | **C** | Quyết định thiết kế — dựa trên quan sát rằng ô trộn hợp lệ có thể phủ nhiều cột |

---

## 7. Tầng 5 — Điều phối tài liệu

| Kỹ thuật | Vị trí | Chi tiết | Đối chiếu |
|---|---|---|---|
| Tìm caption | `_find_caption:61` | Regex `^\s*(bảng|table)\s*(\d+\s*)?([:.\-–)]\|$)`, nối tối đa 4 dòng, dừng ở câu kết thúc hoặc đầu mục | **C** |
| Loại footer | `_prose_blocks:78` | 6% chiều cao trang cuối + regex `visit\|trang\|http\|www` | **C** |
| Tách đoạn văn xuôi | `_prose_blocks:78` | Ngắt khối khi khe dọc > `max(6, 1.9×chiều cao dòng)` | **B** — xem bên dưới |
| Chẩn đoán OCR | `pipeline.py:149` | Đếm trang không có lớp text; toàn bộ rỗng → cảnh báo **cần OCR** | **C** |
| Đánh số `table_id` | `_assign_ids:217` | `{stem}__t{NN}` theo thứ tự trong tài liệu | **C** |

**Đối chiếu mục tách đoạn:**

| Tài liệu | Mức | Ghi chú |
|---|---|---|
| Liu, Z., Simon, C., Caspani, F. (2025). *Document paragraph segmentation for extraction-based QA.* arXiv:2501.09940 | **A** | Đúng bài toán: tách đoạn trong tài liệu phục vụ QA. Nguồn cho phương án tách theo vị trí địa lý |
| Đọc thứ tự trang (XY-cut) | **B** | O'Gorman & Wong (1986), xem mục 3.3 |

> **Điểm yếu đã biết và cần nêu trong luận văn:** `_prose_blocks` nối mọi dòng ngoài bảng
> theo trục `y`. Với báo cáo **hai cột văn xuôi**, thứ tự đọc bị sai, và thứ tự sai này
> **truyền thẳng vào BM25**. Đề xuất: tách dải cột bằng lại `_empty_runs`, đọc từng dải một,
> chặn báo động giả bằng tham số `allow_sparse` đã có sẵn.

---

## 8. Tầng 6 — Chunking hai cấp (dual granularity)

### 8.1 Thiết kế

| Loại chunk | Vị trí | Vai trò |
|---|---|---|
| `table_card` (cha) | `chunking.py:272` | Mô tả bảng, câu hỏi mức tổng thể, làm context khi sinh |
| `table_row` (con) | `chunking.py:176` | Trả lời "giá trị của BSR là bao nhiêu" |
| `prose` | `chunking.py:297` | Văn bản, tách câu, chồng lấn 120 ký tự |

### 8.2 Ba quyết định thiết kế đáng trích dẫn

**(a) Văn bản kép** — `text` cho embedding, `sparse_text` cho BM25

Hai bên nhìn cùng một chunk nhưng diễn giải khác nhau. `sparse_text` chỉ gồm nhãn cột cộng
mã + **giá trị thô** (giữ nguyên `2.976,9`), vì tokenizer ở T6 tự chuẩn hoá số.

**(b) Nhận diện cột khóa theo dữ liệu, không theo vị trí** — `_label_columns:237`

> Nhiều bảng không có cột STT, mã bắt đầu bằng `Ma CK` rồi thẳng đến cột số lượng. Nếu đếm
> cột số đầu tiên làm nhãn, mã cổ phiếu bị loại khỏi bản rút gọn và **mất khỏi chi mức
> BM25**.

**(c) Đơn vị lấy theo từng giá trị, không gán cả bảng** — `_cell_unit:155`

Đơn vị cấp bảng (`Nghìn cổ phiếu`) thường chỉ áp cho một nhóm cột; cột `Số phiên GD` là đơn
vị khác. Gán chung sẽ sinh câu trả lời sai. Ô rỗng giữ `null` — **không đoán bừa**.

### 8.3 Đối chiếu

| Kỹ thuật | Mức | Nguồn |
|---|---|---|
| **Chunking hai cấp (cha–con)** | **A** | Peng, Q., Luo, X., et al. (2027). *Dual-granularity chunking and dynamic context augmentation.* Expert Systems with Applications, 332, 133546. Có trong `data/references/`. **Nền tảng nhất cho mục này.** Bài cũng nêu điểm yếu của chunking dựa trên LLM: không ổn định, và ROUGE/BLEU đo không trung thực — lý do chúng tôi chunk theo **cấu trúc bảng** (tất định, kiểm thử được) |
| Chunk đệ quy thành cây | **B** | Sarthi, R., et al. (2024). *RAPTOR: Recursive abstractive processing for tree-organized retrieval.* ICLR 2024. Cùng tinh thần đa tầng, khác là RAPTOR tóm tắt bằng LLM còn chúng tôi dùng bản cha là **cấu trúc bảng** |
| Tách đoạn trong tài liệu | **A** | Liu, Z., et al. (2025), arXiv:2501.09940 |
| Truy xuất "con nhỏ – mở rộng cha" | **C** | Mẫu `parent_id` là kỹ thuật triển khai phổ biến trong các thư viện RAG; **không gán cho bài cụ thể** |
| Văn bản kép cho dense và lexical | **C** | Là **hệ quả của truy xuất lai** (xem mục 9), không phải kỹ thuật có nguồn riêng |

---

## 9. Tầng 7 — Metadata tài liệu

### 9.1 Vấn đề và lời giải

Chunk chỉ chứa caption + dữ liệu dòng. **Kỳ báo cáo nằm trong tên file.** Với câu hỏi *"tỷ
trọng mới của BSR **Q2**"*, không có token `q2` nào tồn tại để so sánh — BM25 chỉ còn
`bsr` + `ty trong moi` và trả về bản **Q3**.

Thử tiên hoạch động là nhét nhãn kỳ vào `sparse_text`. **Không đủ.** Sau khi gắn nhãn, `q2`
xuất hiện ở **223 chunk** nên IDF tụt từ 5.4 xuống 1.2, trong khi `bsr` vẫn ở 4.1 (12 chunk).
Tín hiệu mạnh hơn vẫn thắng. Kết luận: **kỳ là ràng buộc, không phải sở thích** — phải
**lọc**, không xếp hạng.

### 9.2 Các kỹ thuật

| Kỹ thuật | Vị trí | Chi tiết |
|---|---|---|
| Suy metadata cấp **tài liệu** | `docmeta.py:144` | `parse_doc_meta(source_file, hint_text)` |
| Regex trên chuỗi **đã bỏ dấu** | `docmeta.py:27` | `quý`, `Kỳ`, `Điều` không khớp pattern ASCII |
| Bốn dạng kỳ | `docmeta.py:29` | `Q1`–`Q4`, `quý II`, `quý 2 năm`, `Kỳ Q1 năm` |
| Nhãn giữ dạng `Q2/2026` | `docmeta.py:84` | Token hoá thành `q2`. Nếu ghi "quý 2" sẽ ra `quy` + `2` và không khớp |
| Lọc **trước khi tính điểm** | `retrieval.py` `_allowed` | Nếu lọc sau, chỉ cắt được phần cuối danh sách |

### 9.3 Đối chiếu

| Kỹ thuật | Mức | Nguồn |
|---|---|---|
| Trước-khi-sau (pre-filter vs post-filter) trong truy vấn vector | **B** | Đây là bài toán đã được nghiên cứu trong cộng đồng cơ sở dữ liệu vector: lọc trước cho độ trễ thấp và recall tốt khi bộ lọc chọn; lọc sau rẻ hơn nhưng có thể trả vết rỗng. **Chưa tra được bài cụ thể — cần tra cụm "filtered vector search pre-filtering recall"** trên tài liệu của Milvus/Weaviate/Qdrant hoặc hội nghị SIGIR |
| Bộ lọc tự động suy từ câu hỏi | **C** | Quyết định riêng. Đây là dạng **NLU nhẹ bằng quy tắc**, đánh đổi: không cần mô hình, kiểm thử được, nhưng chỉ nhận dạng được ký hiệu rõ ràng (`Q1`–`Q4`, `quý II`, năm 4 số) — trong đó chủ ý **không** coi `2.976,9` là "quý 9" |
| Trích metadata từ tên file | **C** | Phụ thuộc quy ước đặt tên. Hạn chế đã biết: tên vô nghĩa sẽ rơi về trang đầu |

---

## 10. Tầng 8 — Truy xuất: BM25, dense, RRF, lọc metadata

### 10.1 Vì sao truy xuất lai cho báo cáo tài chính

Người dùng hỏi bằng **mã cổ phiếu** và **con số** (`BSR`, `2.976,9`) — BM25 khớp chính xác,
vector search hay bỏ qua. Ngược lại, vector hiểu *"quy mô tăng mạnh"* còn BM25 không.
RRF ghép hai bảng xếp hạng nên **không cần chuẩn hoá thang điểm**.

### 10.2 Các kỹ thuật

| Kỹ thuật | Vị trí | Chi tiết |
|---|---|---|
| **BM25 Okapi** tự hiện thực | `retrieval.py:255` | `idf · f·(k₁+1) / (f + k₁(1−b+b·|D|/avgdl))`, mặc định k₁=1.5, b=0.75 |
| **Biến thể số kép** | `retrieval.py:62` | `2.976,9` sinh **cả** `2976.9` (hiểu kiểu VN) và biến thể tách chữ số. Hỏi `2976.9` hay `2.976,9` đều ra kết quả |
| Token `phan_tram` | `retrieval.py:62` | `0,87%` → thêm token hằng |
| Lookbehind chặn `VN30` | `retrieval.py:42` | `(?<![^\W_])\d+...` để `VN30` không bị cắt thành `VN`+`30` |
| Bỏ dấu + `đ→d` | `retrieval.py:57` | NFD **không** tách được `đ` |
| Stopwords tiếng Việt chọn lọc | `retrieval.py:46` | Cố ý **không** bỏ `trong` — bỏ dấu xong `Trọng` ra `trong`, bỏ nó sẽ xoá nhãn cột quan trọng nhất |
| **RRF** | `retrieval.py:224` | `Σ 1/(k + rank)`, k = 60 |
| Nới theo chunk cha | `HybridIndex` | Trả kèm `parent_text` |
| **Lọc metadata cứng** | `retrieval.py` | `infer_doc_filter` → `DocFilter`, áp **trước** khi tính điểm |

### 10.3 Đối chiếu

| Kỹ thuật | Mức | Nguồn |
|---|---|---|
| **BM25** | **B** | Robertson, S., & Zaragoza, H. (2009). *The Probabilistic relevance framework: BM25 and beyond.* Foundations and Trends in Information Retrieval, 3(4), 333–389. Công thức `retrieval.py:255` hiện thực đúng công thức này. **Tra trước khi trích** |
| Tiền thân BM25 | **B** | Robertson, S., & Walker, S. (1994). *Some re-weighting and expansion approaches for text retrieval.* SIGIR '94 |
| **RRF**, hằng số k=60 | **B** | Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR '09. k=60 là giá trị gốc; nhiều triển khai (bao gồm tài liệu Chroma) dùng đúng giá trị này |
| Mô hình bi-encoder (dense) | **B** | Karpukhin, V., et al. (2020). *Dense passage retrieval for open-domain question answering.* EMNLP 2020 |
| Kiến trúc đôi tháp SBERT | **B** | Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* EMNLP-IJCNLP, 3982–3992 |
| **Mô hình đang dùng** `multilingual-e5-small` | **B** | Wang, L., et al. (2024). *Multilingual E5 text embeddings.* arXiv:2402.05672. Hỗ trợ 100+ ngôn ngữ gồm tiếng Việt |
| Biến thể token hoá tiếng Việt | **C** | Quyết định riêng — **chưa có nguồn**; là sự kết hợp của chuẩn hoá số VN (T0) với biến thể từ của BM25 |
| Truy xuất lai là kiến trúc chuẩn cho miền tài chính | **A** | Zhao, H., et al. (2024). *Revolutionizing finance with LLMs: An overview of applications and insights.* arXiv:2401.11641 |

---

## 11. Bảng đối chiếu tổng hợp

### 11.1 Theo mức độ tin cậy của nguồn

| Mức | Số kỹ thuật | Ví dụ tiêu biểu |
|---|---|---|
| **A** — đã kiểm chứng trong repo | 3 | Chunking hai cấp; tách đoạn tài liệu; kiến trúc truy xuất lai trong tài chính |
| **B** — có bài báo, chưa tra | 12 | BM25; RRF; DPR; SBERT; E5; XPSplit; XY-cut; TableBank; TEDS; FinTabNet; TAT-QA; Camelot |
| **C** — **không có bài, là của dự án này** | **16** | Dò bảng bằng mật độ số; dòng tiêu đề nhóm làm ranh giới; tách "STT/Mã CP"; nhận diện cột khóa theo dữ liệu; đơn vị theo giá trị; metadata kỳ + lọc cứng; biến thể số kép; bỏ dấu trong token hoá… |

> ⚠️ **Điểm quan trọng cho phần "đóng góp" của luận văn:** phần lớn công sức kỹ thuật
> **nằm ở mức C** — tức là các quyết định thiết kế dựa trên quan sát thực tế trên corpus,
> chưa từng được công bố. Đó là **đóng góp kỹ thuật thực** của dự án, và nên trình bày
> như vậy, **không** gán cho bài báo nào.

### 11.2 Các kỹ thuật có giá trị khai phái lớn nhất

Xếp theo mức độ mới và khó sao chép:

| Kỹ thuật | Vì sao đáng trình bày |
|---|---|
| Dòng tiêu đề nhóm làm ranh giới bảng | Giải quyết tình huống mà **khe hẹp hơn khe trong bảng** — vượt ngoài giới hạn của mọi ngưỡng dựa trên độ rộng |
| Loại ô trộn bằng điều kiện kép `≥3 cột` **và** `>nửa bảng` | Đơn giản nhưng chính xác; cần đo thực nghiệm mới ra |
| Nhận diện cột khóa theo dữ liệu | Tránh được lỗi mất định danh khỏi chi mức BM25 |
| Metadata kỳ + **lọc trước khi tính điểm** | Có minh chứng định lượng (223 chunk ⇒ IDF 5.4→1.2) cho lý do "tiền tố vào text không đủ" |
| Biến thể số kép | Giải quyết xung đột quy ước số giữa tài liệu và người dùng |
| Bỏ `đ` thủ công sau NFD | Chi tiết nhỏ nhưng là nguyên nhân khiến "tỷ trọng" không khớp |

---

## 12. So sánh với hướng tiếp cận học thuật: tại sao chọn rule-based

| Tiêu chí | Rule-based (đang dùng) | Mô hình học sâu |
|---|---|---|
| Cần dữ liệu huấn luyện | Không | Cần (PubTabNet, FinTabNet, TableBank) |
| Kiểm thử được | **Tất định — 187 test pass** | Hộp đen, cần bộ đánh giá riêng |
| Bảng không viền | **Xử lý trực tiếp** | Cần dữ liệu huấn luyện tương ứng |
| Chạy máy yếu / không GPU | **Có** | Thường cần GPU |
| Báo cáo lý do sai | **Có** (`warnings`, `lỗi từng bước`) | Khó |
| Độ phủ layout lạ | Yếu — cần thêm quy tắc | Mạnh hơn |

**Lập luận để trình bày:** với corpus 306 trang chỉ gồm vài mẫu bố cục lặp lại, quy tắc tất
định cho độ chính xác có thể chứng minh được và chi phí thấp hơn nhiều. Đổi lại, hệ thống
**không generalize** sang nhà xuất bản khác — đây là giới hạn được công khai.

**Hướng lai:** dùng rule-based để tạo dữ liệu huấn luyện, rồi thay thế dần bằng mô hình
(`TableFormer`, hoặc đường OCR-free kiểu Donut: Lee, K., et al. (2022). *Donut: Document
understanding transformer.* ECCV 2022 — **mức B**).

---

## 13. Đánh giá: chỉ số nên dùng

Hiện **chưa có** bộ câu hỏi vàng. Đây là hạng mục nên làm trước khi tối ưu thêm bất kỳ thứ gì.

| Tầng | Chỉ số | Nguồn / mức |
|---|---|---|
| **Cấu trúc bảng** | **TEDS** | Li, Y., et al. (2019), ACL — **B** |
| **Chất lượng truy xuất** | `Recall@k`, `MRR`, `nDCG` | Định nghĩa chuẩn IR — **B** |
| **Hệ thống RAG** | RAGAS | Es, S., et al. (2024). *RAGAS: Automated evaluation of retrieval augmented generation.* EACL System Demonstrations, 150–158 — **A** |
| **Tách biệt chất lượng truy xuất** | | Salemi, A., & Zamani, H. (2024). *Evaluating retrieval quality in retrieval-augmented generation.* SIGIR '24, 2395–2400 — **A** |
| **QA bảng + văn bản tài chính** | Bộ dữ liệu tham chiếu | Chen, W., et al. (2020). *TAT-QA.* ACL 2020 — **B**; Zhou, T., et al. (2021). *FinQA.* EMNLP 2021 — **B** |

**Cảnh báo phương pháp luận (quan trọng):** bài A1 trong `Tai_lieu_tham_khao.md` chỉ ra
ROUGE/BLEU **không đo trung thực** chất lượng chunk. Vì vậy đừng đánh giá phần chunk bằng
ROUGE — hãy đánh giá **trực tiếp bằng TEDS** (cấu trúc) và **`Recall@k`** (truy xuất).

---

## 14. Danh mục tài liệu tham khảo (gộp)

Chỉ liệt kê những mục **được trích dẫn trong tài liệu này**. Danh mục đầy đủ nằm ở
`docs/Tai_lieu_tham_khao.md`.

### Mức A — đã kiểm chứng trong `data/references/`

1. Peng, Q., Luo, X., et al. (2027). *Dual-granularity chunking and dynamic context augmentation.* Expert Systems with Applications, 332, 133546.
2. Cheng, H., Wang, K., Wang, Q., Liu, T., Sheng, K. *A Chinese financial event knowledge graph-based retrieval-augmented generation framework for financial question answering* (CFinRAG). Engineering Applications of Artificial Intelligence, 175, 114670.
3. Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS 33, 9459–9474.
4. Liu, Z., Simon, C., Caspani, F. (2025). *Document paragraph segmentation for extraction-based QA.* arXiv:2501.09940.
5. Karpukhin, V., et al. (2020). *Dense passage retrieval for open-domain question answering.* EMNLP 2020.
6. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* EMNLP-IJCNLP, 3982–3992.
7. Zhao, H., Liu, Z., Wu, Z., et al. (2024). *Revolutionizing finance with LLMs: An overview of applications and insights.* arXiv:2401.11641.
8. Es, S., James, J., Anke, L. E., Schockaert, S. (2024). *RAGAS: Automated evaluation of retrieval augmented generation.* EACL System Demonstrations, 150–158.
9. Salemi, A., & Zamani, H. (2024). *Evaluating retrieval quality in retrieval-augmented generation.* SIGIR '24, 2395–2400.

### Mức B — cần tra trước khi trích

10. Robertson, S., & Zaragoza, H. (2009). *The probabilistic relevance framework: BM25 and beyond.* Foundations and Trends in Information Retrieval, 3(4), 333–389.
11. Robertson, S., & Walker, S. (1994). *Some re-weighting and expansion approaches for text retrieval.* SIGIR '94.
12. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR '09.
13. Wang, L., Yang, N., Huang, X., et al. (2024). *Multilingual E5 text embeddings.* arXiv:2402.05672.
14. Nguyen, T., et al. (2020). *XLM-R: Cross-lingual language model pretraining at scale.* arXiv:1911.02116.
15. O'Gorman, L., & Wong, K. Y. (1986). *Page segmentation and structure deduction for image processing.* IEEE TPAMI, 8(3), 196–203.
16. Shur, A., et al. (2014). *XPSplit: A cross-platform, end-user extendable table data extraction library.* SC '14.
17. McKinney, M. (2017). *Camelot: Python library to extract tables from PDFs.*
18. Poddar, S., et al. (2022). *FinTabNet: Towards improved table understanding from financial PDFs.* WACV 2022.
19. Zhong, X., et al. (2020). *PubTabNet: Largest realistic dataset for table structure recognition.* ECCV 2020.
20. Li, Y., et al. (2018). *TableBank: table extraction and understanding.* NeurIPS 2018.
21. Li, Y., et al. (2019). *TEDS: a tree-edit-based similarity for evaluating table structure recognition.* ACL 2019.
22. Lysak, N., et al. (2023). *TableFormer: table structure understanding with transformers.* ICDAR 2023.
23. Lee, K., et al. (2022). *Donut: document understanding transformer.* ECCV 2022.
24. Sarthi, R., et al. (2024). *RAPTOR: Recursive abstractive processing for tree-organized retrieval.* ICLR 2024.
25. Chen, W., et al. (2020). *TAT-QA: a hybrid approach to real-world table and text question answering.* ACL 2020.
26. Zhou, T., et al. (2021). *FinQA: a dataset of numerical reasoning over financial data.* EMNLP 2021.
27. Nguyễn Thành Hưng, Nguyễn Đức Hoàng Linh (2018). *VnCoreNLP: a Vietnamese NLP toolkit.*
28. Smith, M. (2007). *An overview of the Tesseract OCR engine.* ICDAR.

### Mức C — đóng góp của dự án, không có nguồn ngoài

Xem bảng ở mục 11.1. Tóm tắt 16 mục: dò bảng bằng mật độ số; nới vùng có ràng buộc lưới cột;
tiêu chí ổn định lưới; loại văn xuôi trôi vào header; loại ô trộn bằng điều kiện kép; tách
"STT/Mã CP"; dòng tiêu đề nhóm làm ranh giới; ưu tiên khe dọc hơn tiêu đề nhóm; ghép bảng qua
trang bằng chữ ký cột; nhận diện cột khóa theo dữ liệu; đơn vị theo từng giá trị; caption
bằng regex; chẩn đoán OCR; metadata tài liệu + lọc cứng; biến thể số kép; token hoá tiếng Việt
giữ nguyên cụm.

---

## 15. Danh sách kiểm chứng trước khi viết luận văn

| # | Việc | Vì sao |
|---|---|---|
| 1 | Tra **mọi mục B** ở mục 14 và điền số trang/DOI thật | Tôi không tra được mạng |
| 2 | Sửa tác giả bài RRF nếu `Buettcher, M.` không đúng | Xem cảnh báo mục 0 |
| 3 | Xác minh XPSplit: vị trí tác giả `Ameri`/`Premkupar`/`Buttener` | Mục 3.3 |
| 4 | Tra **"filtered vector search pre-filtering recall"** | Mục 9.3 còn nợ một nguồn |
| 5 | Kiểm chứng số liệu FinTabNet "1,52% trang chứa bảng" | Mục 5.3, dùng làm luận điểm |
| 6 | Tải FinTabNet / PubTabNet về `data/references/` | Nếu muốn so sánh định lượng |
| 7 | Dựng bộ câu hỏi vàng và đo `Recall@k` + TEDS | Mục 13 — ưu tiên cao nhất |

---

## 16. Các điểm mù đã đo — ưu tiên công việc tiếp theo

Đo trên PDF tổng hợp, không phải suy đoán:

| Đặc trưng | Kết quả | Nguyên nhân | Đề xuất |
|---|---|---|---|
| Bảng 2 cột **văn bản thuần**, 0 số | **0 bảng** → rơi thành prose rác | `_find_regions` đòi `≥3 ô số` | Thêm điều kiện vào dựa trên cấu trúc (mục 4.3) |
| Bảng **xoay 90°** | **0 bảng**, mỗi từ thành một dòng; `char_w` bị bóp méo | `build_lines` chỉ gom theo `y` | Dùng `line["dir"]` từ `get_text("dict")` — đã kiểm chứng cho `(0.0,-1.0)` khi xoay, `(1.0,0.0)` khi thẳng |
| Văn xuôi 2 cột | Thứ tự đọc sai → BM25 nhiễu | `_prose_blocks` nối theo `y` | Tách dải cột bằng `_empty_runs` |
| Vùng nghi là bảng bị loại | **Không có dấu vết** | `_is_plausible` từ chối rồi bỏ | Ghi `warnings` theo trang kèm lý do |
| Bảng có viền | Bỏ qua | Không dùng `find_tables()` | Dùng `strategy="lines_strict"` làm ý kiến thứ hai |

Chi tiết thiết kế và ước lượng công sức nằm ở `docs/Ky_thuat_da_su_dung.md` mục 15.