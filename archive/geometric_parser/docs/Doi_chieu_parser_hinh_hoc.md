# Đối chiếu kỹ thuật ↔ nguồn: parser hình học cũ (lưu trữ)

> Tách ra từ `docs/04_tham_khao/Tham_khao_va_nguon.md` các mục 3, 4, 5.1–5.3, 6, 7. Không còn thuộc pipeline chính; giữ để tra cứu.

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

> **Phát hiện đáng lưu ý về điểm mù (đã đo, xem `archive/geometric_parser/docs/Huong_dan_ky_thuat.md` mục 15):** vì cổng
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
| **Cấu trúc cây header nhiều tầng** | **B** | Li, M., Cui, L., Huang, S., Wei, F., Zhou, M., & Li, Z. (2020). *TableBank: A Benchmark Dataset for Table Detection and Recognition.* LREC 2020, 1918–1925 (arXiv:1903.01949). — biểu diễn bảng thành chuỗi thẻ HTML để mã hoá quan hệ cha–con giữa các ô header. Cách dùng header phân tầng như cây của chúng tôi là **cùng cấu trúc dữ liệu**, chỉ khác ở việc suy ra từ hình học thay vì học. ⚠️ *Bản "NeurIPS 2018 / Li, Y." từng ghi ở đây là **sai** — không tồn tại; đã tra cả ACL Anthology lẫn GitHub chính thức* |
| **Chỉ số đo TEDS** đánh giá cấu trúc | **B** | Cùng bài với PubTabNet: Zhong, X., ShafieiBavani, E., & Jimeno-Yepes, A. (2020) — TEDS **được đề xuất trong chính bài này**, không phải một bài riêng. Nên dùng làm chỉ số đánh giá nếu dựng được ground truth cấu trúc |
| Bảng tài chính khó hơn bảng thường | **B** | FinTabNet — báo cáo thường niên **S&P 500**, ảnh PDF trang đơn, có bbox từng ô (khác PubTabNet là không kèm PDF gốc). **Sát nhất với corpus dự án.** Xem mục 5.4 |
| Bộ dữ liệu chuẩn để đối chiếu | **B** | Zhong, X., ShafieiBavani, E., & Jimeno-Yepes, A. (2020). *Image-based table recognition: data, model, and evaluation.* ECCV 2020, 564–580 (arXiv:1911.10683). ⚠️ *Tiêu đề "PubTabNet: Largest realistic dataset…" từng ghi ở đây là **sai** — đó không phải tên bài báo* |
| Hướng thay thế bằng mô hình | **B** | Nassar, A., Livathinos, N., Lysak, M., & Staar, P. (2022). *TableFormer: Table structure understanding with transformers.* CVPR 2022, 4614–4623. Dùng trong IBM Docling — **hướng thay thế cho `tables.py`**, không phải bổ sung. ⚠️ *Tác giả đầu là Nassar, không phải Lysak; venue là CVPR 2022, không phải ICDAR 2023* |


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
