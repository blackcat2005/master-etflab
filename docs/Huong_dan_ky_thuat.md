# Hướng dẫn kỹ thuật — code `etflab` đang chạy thế nào

> **Cập nhật 2026-10:** parser bảng bằng hình học (`geometry.py`, `tables.py`,
> `pipeline.py`, `chunking.py`) đã được **lưu trữ** tại
> `archive/geometric_parser/` và không còn là pipeline chính. Pipeline hiện tại
> dùng `unstructured` (`hi_res` + Table Transformer): xem `etflab/uelements.py`,
> `etflab/chunking_methods.py` và `scripts/ingest/parse_unstructured.py`.
> Các mục dưới đây mô tả parser cũ để tra cứu lịch sử; đọc kèm
> `archive/geometric_parser/README.md`.

Tài liệu này ghi lại **kỹ thuật đã cài đặt thật trong code**, kèm vị trí cụ thể để tra cứu được,
và ghi rõ hạn chế đã biết của từng kỹ thuật. Trả lời câu hỏi **"code này làm gì, ở dòng nào"**.

> **Phân biệt với [`Tham_khao_va_nguon.md`](./Tham_khao_va_nguon.md):** file đó trả lời
> **"việc này dựa trên bài báo nào, mức tin cậy ra sao"** — dùng khi viết phần "phương pháp"
> trong luận văn.

> Phạm vi: bảng tài chính trong PDF tiếng Việt, **không có đường viền** (borderless).
> Corpus mục tiêu: báo cáo phân tích ETF, bản tin ETF, tài liệu quy chế.

---

## 0. Kiến trúc tổng thể

```
PDF (không viền)
  │
  ├─ T1  word → dòng (build_lines)              geometry.py:154
  ├─ T2  dòng → ô (group_cells)                 geometry.py:173
  ├─ T3  phát hiện thung lũng khoảng trắng → cột geometry.py:252
  ├─ T4  siết cột (tighten_columns)             geometry.py:294
  ├─ T5  gán ô vào cột (assign_to_columns)      geometry.py:314
  │
  ├─ T6  tìm vùng bảng (mật độ số + độ phủ lưới) tables.py:122
  ├─ T7  giữ dòng số lẻ (_grow_run)             tables.py:153
  ├─ T8  ổn định lưới (_grid_is_stable)          tables.py:198
  ├─ T9  header nhiều tầng (_header_levels)      tables.py:572
  ├─ T10 caption Bảng N / Table N                pipeline.py:61
  ├─ T11 tách bảng cạnh nhau (_split_slab)       tables.py:287
  ├─ T11b đọc tiêu đề nhóm khi khe quá hẹp      tables.py:336
  ├─ T12 gộp bảng nối trang (_merge_continuations) pipeline.py:170
  ├─ T13 phát hiện PDF scan / cảnh báo OCR       pipeline.py:112
  │
  ├─ T14 chuẩn hóa số Việt Nam (parse_number)    vnnum.py:80
  │
  ├─ T15 chunk hai cấp: table_card / table_row / prose   chunking.py:339
  ├─ T16 nhận diện cột nhãn theo dữ liệu        chunking.py:233
  ├─ T17 metadata tài liệu (loại, chỉ số, kỳ)     docmeta.py:144
  │
  ├─ T18 BM25 tự hiện thực                       retrieval.py:237
  ├─ T19 tokenizer tài chính, chuẩn hóa số       retrieval.py:100
  ├─ T20 metadata cấp tài liệu trong chunk       chunking.py:339
  ├─ T21 dense retrieval (opt-in)                retrieval.py:349
  ├─ T22 hợp nhất RRF                            retrieval.py:428
  ├─ T23 lọc cứng theo kỳ / chỉ số               retrieval.py:386
  └─ T24 mở rộng parent (parent expansion)       retrieval.py:445
```

---

## 1. Tách lớp trích xuất: word → dòng → ô → cột

**Kỹ thuật.** Không dùng `find_tables()` của PyMuPDF. Thay vào đó tự dựng lại bảng từ
toạ độ hộp chữ (`Word`), vì bảng mục tiêu không có đường kẻ.

| Bước | Hàm | Nguyên tắc |
|---|---|---|
| Gom dòng | `build_lines` (`geometry.py:154`) | Chồng lấn theo trục **y**; dung sai `y_tol=3.0`. Đây là cách ổn định hơn so với `get_text("blocks")` vì bảng tài chính có nhiều cột thẳng hàng. |
| Đo bề rộng ký tự | `median_char_width` (`geometry.py:144`) | Dùng **trung vị** thay vì trung bình để chịu được vài ký tự lạ trong footer. |
| Gom ô | `group_cells` (`geometry.py:173`) | Khoảng trắng nội tuyến **≥ 1,5 × bề rộng ký tự** ⇒ chèn khoảng trắng. Ngưỡng này là tham số quan trọng nhất khi chuyển sang corpus khác. |

**Vì sao không dùng `find_tables`.** Bảng trong báo cáo SSI ETF không có đường viền; chạy thử
`pymupdf` với chiến lược dòng kẻ cho kết quả rỗng hoặc sai lệch hàng. Quyết định này được
ghi lại trong `README.md` để tránh người sau vô tình bật lại.

**Hạn chế.** Một ô chứa hai từ sát nhau không tách được; bảng có ô hợp nhất theo chiều ngang
(merged cell) chưa được xử lý — hiện nội dung ô bị gộp vào một cột.

---

## 2. Phát hiện ranh giới cột bằng thung lũng khoảng trắng

**Kỹ thuật.** `column_valleys` (`geometry.py:252`) dựng histogram phủ x của các từ trong
vùng bảng, tìm các **valley** (khoảng trắng đủ rộng, không có từ nào lọt qua), rồi
`tighten_columns` (`geometry.py:294`) siết biên về mép thật của chữ.

Đây đúng là ý tưởng **stream mode** của Camelot, chỉ khác ở chỗ Camelot coi số là ranh giới
và bản này dùng chuỗi **thung lũng khoảng trắng liên tiếp** để ổn định hơn khi bảng có
ô rỗng.

**`allow_sparse`.** Vùng đầu bảng header thường có ít dữ liệu số; tham số `allow_sparse`
cho phép thung lũng được chấp nhận dựa trên số dòng mà nó xuất hiện, thay vì số từ.

**Hạn chế.** Bảng có cột cách nhau rất hẹp (ví dụ hai cột phần trăm) có thể bị gộp làm một.
Nếu gặp, hạ `char_w` hoặc thêm ranh giới từ chính header.

---

## 3. Phát hiện vùng bảng

**Kỹ thuật.** `_find_regions` (`tables.py:122`) quét dọc trang, tích luỹ các dòng, và đánh
dấu vùng bảng khi đồng thời thỏa:

1. `numeric_density` (`geometry.py:96`) đủ cao — bảng tài chính phần lớn ô là số;
2. `_grid_is_stable` (`tables.py:198`) — phần lớn dòng khớp được **cùng một bộ cột**.

Điều kiện (2) là điểm mấu chốt: một khối văn bản có thể chứa nhiều số (mã chứng khoán,
ngày) nhưng các dòng không cùng chia sẻ một lưới cột, nên bị loại.

**So sánh với `_grid_is_stable` cũ.** Phiên bản đầu dùng điều kiện *mọi dòng đều phải khớp
đủ số cột*, và vì thế loại mất dòng `BSR` (dòng đầu bảng, cột đầu trống). Bản sửa chỉ yêu
cầu **phần lớn** dòng khớp (`tables.py:198`).

---

## 4. Giữ dòng số lẻ có ô rỗng — `_grow_run`

**Kỹ thuật.** Khi dòng trước và dòng sau đều là dòng số, nhưng dòng giữa không parse được
số nào (ví dụ dòng tiêu đề con `BSR` nằm giữa hai dòng dữ liệu), `_grow_run`
(`tables.py:153`) nới điều kiện "không được là dòng số" cho **đúng vị trí đó**.

**Vì sao cần.** Dòng `BSR` trong báo cáo SSI có một số ở giữa và ô đầu trống. Bộ lọc
"mật độ số" thuần sẽ cắt nó khỏi vùng bảng, và `BSR` mất vai trò cột định danh — nên
câu hỏi "BSR của BSR là bao nhiêu" không còn dữ liệu để trả lời.

**Hạn chế.** Nới điều kiện ở vị trí lẻ chỉ an toàn khi lân cận đã là dòng số; một khối văn
bản 3 dòng không số sẽ không bị nới.

---

## 5. Header nhiều tầng

**Kỹ thuật.** `_header_levels` (`tables.py:572`) gom các dòng header thành nhiều tầng; mỗi
tầng được khớp vào cột qua `_match_column` (`tables.py:466`). `_segment_header_cells`
(`tables.py:556`) tách ô của dòng header khi một nhãn trải nhiều cột.

`header_paths` (`chunking.py:90`) ghép tầng thành đường dẫn đầy đủ, ví dụ
`Thay đổi / Giá trị`.

**Vì sao quan trọng cho RAG.** Cột trong bảng tài chính **không có nghĩa nếu thiếu header**.
Đường dẫn header được nhúng vào mọi row chunk, nên BM25 không còn nhất thiết phải khớp
tên cột một cách tình cờ.

---

## 6. Caption và loại running header/footer

**Kỹ thuật.** `_find_caption` (`pipeline.py:61`) dò phía trên vùng bảng theo mẫu
`Bảng N` / `Table N`. `_restrict` (`tables.py:397`) cắt bỏ running header/footer theo vị trí
và tần suất lặp.

---

## 7. Bảng cạnh nhau trên cùng trang

**Kỹ thuật.** `_split_slab` (`tables.py:287`) phát hiện khoảng trắng ngang lớn giữa hai khối
và tách thành hai bảng độc lập, thay vì gộp thành một bảng nhiợt.

**Khi khe trắng không đáng tin.** Bảng tin Q1/2026 có khe giữa hai bảng chỉ 12pt, *nhỏ hơn*
khe giữa hai cột của cùng một bảng (16-70pt), nên `vertical_bands` không tách được. Dấu hiệu
đáng tin là **tiêu đề nhóm** (`Kịch bản 1: ...` / `Kịch bản 2: ...`) trong header, mỗi ô phủ
nhiều cột dữ liệu. `_group_title_bands` (`tables.py:336`) chỉ đọc dòng này khi
`vertical_bands` thất bại, và chỉ nhận dòng mà các ô **khác nhau về chữ** - header nhiều tầng
thật (`Số lượng CP mua/bán` lặp hai bên) lặp cùng một nhãn nên bị loại.

---

## 8. Gộp bảng nối trang

**Kỹ thuật.** `_merge_continuations` (`pipeline.py:170`) + `_continuation_target`
(`pipeline.py:190`) nối bảng bị cắt đôi giữa hai trang. Điều kiện nối:

- chữ ký cột khớp (`Table.signature`, `tables.py:67`);
- khoảng cách ngang `x_span` khớp (tránh nối nhầm hai bảng khác nhau).

Đây là vấn đề "stitching" của nhánh *complex table recognition* — trường hợp báo cáo SSI
và bản tin ETF đều có bảng trải trên 2 trang.

**Hạn chế.** Chữ ký cột phải khớp gần như tuyệt đối; bảng nối trang có thêm/xoá một cột
sẽ không được gộp. Chưa hỗ trợ header lặp lại ở trang sau (phần header trang 2 sẽ thành
một dòng dữ liệu).

---

## 9. Phát hiện PDF scan

**Kỹ thuật.** `process_pdf` (`pipeline.py:112`) đo số khối chữ trên trang; trang gần như
không có chữ ⇒ thêm cảnh báo vào `DocumentResult.warnings` thay vì trả bảng rỗng âm thầm.

**Trạng thái.** Chưa tích hợp OCR. 4/8 file trong corpus hiện phát cảnh báo. Cần OCR
(định tuyến: Tesseract hoặc model layout-aware) ở phase kế tiếp.

---

## 10. Chuẩn hóa số Việt Nam

**Kỹ thuật.** `parse_number` (`vnnum.py:80`) trả về `Num(value, unit)`, không phải `float`.

Quy tắc đã kiểm chứng bằng test:

| Token | Kết quả | Ghi chú |
|---|---|---|
| `2.976,9` | `2976.9` | dấu chấm là phân tách nghìn |
| `(69.559)` | `-69559.0` | ngoặc = số âm kế toán |
| `-1,3` | `-1.3` | |
| `0,87%` | `Num(value=0.87, unit="percent")` | **không** nhân thêm 100 |
| `30/06/2026` | không phải số | `_is_date` loại trước |
| `1.234,5%` | `Num(1234.5, "percent")` | giữ nguyên độ lớn |

**Quyết định thiết kế quan trọng.** Không nhân `100` cho phần trăm. Báo cáo tài chính Việt
Nam ghi `0,87%` nghĩa là 0,87 phần trăm theo nghĩa thường; nhân thêm sẽ sai thêm 100 lần.
Vì vậy phần trăm là **unit flag**, không phải phép tính.

**Bug đã sửa.** `vnnum.py:80` có hai nhánh `if` giống hệt nhau khiến tham số
`dot_as_thousands=False` bị vô hiệu hóa âm thầm.

---

## 11. Chunk hai cấp + quan hệ cha–con

**Kỹ thuật.** `build_chunks` (`chunking.py:339`) sinh ba loại chunk:

| Loại | Hàm | Dùng để |
|---|---|---|
| `table_card` | `chunking.py:268` | mô tả bảng: caption, header, đơn vị, số dòng |
| `table_row` | `chunking.py:172` | trả lời câu hỏi về **một dòng** |
| `prose` | `chunking.py:293` | văn bản ngoài bảng |

Mỗi `table_row` có `text` (đầy đủ, dễ đọc cho LLM) và `sparse_text` (chỉ giá trị, dùng để
embedding và so khớp), cùng `cells`, `row_index`, `parent_id` trỏ về `table_card`.

**Nguyên tắc.** Chunk nhỏ để **truy xuất chính xác** (đúng dòng), chunk lớn để **sinh câu
trả lời đủ ngữ cảnh**. Đây là cùng một ý tưởng với dual-granularity chunking, chỉ khác ở
chỗ đơn vị nhỏ là **một dòng bảng** chứ không phải một câu văn.

**Đơn vị.** `_cell_unit` (`chunking.py:151`) và `_display_units` (`chunking.py:141`) giữ
đơn vị ở mức ô, **không** gán đơn vị cấp bảng cho mọi ô — lỗi này từng làm cột phiên xu
không liên quan bị gắn "Nghìn cổ phiếu".

**Round-trip JSON.** `Table.rows` luôn dùng khóa chuỗi (`"0"`, `"1"`, …) ở cả RAM và JSON,
nếu không JSON sẽ biến `"0"` thành `0` và test round-trip sẽ hỏng — đây là ràng buộc bắt
buộc cho phase PoT/SQL sau này.

---

## 12. Nhận diện cột nhãn theo dữ liệu

**Kỹ thuật.** `_label_columns` (`chunking.py:233`) đo **tỉ lệ ô parse được thành số** theo
từng cột; cột nhãn là cột có tỉ lệ số thấp nhất.

**Vì sao không đếm cứng.** Bản đầu dùng ngưỡng "cột đầu tiên không phải STT là cột nhãn".
Khi bảng **không có cột STT**, mã chứng khoán (ví dụ `MCH`) nằm ở cột đầu và bị loại khỏi
vùng nhúng ⇒ BM25 không match được tên cổ phiếu, dù `sparse_text` có nó.

---

## 13. BM25 tự hiện thực

**Kỹ thuật.** `BM25` (`retrieval.py:237`) hiện thực đúng công thức Okapi BM25 với
`k1=1.5`, `b=0.75`.

**Vì sao tự viết.** Không kéo thêm dependency cho thư viện IR (rank-bm25 chỉ hỗ trợ tiếng
Anh, không strip dấu tiếng Việt). Khoảng 40 dòng, dễ kiểm thử bằng tay.

**Hạn chế.** Không có stemming, không có trường có trọng số riêng (ví dụ ưu tiên tên cổ
phiếu và caption).

---

## 14. Tokenizer tài chính

**Kỹ thuật.** `tokenize` (`retrieval.py:100`) + `_number_tokens` (`retrieval.py:63`):

- `_fold` (`retrieval.py:58`) bỏ dấu và đổi `đ → d`, để "cổ phiếu" và "co phieu" là một token;
- số được chuẩn hóa về **một dạng duy nhất** (`2.976,9` → `2976.9`), nên truy vấn "2.976,9"
  khớp được giá trị `2976.9` trong bảng;
- token dạng số sinh thêm biến thể không phân tách để hỗ trợ so khớp từng chữ số.

**Bug đã sửa.** Stopword tiếng Việt ban đầu có `"trong"`. Chuẩn hóa bỏ dấu biến `trong`
thành token rỗng ở các từ chứa nó — làm mất chữ **"trọng"** trong "trọng vốn", "trọng số",
"trọng tâm". Đã gỡ khỏi danh sách.

---

## 15. Metadata tài liệu + lọc theo kỳ

**Vấn đề.** Câu hỏi *"Tỷ trọng mới của BSR **Q2**"* trả về bản tin **Q3**.

Chunk chỉ chứa caption + dữ liệu dòng. Kỳ báo cáo nằm trong **tên file**, nên không token
`q2` nào tồn tại để so sánh — BM25 chỉ còn `bsr` + `ty trong moi` và xếp hạng bản Q3 lên
trên. Nhét kỳ vào `sparse_text` **không đủ**: `q2` là *ràng buộc*, không phải *sở thích*.
Sau khi gắn nhãn, `q2` xuất hiện ở 223 chunk nên IDF tụt từ 5.4 xuống 1.2, trong khi `bsr`
vẫn ở 4.1 (12 chunk) — tín hiệu mạnh hơn vẫn thắng.

**Kỹ thuật.** `etflab/docmeta.py` suy ra metadata cấp tài liệu:

| Trường | Nguồn | Ví dụ |
|---|---|---|
| `doc_type` | tiền tố tên file | `review`, `forecast`, `charter` |
| `index` | tên chỉ số | `VN30`, `VNDIAMOND`, `VNFINLEAD` |
| `quarter` / `year` | `Q2_2026`, `kỳ Q2/2026`, `quý II/2026` | `Q2` / `2026` |
| `date` | tiền tố `20260508`, hoặc `08.05.2026` | `2026-05-08` |
| `label` | tổng hợp, dạng đọc được | `Bản tin thay đổi thành phần chỉ số VN30 kỳ Q2/2026` |

Mọi regex chạy trên chuỗi **đã bỏ dấu** (`_fold`), vì `quý`/`Kỳ`/`Điều` không khớp pattern
viết bằng ký tự ASCII. `label` giữ nguyên dạng `Q2/2026` để token hoá thành `q2` — viết
"quý 2" sẽ chỉ ra `quy` + `2` và không khớp được.

`build_chunks` gắn `doc` vào mọi chunk, tiền tố nhãn vào `sparse_text` (tín hiệu mềm, vẫn
giúp khi người dùng không lọc) và đưa vào `metadata` để LLM/reranker dùng.

**Lọc cứng.** `DocFilter` + `infer_doc_filter` (`retrieval.py`) suy ra ràng buộc từ câu
hỏi — chỉ nhận dạng ký rõ ràng (`Q1`–`Q4`, `quý II`, năm 4 số, tên chỉ số), không suy
diễn, nên `2.976,9` không bị hiểu là "quý 9". `_allowed` loại chunk khỏi *chỉ mục* trước khi
tính điểm:

```python
index.search("Tỷ trọng mới của BSR Q2", doc_filter=infer_doc_filter("Tỷ trọng mới của BSR Q2"))
```

Câu hỏi không nêu kỳ thì không lọc — `tỷ trọng mới của BSR` vẫn trả cả Q2 lẫn Q3, xếp
hạng theo BM25.

---

## 16. Dense retrieval (opt-in) + RRF

**Kỹ thuật.**

- `intfloat/multilingual-e5-small` — hỗ trợ tiếng Việt, chạy CPU được. Cosine similarity
  (`retrieval.py:377`).
- `RRF` (`retrieval.py:428`): `score = Σ 1/(k + rank)`, `k = 60`. Hợp nhất BM25 và dense mà
  **không cần chuẩn hóa thang điểm** — điểm mấu chốt khi BM25 (vô hạn) và cosine (∈ [-1,1])
  đứng chung thang đo.
- Parent expansion: sau khi hợp nhất, mở rộng `table_row` về `table_card` để LLM có header
  và đơn vị.

**Quyết định vận hành.** `RetrievalConfig.use_dense=False` (`retrieval.py:243`) mặc định.
Bật lên sẽ tải model ~500 MB, làm chậm lần chạy đầu — không nên ép vào đường chạy mặc định.

---

## 17. Đánh giá

`tests/` gồm **187 test, tất cả đều pass** (`python -m pytest`). Golden test trên
toàn bộ 7 PDF có text layer; 4 điều lệ scan chỉ kiểm tra cảnh báo OCR.

| File | Kỳ vọng |
|---|---|
| `ssi_review_announcements/Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf` | 1 bảng, trang `[1, 2]`, `31×7`, dòng đầu `BSR`, dòng cuối `DGC` |
| `ssi_review_announcements/Thay_doi_thanh_phan_chi_so_VN30_Q3_2026.pdf` | 2 bảng cạnh nhau, trang `[2]`, `32×7` và `32×4` |
| `ssi_review_announcements/Thay_doi_thanh_phan_chi_so_VNDiamond_VN30_VNFIN_Q1_2026.pdf` | 5 bảng, `31×8 / 18×7 / 18×3 / 23×5 / 46×8`, trang `[3] [4] [4] [4] [5]` |
| `ssi_review_announcements/Thay_doi_thanh_phan_chi_so_VNDiamond_VN30_VNFIN_Q2_2026.pdf` | 3 bảng, `18×5 / 18×4 / 45×7`, trang `[2] [2] [3]` |
| `broker_forecast_reports/Du_bao_thay_doi_danh_muc_chi_so_VN30_VNFINLEAD_Q3_2026.pdf` | 2 bảng, trang `[2, 3]`, `41×6` và `41×5` |
| `broker_forecast_reports/Du_bao_thay_doi_danh_muc_chi_so_VN30_VNFINLEAD_Q1_2026.pdf` | 2 bảng cạnh nhau, trang `[3]`, `40×6` và `40×5`, dòng đầu `VPB` |
| `broker_forecast_reports/Du_bao_thay_doi_danh_muc_chi_so_VNDiamond_Q2_2026.pdf` | 1 bảng, trang `[2]`, `18×8`, dòng đầu `FPT`, dòng cuối `KDH` |

**Hồi quy toàn corpus** (`tests/test_corpus.py`): mọi PDF có text đều ra ít nhất 1 bảng;
tên cột không được chứa văn xuô (`Cần lưu ý`, `Lưu ý:`, `Kịch bản`, `Bảng:`, ngày tháng);
khoá dòng luôn là `str`; mọi ô của cột số phải parse được hoặc là `-`.

**Không còn lỗi đã biết.** Ba lỗi từng khoá bằng `xfail(strict=True)` đã được sửa và
chuyển thành test thật:

| Lỗi đã sửa | Triệu chứng cũ | Cách sửa |
|---|---|---|
| Chú thích ngày tháng lẫn vào tên cột | `22/04/2026. STT`, `dữ liệu đến ngày 21/1/2026 Mã CP` | `_is_prose_residue` cắt dòng 1 ô lệch trái ra ngoài mép bảng |
| Bảng tin Q1 chưa tách đôi | 1 bảng `40×11` thay vì 2 bảng | `_group_title_bands` đọc dòng tiêu đề nhóm `Kịch bản 1/2` |
| `STT`/`Mã CP` bị gộp thành một ô | cột 0 và 1 cùng là `STT Mã CP` | `_text_in_column` chỉ tách khi mỗi nhóm từ nằm trọn trong một cột |

Hai điểm kỹ thuật đáng lưu ý khi tách bảng cạnh nhau:

- **Khe trắng không đáng tin.** Bảng tin Q1/2026 có khe giữa hai bảng chỉ 12pt, *nhỏ hơn*
  khe giữa hai cột của cùng một bảng (16–70pt). `vertical_bands` vì thế không tách được;
  dấu hiệu đáng tin là **tiêu đề nhóm** (`Kịch bản 1: ...` / `Kịch bản 2: ...`), mỗi ô
  phủ nhiều cột dữ liệu. Vì vậy `_group_title_bands` chỉ chạy khi `vertical_bands`
  thất bại, và chỉ chấp nhận dòng mà các ô **khác nhau về chữ** — header nhiều tầng thật
  (`Số lượng CP mua/bán` lặp hai bên) lặp lại cùng một nhãn nên bị loại.
- **Tiêu đề nhóm là tên bảng, không phải tên cột.** Sau khi tách, dòng đó được đánh dấu
  theo toạ độ `y` và bỏ qua khi dựng tên cột, nhưng vẫn giữ trong `header_lines` để
  logic nối trang vẫn so khớp được header giữa các trang.

**Còn thiếu.** Chưa có bộ câu hỏi vàng và chưa đo `Recall@k` / `MRR` / PoT accuracy. Đây là
việc cần làm trước khi tối ưu thêm bất kỳ thứ gì.

---

## 18. Tổng hợp hạn chế đã biết

| Hạn chế | Vị trí liên quan | Hướng xử lý |
|---|---|---|
| Chưa có OCR | `pipeline.py:112` | Tesseract hoặc model layout-aware |
| Ô hợp nhất theo chiều ngang | `tables.py` (`_header_levels`) | Cần `colspan` thật sự |
| Cột cách nhau hẹp bị gộp | `geometry.py:252` | Tinh chỉnh `char_w` |
| Header lặp ở trang nối thành dòng dữ liệu | `pipeline.py:190` | Lọc theo chữ ký header |
| Không stemming / không trường có trọng số | `retrieval.py:237` | BM25+ / trường riêng cho tên cổ phiếu |
| Chưa đo Recall@K | `tests/` | Bộ câu hỏi vàng |
| Bảng tiếng Anh cho tên cột kém | `tables.py` (`_column_names`) | Bộ rule nhận diện header theo ngôn ngữ |

Ghi chú về dòng `_header_levels` (`tables.py:572`): ô bị loại khỏi tên cột khi nó phủ
**≥ 3 cột và đồng thời > nửa bảng**, hoặc khi nó thuộc dòng tiêu đề nhóm đã dùng để tách
bảng. Ô gộp 2 cột hợp lệ vẫn được giữ. Trước đây dùng ngưỡng "rộng ≥ 45% bảng" — cách đó
giết nhầm tiêu đề nhóm hợp lệ trong bảng tin Q1 và làm mất `unit_notes`.