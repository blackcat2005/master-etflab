# Đối chiếu kỹ thuật với nguồn học thuật

> **Cập nhật 2026-10:** phần parser bảng bằng hình học trong tài liệu này đã
> được lưu trữ (`archive/geometric_parser/`). Pipeline đánh giá hiện tại dùng
> `unstructured` + Table Transformer; các đối chiếu học thuật về chunking,
> retrieval và ColPali vẫn giữ nguyên giá trị.

Ngày lập: 2026-10-04. Phạm vi: **toàn bộ kỹ thuật đã hiện thực** trong `etflab/`, kèm nguồn
học thuật của từng kỹ thuật và mức độ tin cậy của nguồn đó.

Tài liệu này gộp cả bảng đối chiếu kỹ thuật lẫn danh mục tham khảo — là nguồn tham khảo
duy nhất, không còn file danh mục riêng. Trả lời câu hỏi **"việc này dựa trên bài báo nào,
mức độ tin cậy ra sao"**. Dùng khi viết phần "phương pháp" trong luận văn.

> **Phân biệt với [`Huong_dan_ky_thuat.md`](../archive/geometric_parser/docs/Huong_dan_ky_thuat.md):** file đó mô tả **code
> đang chạy thế nào**, dùng khi sửa code.

| File | Vai trò |
|---|---|
| `archive/geometric_parser/docs/Huong_dan_ky_thuat.md` | Mô tả cách **parser hình học cũ** chạy (đã lưu trữ) |
| **`docs/Tham_khao_va_nguon.md` (file này)** | **Bảng đối chiếu kỹ thuật ↔ nguồn gốc** + **danh mục tham khảo**, để viết phần "phương pháp" trong luận văn |

**Cấu trúc file:**

| Phần | Mục | Nội dung |
|---|---|---|
| **I** | 1–16 | Kỹ thuật **đã hiện thực**, đối chiếu nguồn, danh mục chuẩn (mục 14), hạng mục việc (14.1), điểm mù đã đo (16) |
| **II** | 17–26 | Danh mục mở rộng: các hướng **chưa làm** (PoT, đánh giá, đồ thị, huấn luyện LLM) |

> ⚠️ **Danh mục chuẩn là mục 14.** Khi viết phần "phương pháp" chỉ trích mục ở mục 14 — phần II
> chỉ để biết hướng nào đã có người làm.

---

## 0. Quy ước mức độ tin cậy của nguồn

| Mức | Ý nghĩa | Hành động cần làm |
|---|---|---|
| **A** | Tên/đề bài lấy nguyên văn từ **PDF hoặc trang nhà xuất bản** đã mở và đọc | Trích được |
| **B** | Từ kiến thức chung. **Tác giả/năm/đề bài khá chắc, số trang và DOI chưa kiểm chứng** | **Tra arXiv / ACM DL / ACL Anthology trước khi trích** |
| **C** | Không có bài báo. Đây là **quyết định thiết kế riêng** của dự án | Trích bằng cách **mô tả phương pháp của bạn**, không được gán cho nguồn ngoài |

Một mục chuyển từ **B → A** sau khi đọc PDF gốc: **Cormack et al. 2009** (mục 14.12). Tác giả
thứ ba là **Stefan Buettcher** (Google), *không phải* "M." như bản cũ ghi. Trang 758–759,
DOI `10.1145/1571941.1572114`. Bài xác nhận `k = 60` là hằng số **do chính tác giả chốt trong
thí nghiệm pilot** — đúng giá trị `rrf_k` trong `retrieval.py`.

> ⚠️ Mục **FinTabNet** (mục 14.23) vẫn **chưa chốt được** tác giả/venue: tài liệu cũ ghi "Poddar
> et al. 2022, WACV", nhiều bài đã công bố lại ghi "Zheng et al., 2021". Phải lấy PDF chính
> thức trước khi trích.

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

## 3–7. Các tầng parser hình học (đã lưu trữ)

> Các mục 3 (hình học), 4 (dò vùng bảng), 5.1–5.3 (header nhiều tầng), 6 (tách/ghép bảng) và 7 (điều phối tài liệu) mô tả **parser hình học cũ** và đã chuyển sang
> [`archive/geometric_parser/docs/Doi_chieu_parser_hinh_hoc.md`](../archive/geometric_parser/docs/Doi_chieu_parser_hinh_hoc.md).
> Giữ lại ở đây: 5.4 (quy mô dữ liệu) và 5.5 (thực nghiệm Table Transformer trên corpus), vì chúng nằm trong pipeline hiện tại.

### 5.4 Quy mô tập dữ liệu — con số đã kiểm chứng

Các con số dưới đây đã đối chiếu với **bài báo gốc / repo chính thức**, không lấy từ blog
tóm tắt. Số "tổng" và số "tập train" tách riêng vì hay bị gộp nhầm.

| Bộ dữ liệu | Tổng | Train / Val / Test | Nguồn dữ liệu | Nhiệm vụ | Bài báo |
|---|---|---|---|---|---|
| **PubTabNet** | **568k** ảnh bảng | Train **548.592** (tách 60/20/20) | Bài báo khoa học trong **PMCOA**, >6.000 nhà xuất bản | TSR + TR (**không** có TD) | Zhong, ShafieiBavani, Jimeno-Yepes, ECCV 2020 |
| **FinTabNet** | **112.875** bảng (bản gốc) | xấp xỉ 78,5k / 9.650 / 9.289 | Báo cáo thường niên **S&P 500**, PDF | TSR + bbox ô | IBM; số liệu từ Smock et al. |
| **TableBank** | TD **417.234** bảng · TSR **145.463** bảng | TD: **260.582** / 10.000 / 8.000 · TSR: **130.463** / 10.000 / 5.000 | **Word + LaTeX** trên internet (weak supervision) | TD + TSR | Li, M., et al., LREC 2020 |

**Số lệch cần nói rõ khi trích (không được bỏ qua):**

- **PubTabNet — 568k là con số trích trong abstract, không khớp mọi bản phát hành.**
  Tách 60/20/20 cho train = 548.592. Nhưng TableFormer (CVPR 2022) đếm **509k**, và bản
  thi đấu ICDAR 2021 dùng **500.777 / 9.115 / 9.064**. Đây là **các bản phát hành và ngăn
  đánh giá khác nhau**, không phải sai số — nhưng phải ghi rõ mình đang nói về bản nào.
- **FinTabNet — 113k hay 97k?** Cả hai đều đúng, ở hai giai đoạn khác nhau. Smock et al.
  (arXiv:2303.00716) đo trực tiếp: bản gốc **112.875** bảng → sau khi hoàn thiện bbox và
  lọc chất lượng còn **97.475** bảng. Khoảng **97,4k** mà thấy trên Kaggle/HF gần như
  chính là con số sau lọc.
- **TableBank — con số "2,7 triệu bảng" là ứng viên thô trước lọc**, không phải benchmark
  cuối cùng. Con số nên dùng là **417.234 / 145.463**. Ngoài ra TableBank **không phải
  PDF tài chính** — nguồn là Word và LaTeX, nên giá trị chuyển miền sang báo cáo tài chính
  thấp hơn FinTabNet rõ rệt.

**Trả lời câu hỏi "cần bao nhiêu dữ liệu để huấn luyện":**

Không có con số ngưỡng cố định — nhưng Smock et al. cho **ví dụ định lượng rất rõ**, dùng
đúng kiến trúc TATR giữ nguyên và chỉ thay dữ liệu:

| Mô hình | Dữ liệu | Quy mô | Kết quả |
|---|---|---|---|
| TATR | ICDAR-2013, 34 bảng (DeepDeSRT) | ~34 | overfit nặng |
| TATR | FinTabNet gốc | 112.475 | **42%** exact-match trên ICDAR-2013 |
| TATR | FinTabNet đã canonicalize | 97.475 | **65%** — *cùng kiến trúc, cùng số lượng bảng* |
| TATR | PubTables-1M | ~1 triệu | 65% → **75%** |

> **Kết luận quan trọng cho luận văn:** ở quy mô ~100k bảng, **chất lượng annotation ăn
> đứt quy mô dữ liệu** (42% → 65% chỉ nhờ sửa nhãn). Vài chục nghìn bảng *đúng* vượt xa
> hàng trăm nghìn bảng *sai*. Đây là luận điểm mạnh ủng hộ lựa chọn rule-based: quy tắc
> tất định cho nhãn không nhiễu, khác với nhãn crowdsourced vốn bị lệch cấu trúc.

Về **ngân sách huấn luyện thực tế**: TATR định nghĩa 1 epoch = **720.000 mẫu** và chạy
30 epoch (tức ~21,6 triệu lượt mẫu, lấy mẫu ngẫu nhiên từ ~97–112k bảng duy nhất, có
cropping augmentation). Nghĩa là **một checkpoint mạnh không cần hàng triệu bảng**, nhưng
cần **hàng chục epoch lặp qua dữ liệu đã canonicalize**.

**Hai tài liệu đáng chú ý thêm:**

- **FinTabNet.c** — bản đã sửa của Microsoft, công bố kèm quy trình xử lý và mã nguồn tại
  `github.com/microsoft/table-transformer`. Nên dùng bản `.c` nếu cần đo độ chính xác
  đáng tin, vì bản gốc có lỗi nhãn và còn **lẫn lộn val/test** trong nhiều kết quả
  công bố.
- **ViFinTab** (Springer, DOI `10.1007/978-981-95-4963-4_32`) — bộ dữ liệu báo cáo tài
  chính **tiếng Việt**, có dữ liệu bảng xoay. Đây là ứng viên duy nhất tìm được khớp cả
  ngôn ngữ lẫn miền. ⚠️ *Chưa xác minh được quy mô — cần tải bài để chốt số.*

### 5.5 Thực nghiệm: Table Transformer trên chính corpus này

Chạy thật hai checkpoint `microsoft/table-transformer-detection` và
`microsoft/table-transformer-structure-recognition` (đều huấn luyện trên PubTables-1M,
mở rộng thêm FinTabNet) trên **toàn bộ corpus hiện tại: 12 PDF / 340 trang**
(33 trang có text layer, 307 trang scan). Mã ở ngoài repo, không đụng vào mã nguồn.

**Quy mô phát hiện:**

| | Rule-based | Table Transformer |
|---|---|---|
| 33 trang có text layer | **18** bảng | **43** bảng |
| 307 trang scan | 0 (không có text layer để dò) | **343** bảng |

**Phát hiện — mức trang, trên 33 trang có text layer (có ground truth):**

| | Số trang |
|---|---|
| Cả hai cùng thấy bảng (TP) | 13 |
| Chỉ TATR thấy (FP) | **13** |
| Chỉ rule-based thấy (FN) | 1 |

→ Precision **50%**, Recall **93%**. TATR nhạy, nhưng báo động giả nhiều.

**Phát hiện — 307 trang scan: 343 phát hiện, đa số là báo động giả.**

Không có ground truth cho phần scan, nên đã đo bằng **khe dọc lặp lại**: một bảng — kể
cả không viền — luôn thể hiện cùng các khe cột nội bộ trên nhiều dòng, còn văn xuôi thì
 méo. Ba nhóm đối chứng, cùng ngưỡng phát hiện 0.5:

| Nhóm | n | Khe cột lặp (median) | Tỉ lệ có ≥2 khe lặp |
|---|---|---|---|
| **POS** — trang rule-based xác nhận có bảng | 15 | **8.0** | **100%** |
| **NEG** — trang văn xuôi, rule-based loại | 28 | 1.0 | 29% |
| **SCAN** — 307 trang scan | 40 | **1.0** | **38%** |

Nhóm SCAN **thống kê giống nhóm báo động giả**, và xa nhóm bảng thật. Kết luận: phần lớn
343 phát hiện trên trang scan là văn xuôi bị nhận nhầm, **dù điểm tin cậy rất cao
(0.83–0.999)**. Bộ lọc khe lặp loại 62% của NEG trong khi giữ 100% của POS — tức dùng được
làm **bộ kiểm lọc đề xuất của TATR**.

**Cấu trúc — 15/18 bảng đo được (3 bảng bị bỏ vì ghép qua trang):**

| Kiểm tra | Kết quả |
|---|---|
| Số cột khớp | **11/15 (73%)** |
| Số dòng | **không đáng tin** — lệch từ ±1 đến +12 |

Bốn trường hợp lệch số cột đều rơi đúng vào vùng lỗi đã biết:
`18×5 → TATR 18×4` và `46×8 → TATR 61×6` là **hai bảng cạnh nhau bị chúng ta tách nhầm**;
`18×8 → 18×9` và `32×7 → 32×6` là chênh lệch biên.

**Chỗ rule-based thua (TATR tìm đúng bảng):**
`Thay_doi_thanh_phan_chi_so_VNDiamond_Q4_2024.pdf` trang 5, điểm 0.953 — phụ lục *"Tiêu chí
đánh giá chỉ số VN DIAMOND INDEX 3.0"* với header `Tiêu chí | Cổ phiếu thuộc bộ chỉ số |
Cổ phiếu không thuộc bộ chỉ số`. Đây chính là điểm mù đã ghi ở mục 16: bảng **văn bản
thuần, ít cột, ít số** mà `_find_regions` loại vì đòi `≥3` ô số. Trang 2 cùng file
(điểm 1.0) cũng nghi là danh sách 5 quỹ ETF dạng bảng.

**Chỗ TATR thua:**
Các trang **miễn trừ trách nhiệm và thông tin liên hệ** ("CAM KẾT PHÂN TÍCH", "TUYÊN BỐ MIỄN
TRỪ", khối email/điện thoại) bị TATR nhận là bảng với điểm tới **1.0** — 13 trang FP gần như
toàn bộ thuộc nhóm này. Ngược lại `VN30_VNFINLEAD_Q3` trang 3 có **2 bảng thật** mà TATR
phát hiện **0 bảng**. Cả hai hệ đều không có khái niệm bảng xuyên trang.

**Kết luận cho luận văn — không thay thế, mà là ý kiến thứ hai.**

TATR không thể thay `tables.py`: precision 50% trên chính miền của dự án, báo động giả
mạnh trên văn bản tiếng Việt, và không xử lý bảng xoay hay bảng ghép trang. Nhưng nó bắt
được đúng lớp bảng mà quy tắc của chúng ta **cấu trúc loại có chủ đích**. Vì vậy kiến
trúc đúng là **lai**: rule-based làm neo độ chính xác, TATR làm **bổ sung recall** cho
lớp "bảng không viền, ít số", mọi đề xuất của TATR phải qua bộ lọc khe lặp trước khi
nhận. Đây cũng là câu trả lời dứt điểm cho câu hỏi ở mục 12: bằng chứng không cho phép
kết luận "rule-based thắng mô hình", mà cho phép kết luận chính xác hơn là **quy tắc
tốt hơn về độ chính xác, mô hình tốt hơn về độ bao phủ, và cần cả hai**.

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

### 8.4 Khoảng trống đã biết — nơi cần nâng cấp

Chunking hai cấp là **nền tảng**, không phải đỉnh. Ba khoảng trống đã định danh, xếp theo mức
đáng làm (bảng công việc ở mục 14.1):

| # | Khoảng trống | Bài báo chỉ ra | Vì sao là giới hạn thật |
|---|---|---|---|
| 1 | **Chưa có đường đo** | — | Không có page-hit@k hay câu hỏi mẫu thì không chứng minh được cải tiến nào. **Chặn tất cả mục còn lại.** |
| 2 | Hai cấp là **cố định**, không theo loại phần tử | Mục 31 | Chunk theo **element type** cho kết quả *"largely improve"* trên báo cáo tài chính và **nhất quán hơn** giữa page-level với paragraph-level. Chunk cha của ta là **toàn bộ bảng**, chưa phân biệt caption / header / body / footnote |
| 3 | Chunk con **không đủ dữ liệu để tính** | Mục 32 | TableRAG nêu đúng lỗi: RAG *"computes percentage over the top-N most relevant chunks rather than the full table"*. Ta có `table_card` nhưng chưa dùng làm schema để chạy phép tính trên cả bảng |
| 4 | Row-level chunk có thể **quá nhỏ** | Mục 36 | Thực nghiệm mở: chunk 500 cho faithfulness 0.67 nhưng correctness 0.17 — trung thực mà thiếu bằng chứng |

**Điểm cần nói rõ khi trình bày:** `table_card` là *context* để LLM hiểu bảng, **không phải**
đơn vị tính toán. Mọi phép tính hiện chỉ chạy trên một dòng. Đây là giới hạn đã biết, không
phải lỗi ẩn.

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
| **RRF**, hằng số k=60 | **A** ✅ | Cormack, G. V., Clarke, C. L. A., & **Buettcher, S.** (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR '09, 758–759. DOI 10.1145/1571941.1572114. **Đã đọc PDF gốc**: tác giả thứ ba là *Stefan* Buettcher (Google), và bài ghi rõ *"k = 60 was fixed during a pilot investigation"* — `rrf_k` trong code lấy đúng hằng số gốc |
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
| Cần dữ liệu huấn luyện | Không | Cần — PubTabNet 568k, FinTabNet ~113k, TableBank 417k (mục 5.4) |
| Kiểm thử được | **Tất định — 187 test pass** | Hộp đen, cần bộ đánh giá riêng |
| Bảng không viền | **Xử lý trực tiếp** | Cần dữ liệu huấn luyện tương ứng |
| Chạy máy yếu / không GPU | **Có** | Thường cần GPU |
| Báo cáo lý do sai | **Có** (`warnings`, `lỗi từng bước`) | Khó |
| Độ phủ layout lạ | Yếu — cần thêm quy tắc | Mạnh hơn |

**Lập luận để trình bày — có số đo, không chỉ lập luận.** Trước đây mục này chỉ suy luận từ
quy mô corpus; nay đã đo thật trên 340 trang (mục 5.5):

| Tiêu chí | Rule-based | Table Transformer (PubTables-1M + FinTabNet) |
|---|---|---|
| Precision phát hiện bảng | 18/18 bảng tìm được đều có thật | **50%** (13 TP / 26 trang) |
| Recall | Bỏ sót lớp bảng ít cột, ít số | 93% |
| Cấu trúc | 11/15 khớp số cột với TATR | — |
| Báo động giả trên văn xuôi tiếng Việt | Không | Rất nhiều, điểm tin cậy tới 1.0 |
| Bảng xoay / bảng xuyên trang | Chưa xử lý | Chưa xử lý |
| Chi phí | Quy tắc, không cần dữ liệu, tất định | Cần checkpoint 110MB, hộp đen |

Điểm mấu chốt: **hai hệ thất bại ở hai hướng ngược nhau**. Quy tắc chính xác nhưng hụt lớp
bảng không viền ít số; mô hình bao phủ rộng nhưng không phân biệt được bảng với văn xuôi
tiếng Việt. Vì corpus chỉ có 340 trang vài bố cục lặp lại, quy tắc tất định cho độ chính
xác có thể chứng minh được và chi phí thấp hơn nhiều; nhưng luận văn **không nên** tuyên bố
rule-based thắng — hướng đúng là kiến trúc lai ở mục 5.5.

**Hướng lai:** dùng rule-based để tạo dữ liệu huấn luyện, rồi thay thế dần bằng mô hình
(`TableFormer`, hoặc đường OCR-free kiểu Donut: Lee, K., et al. (2022). *Donut: Document
understanding transformer.* ECCV 2022 — **mức B**).

---

## 13. Đánh giá: chỉ số nên dùng

Hiện **chưa có** bộ câu hỏi vàng. Đây là hạng mục nên làm trước khi tối ưu thêm bất kỳ thứ gì.

| Tầng | Chỉ số | Nguồn / mức |
|---|---|---|
| **Cấu trúc bảng** | **TEDS** | Zhong, X., ShafieiBavani, E., & Jimeno-Yepes, A. (2020), ECCV — **B**. Lưu ý: TEDS **không** có bài báo riêng, nó là đóng góp của chính bài PubTabNet |
| **Chất lượng truy xuất** | `Recall@k`, `MRR`, `nDCG` | Định nghĩa chuẩn IR — **B** |
| **Hệ thống RAG** | RAGAS | Es, S., et al. (2024). *RAGAS: Automated evaluation of retrieval augmented generation.* EACL System Demonstrations, 150–158 — **A** |
| **Tách biệt chất lượng truy xuất** | | Salemi, A., & Zamani, H. (2024). *Evaluating retrieval quality in retrieval-augmented generation.* SIGIR '24, 2395–2400 — **A** |
| **QA bảng + văn bản tài chính** | Bộ dữ liệu tham chiếu | Chen, W., et al. (2020). *TAT-QA.* ACL 2020 — **B**; Zhou, T., et al. (2021). *FinQA.* EMNLP 2021 — **B** |

**Cảnh báo phương pháp luận (quan trọng):** bài A1 (mục 17) chỉ ra
ROUGE/BLEU **không đo trung thực** chất lượng chunk. Vì vậy đừng đánh giá phần chunk bằng
ROUGE — hãy đánh giá **trực tiếp bằng TEDS** (cấu trúc) và **`Recall@k`** (truy xuất).

---

## 14. Danh mục tài liệu tham khảo (gộp)

Đây là **danh mục chuẩn**: 36 mục được trích dẫn trong tài liệu này, phân theo mức độ tin cậy
(mục 0). Các hướng **chưa làm** nằm ở Phần II (mục 17–26) — khi viết phần "phương pháp" chỉ
trích mục ở đây.

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
12. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR '09, 758–759. DOI 10.1145/1571941.1572114. **A** — đã đọc PDF gốc; tác giả thứ ba là *Stefan* Buettcher (Google).
13. Wang, L., Yang, N., Huang, X., et al. (2024). *Multilingual E5 text embeddings.* arXiv:2402.05672.
14. Nguyen, T., et al. (2020). *XLM-R: Cross-lingual language model pretraining at scale.* arXiv:1911.02116.
15. O'Gorman, L., & Wong, K. Y. (1986). *Page segmentation and structure deduction for image processing.* IEEE TPAMI, 8(3), 196–203.
16. Shur, A., et al. (2014). *XPSplit: A cross-platform, end-user extendable table data extraction library.* SC '14.
17. McKinney, M. (2017). *Camelot: Python library to extract tables from PDFs.*
18. Zhong, X., ShafieiBavani, E., & Jimeno-Yepes, A. (2020). *Image-based table recognition: data, model, and evaluation.* ECCV 2020, 564–580. DOI 10.1007/978-3-030-58589-1_34 (arXiv:1911.10683). — Giới thiệu **PubTabNet** (568k) **và** đề xuất chỉ số **TEDS**.
19. Li, M., Cui, L., Huang, S., Wei, F., Zhou, M., & Li, Z. (2020). *TableBank: A Benchmark Dataset for Table Detection and Recognition.* LREC 2020, 1918–1925 (arXiv:1903.01949). License Apache-2.0.
20. Smock, B., Pesala, R., & Abraham, R. (2023). *Aligning benchmark datasets for table structure recognition.* arXiv:2303.00716. — Nguồn cho **số liệu đã đo trực tiếp** của FinTabNet (112.875 → 97.475 sau khi lọc) và cho kết luận "canonicalize annotation quan trọng hơn quy mô dữ liệu".
21. Smock, B., Pesala, R., & Abraham, R. (2022). *PubTables-1M: Towards comprehensive table extraction from unstructured documents.* CVPR 2022, 4634–4642. — Bộ TSR lớn nhất hiện có, dùng làm đối chiếu.
22. Nassar, A., Livathinos, N., Lysak, M., & Staar, P. (2022). *TableFormer: Table structure understanding with transformers.* CVPR 2022, 4614–4623.
23. FinTabNet (IBM Developer Data Hub, `developer.ibm.com/exchanges/data/all/fintabnet/`). Bản canonicalize: `huggingface.co/datasets/bsmock/FinTabNet.c`. ⚠️ *Tác giả bài báo **chưa xác minh được**: tài liệu này trước đây ghi "Poddar, S., et al. (2022), WACV", còn nhiều bài đã công bố lại ghi "Zheng et al., 2021". Phải chốt từ PDF chính thức trước khi trích.*
24. *ViFinTab.* Springer, DOI 10.1007/978-981-95-4963-4_32. — Báo cáo tài chính **tiếng Việt**. ⚠️ *Chưa xác minh quy mô.*
25. Lee, K., et al. *Donut: Document understanding transformer.* ⚠️ *Chưa xác minh venue.*
26. Sarthi, R., et al. (2024). *RAPTOR: Recursive abstractive processing for tree-organized retrieval.* ICLR 2024.
27. Chen, W., et al. (2020). *TAT-QA: a hybrid approach to real-world table and text question answering.* ACL 2020.
28. Zhou, T., et al. (2021). *FinQA: a dataset of numerical reasoning over financial data.* EMNLP 2021.
29. Nguyễn Thành Hưng, Nguyễn Đức Hoàng Linh (2018). *VnCoreNLP: a Vietnamese NLP toolkit.*
30. Smith, M. (2007). *An overview of the Tesseract OCR engine.* ICDAR.

31. Jimeno Yepes, A., You, Y., Milczek, J., Laverde, S., & Li, R. (2024). *Financial
    report chunking for effective retrieval augmented generation.* arXiv:2402.05131.
    — **Ưu tiên cao nhất trong nhóm này.** Chunk theo **element type** do document
    understanding model gán, thay vì paragraph-level; đánh giá trên FinanceBench. Kết luận:
    *"element type based chunking largely improve RAG results on financial reporting"*, và
    element-based **nhất quán hơn** giữa page-level và paragraph-level accuracy (bài nói rõ
    chunking cơ bản đạt page-level cao nhưng paragraph-level thấp, thiếu nhất quán).
    ⚠️ Miền khác: FinanceBench là SEC filings tiếng Anh. Cần chạy lại trên corpus chứng
    chỉ quỹ tiếng Việt trước khi khẳng định. Tác giả đầu tiên đồng thời là tác giả PubTabNet.
32. Yu, X., Jian, P., & Chen, C. (2025). *TableRAG: A retrieval augmented generation framework
    for heterogeneous document reasoning.* EMNLP 2025, 5676–5684.
    [arXiv:2506.10380](https://arxiv.org/pdf/2506.10380) · code `github.com/yxh-y/TableRAG`.
    — Nêu đúng lỗi của row-level chunking: RAG *"computes percentage over the top-N most
    relevant chunks rather than the full table, and thus results in an incorrect answer"*.
    Giải pháp: ánh xạ chunk → **table schema gốc**, rồi **NL2SQL** chạy trên bảng đầy đủ.
    **Gap thật của dự án:** `table_card` + `expand_parent` đã hiện thực phần "mapping" nhưng
    chưa dùng schema để tính toán. Xem mục 8.4.
33. Günther, M., Mohr, I., Williams, D. J., Wang, B., & Xiao, H. (2024). *Late chunking:
    contextual chunk embeddings using long-context embedding models.* arXiv:2409.04701.
    — Embed **cả tài liệu**, chunk **sau** transformer trước mean-pooling.
    🚫 **Loại với cấu hình hiện tại:** đòi embedding model context ≥8192 token (jina-v2), còn
    E5-small chỉ 512 token và model card ghi *"Long texts will be truncated to at most 512
    tokens"*. Muốn làm phải đổi model. Giữ để nêu là hướng đã cân nhắc và loại, không phải thiếu sót.
34. Wang, Z., et al. (2026). *Reconstructing context: a rigorous analysis of late chunking and
    contextual retrieval.* arXiv:2504.19754. — Đối chiếu trực tiếp hai kỹ thuật trên NFCorpus.
    Kết luận: *"neither technique offers a definitive solution"*; **Contextual thắng Late**
    (0.202 vs 0.192). Quan trọng vì `Chunk.text` của dự án **đã prepend sẵn** header path →
    dự án đang ở phía thắng mà không cần late chunking.
35. Ghoneim, A., et al. (2026). *A systematic investigation of document chunking strategies and
    embedding sensitivity.* arXiv:2603.06976. — Quét **36 chiến lược chunking** trên 6 miền ×
    5 embedding model. Kết luận: structure-aware và adaptive **thắng fixed-size một cách nhất
    quán**; late chunking vào top-3 ở vài miền. Dùng làm số liệu đối chiếu khi chọn chiến lược.

### Mức B' — bằng chứng thực nghiệm mở, đo được

36. *FinanceBench RAG* — mã nguồn mở, 42 SEC filings, 100 câu hỏi, LLM judge + RAGAS +
    page-hit@k. [github.com/GoKu1402/financebench-rag](https://github.com/GoKu1402/financebench-rag).
    — Ba kết quả có giá trị trực tiếp cho dự án:
    · **metadata filter thắng**: page-hit@1 **0.21 → 0.29**, là cải tiến rẻ nhất và mạnh nhất.
    · **RRF không cải thiện**: BM25 + dense + RRF cho page-hit@1 **0.20**, *thấp hơn* dense
      đơn thuần (0.21). → Cần đo lại trên corpus của ta trước khi giữ RRF.
    · **cái bẫy chunk nhỏ**: chunk 500 cho faithfulness **0.67** nhưng correctness chỉ
      **0.17** — chunk nhỏ dễ trung thực nhưng thiếu bằng chứng để trả lời. Row-level chunk
      của dự án đang ở rìa cái bẫy này.
    ⚠️ *Số liệu từ blog/repo tự báo cáo, chưa peer-review. Dùng để định hướng thử nghiệm, không
    dùng làm số liệu trích trong luận văn.*

### Mức C — đóng góp của dự án, không có nguồn ngoài

Xem bảng ở mục 11.1. Tóm tắt 16 mục: dò bảng bằng mật độ số; nới vùng có ràng buộc lưới cột;
tiêu chí ổn định lưới; loại văn xuôi trôi vào header; loại ô trộn bằng điều kiện kép; tách
"STT/Mã CP"; dòng tiêu đề nhóm làm ranh giới; ưu tiên khe dọc hơn tiêu đề nhóm; ghép bảng qua
trang bằng chữ ký cột; nhận diện cột khóa theo dữ liệu; đơn vị theo từng giá trị; caption
bằng regex; chẩn đoán OCR; metadata tài liệu + lọc cứng; biến thể số kép; token hoá tiếng Việt
giữ nguyên cụm.

---

## 14.1 Hạng mục việc: nâng cấp chiến lược truy xuất

Sắp theo mức đáng làm, đã cân nhắc chi phí và rủi ro. Xem mục 15 để chốt thứ tự.

| # | Việc | Nguồn | Chi phí | Vì sao | Rủi ro |
|---|---|---|---|---|---|
| 1 | Bổ sung benchmark tự đánh giá (page-hit@k, câu hỏi tự viết theo corpus) | — | thấp | **Chặn mọi thứ khác**: hiện chưa có đường đo nào nên không chứng minh được cải tiến nào | Cần viết ~30–50 câu hỏi có đáp án |
| 2 | Đo lại BM25 vs dense vs RRF trên chính corpus | Mục 36 | thấp | RRF có thể **không** giúp ích; mà `bm25_weight` đang để mặc định mà không có số | Kết quả có thể chứng minh RRF thừa |
| 3 | Áp **metadata pre-filter** trước khi xếp hạng chunk | Mục 36 | thấp | Rẻ nhất, hiệu quả nhất theo thực nghiệm công khai; `docmeta.py` + `infer_doc_filter` đã có sẵn | Rất thấp |
| 4 | Chunking theo **element type** thay vì 2 cấp cố định | Mục 31 | trung bình | Là chuẩn so sánh đúng cho bài báo trùng miền; cấu trúc đã có sẵn trong `tables.py` | Cần chạy lại trên tiếng Việt |
| 5 | Dùng `table_card` làm schema để **NL2SQL** trên bảng đầy đủ | Mục 32 | trung bình | Vá đúng lỗi "chỉ tính trên chunk", không phải cải tiến hình thức | Cần bảng ≥20 dòng như TableRAG yêu cầu |
| 6 | Đo ngưỡng độ dài row-chunk (rủi ro mất bằng chứng) | Mục 36 | thấp | Row-level chunk có thể quá nhỏ → correctness giảm | Thấp |
| 7 | Semantic chunking / LLM boundary | Mục 35 | cao | Chưa rõ hơn cấu trúc trên bảng tài chính | Cao, khó chứng minh |

**Không làm:** late chunking (mục 33) — E5 512 token không đủ cửa sổ ngữ cảnh.

---

## 15. Danh sách kiểm chứng trước khi viết luận văn

| # | Việc | Vì sao |
|---|---|---|
| 1 | Tra **mọi mục B** ở mục 14 và điền số trang/DOI thật | Tôi không tra được mạng |
| 2 | Sửa tác giả bài RRF nếu `Buettcher, M.` không đúng | Xem cảnh báo mục 0 |
| 3 | Xác minh XPSplit: vị trí tác giả `Ameri`/`Premkupar`/`Buttener` | Mục 3.3 |
| 4 | Tra **"filtered vector search pre-filtering recall"** | Mục 9.3 còn nợ một nguồn |
| 5 | ~~Kiểm chứng "1,52% trang chứa bảng"~~ | **Đã gỡ** — không tìm được nguồn, đã xoá khỏi mục 5.3. Dùng số liệu đã đo ở mục 5.4 thay thế |
| 6 | Tải FinTabNet / PubTabNet về `data/references/` | Nếu muốn so sánh định lượng |
| 7 | Dựng bộ câu hỏi vàng và đo `Recall@k` + TEDS | Mục 13 — ưu tiên cao nhất |

---

## 16. Các điểm mù đã đo — ưu tiên công việc tiếp theo

Đo trên PDF tổng hợp, không phải suy đoán:

| Đặc trưng | Kết quả | Nguyên nhân | Đề xuất |
|---|---|---|---|
| Bảng 2 cột **văn bản thuần**, 0 số | **0 bảng** → rơi thành prose rác | `_find_regions` đòi `≥3 ô số` | **ĐÃ XÁC NHẬN trên dữ liệu thật** (mục 5.5): bảng *"Tiêu chí đánh giá chỉ số VN DIAMOND INDEX 3.0"* trong `VNDiamond_Q4_2024` p5 bị bỏ sót, TATR bắt được với điểm 0.953. Đề xuất: thêm điều kiện dựa trên cấu trúc (mục 4.3) **hoặc** dùng TATR làm ý kiến thứ hai |
| Bảng **xoay 90°** | **0 bảng**, mỗi từ thành một dòng; `char_w` bị bóp méo | `build_lines` chỉ gom theo `y` | Dùng `line["dir"]` từ `get_text("dict")` — đã kiểm chứng cho `(0.0,-1.0)` khi xoay, `(1.0,0.0)` khi thẳng. **TATR không giải quyết được**: trên file xoay nó vẫn phải qua bộ lọc khe lặp như mọi trang scan khác |
| Văn xuôi 2 cột | Thứ tự đọc sai → BM25 nhiễu | `_prose_blocks` nối theo `y` | Tách dải cột bằng `_empty_runs` |
| Vùng nghi là bảng bị loại | **Không có dấu vết** | `_is_plausible` từ chối rồi bỏ | Ghi `warnings` theo trang kèm lý do |
| Bảng có viền | Bỏ qua | Không dùng `find_tables()` | Dùng `strategy="lines_strict"` làm ý kiến thứ hai |

---
---

# PHẦN II — DANH MỤC MỞ RỘNG (hướng chưa làm)

Phần I (mục 1–16) là phần **đã hiện thực** trong `etflab/` và là nơi trích khi viết phần
"phương pháp". Phần II gom những bài **chưa dùng** — giữ để biết hướng nào đã có người làm,
và để tránh phát minh lại. Danh mục chuẩn vẫn là **mục 14**; ở đây chỉ liệt kê mục **chưa**
nằm trong đó.

Trước khi trích bất kỳ mục B nào: **tra arXiv / ACM / ACL Anthology**. Mức độ tin cậy xem mục 0.

## 17. Ghi chú về hai bài nền có sẵn trong repo

`data/references/` có 2 file PDF, và là nguồn cho **toàn bộ mục A**:

**A1. Peng, Q., Luo, X., et al. — *Dual-granularity chunking and dynamic context augmentation*.
ESWA 332, 133546.** Xem mục 14.1 trong danh mục.
- Bản PDF trong repo là bản **tiếng Việt**, tạp chí `Expert Systems with Applications`.
- Bài **cũng nêu điểm yếu của chính phương pháp nó đề xuất**: phân đoạn dựa trên LLM
  *"ảo giác"*, không ổn định; và ROUGE/BLEU đo không trung thực. Đây chính là lý do dự án
  không sao chép prompt-based chunking mà chunk theo **cấu trúc bảng** (tất định, kiểm thử được).

**A2. Cheng, H., Wang, K., et al. — *CFinRAG*. EAAI 175, 114670.** Xem mục 14.2.
- `data/references/1-s2.0-S0952197626009528-main.pdf`
- Pipeline *textualize → knowledge graph → RL query reconstruction → dual-level retrieval*.
  Bảng benchmark của bài là baseline so sánh của dự án (`NaiveRAG`, `Self-RAG`, `CRAG`,
  `HippoRAG`, `CFinRAG`).
- Hạn chế do chính tác giả nêu: đánh giá trên tập tự xây, LLM-as-judge có thiên lệch, chưa có
  đối chiếu chuyên gia.

> ⚠️ **Cảnh báo tra cứu.** Trong A1, trích dẫn "(Grattafiori và cộng sự, 2024)" *trông như*
> một bài về chunking, nhưng tra danh mục thì đó là **báo cáo kỹ thuật Llama 3**
> (arXiv:2407.21783), dùng làm baseline *"không có truy xuất"*. Đừng mất công tìm nó như một
> bài về RAG.

## 18. Nền tảng RAG và khảo sát

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Gao, Y., Xiong, Y., Gao, X., et al. (2023). *Retrieval-augmented generation for large language models: A survey.* arXiv:2312.10997. | Mục 1.3 "sai lầm trong RAG" — nền cho phần đánh giá |
| **A** | Fan, W., Ding, Y., Ning, L., et al. (2024). *A survey on RAG meeting LLMs.* KDD '24, 6491–6501. | Góc nhìn hệ thống |
| **A** | Zhu, Y., et al. (2023). *Retrieval-augmented language models: A survey.* (trích trong A2) | Góc nhìn lịch sử |

## 19. Chunking và ngữ cảnh — bổ sung

Các mục chunking **đã dùng** nằm ở mục 8 và 14.31–14.36. Ở đây chỉ bổ sung phần chưa dùng:

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Li, Y., Dong, B., Guerin, F., Lin, C. (2023). *LLMLingua: Compressing prompts for accelerated inference of large language models.* EMNLP '23, 6342–6353. | Nén prompt, giảm token — hữu ích khi context bảng lớn |
| **A** | Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A. (2023). *Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step tasks.* (IRCoT) ACL '23, 10014–10037. | Suy luận nhiều bước trên văn bản đã truy xuất |

## 20. Truy xuất nâng cao

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Koroteev, M. V. (2021). *BERT: A review of applications in NLP and understanding.* arXiv:2103.11943. | Tổng quan nền tảng cho phía dense |
| **A** | Ma, X., Gong, Y., He, P., et al. (2023). *Query rewriting for retrieval-augmented large language models.* EMNLP '23, 5303–5315. | Chuẩn hoá câu hỏi tài chính trước khi truy xuất — hợp với `tokenize` số VN |
| **A** | Jiang, Z., Xu, F. F., Gao, L., et al. (2023). *Active retrieval augmented generation.* (FLARE) EMNLP '23, 7969–7982. | Truy xuất có điều kiện trong quá trình sinh |

## 21. PoT / suy luận số — phase kế tiếp

Chưa cài đặt. Nền tảng nếu muốn trả lời *"cổ phiếu nào tăng mạnh nhất tháng 6"*. Lưu ý mục
8.4: dùng `table_card` làm schema để chạy phép tính trên **cả bảng**.

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **B** | Liu, T., et al. (2022). *Program of Thoughts prompting.* arXiv:2211.10435. | LLM sinh **chương trình** thay vì sinh đáp án từng bước |
| **B** | Gao, L., et al. (2022). *PAL: Program-aided language models.* arXiv:2211.11435. | Sinh chương trình rồi chạy bằng interpreter |

## 22. Đánh giá RAG — bổ sung

Mục 13 chọn chỉ số; hai mục RAGAS + Salemi đã ở mục 14.8–14.9. **Nên làm trước khi tối ưu
thêm bất kỳ thứ gì** (xem hạng mục 1 ở mục 14.1):

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Sudhi, V., et al. (2024). *RAG-Ex: A generic framework for explaining retrieval augmented generation.* SIGIR '24, 2776–2780. | Giải thích vì sao một đoạn được truy xuất |
| **A** | Wang, J., et al. (2023). *Is ChatGPT a good NLG evaluator? A preliminary study.* arXiv:2303.04048. | Nền cho LLM-as-judge |
| **A** | Zheng, L., Chiang, W.-L., et al. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.* NeurIPS 36, 46595–46603. | MT-Bench — chuẩn đánh giá bằng LLM giám khảo |
| **B** | Robertson, S., et al. — chỉ số MRR, NDCG | Định nghĩa `Recall@k` và `MRR` cho phase đánh giá |

> Lưu ý: "Zheng et al." ở đây là **MT-Bench**, không liên quan gì tới cuộc tranh tác giả của
> FinTabNet ở mục 14.23. Đừng nhầm hai nguồn này.

## 23. RAG nâng cao và truy xuất theo đồ thị

Chưa cài đặt. Hữu ích nếu muốn nối bảng với tri thức tài chính khác.

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Jimenez Gutierrez, B., et al. (2024). *HippoRAG.* NeurIPS 37, 59532–59569. | Baseline so sánh trong A2 |
| **A** | Edge, D., et al. (2024). *From local to global: A graph RAG approach to query-focused summarization.* arXiv:2404.16130. | Microsoft GraphRAG — tổng hợp cộng đồng |
| **A** | Peng, B., Zhu, Y., Liu, Y., et al. (2025). *Graph retrieval-augmented generation: A survey.* ACM TOIS 44(2). | Khảo sát nhánh đồ thị |
| **A** | Guo, Z., Xia, L., Yu, Y., et al. (2024). *LightRAG.* arXiv:2410.05779. | RAG nhẹ, dễ học |
| **A** | Zhang, X., Wang, M., Yang, X., et al. (2024). *Hierarchical RAG model with rethink for multi-hop QA.* arXiv:2408.11875. | Gần ý *"truy xuất dòng nhỏ, mở rộng chunk lớn"* |
| **A** | Yu, Y., Ping, W., Liu, Z., et al. (2024). *RankRAG.* NeurIPS 37, 121156–121184. | Reranking context |
| **A** | Yan, S.-Q., Gu, J.-C., Zhu, Y., Ling, Z.-H. (2024). *Corrective retrieval augmented generation* (CRAG). arXiv:2401.15884. | Baseline; tự sửa khi truy xuất sai |
| **A** | Asai, A., Wu, Z., Wang, Y., et al. (2023). *Self-RAG.* ICLR 2024. | Baseline; phản chiếu điển hình |
| **A** | Jeong, S., et al. (2024). *Adaptive-RAG.* NAACL 2024, 7029–7043. | Chọn số lần truy xuất theo độ khó câu |
| **A** | Ram, O., et al. (2023). *In-context retrieval-augmented language models.* TACL 11, 1316–1331. | Nền học thuyết |

## 24. Tài chính và LLM — bổ sung

| Mức | Tài liệu | Liên quan tới |
|---|---|---|
| **A** | Li, Y., Wang, S., Ding, H., Chen, X. (2023). *Large language models in finance: A survey.* ACM ICAIF '23, 374–382. | Khảo sát riêng miền tài chính |
| **A** | Shah, S., Ryali, S., Venkatesh, R. (2024). *Multi-document financial question answering using LLMs.* arXiv:2411.07264. | QA tài chính đa tài liệu — rất gần bài toán của dự án |
| **A** | Li, J., Bian, Y., Wang, G., et al. (2023). *CFGPT.* arXiv:2309.10654. | Trợ lý tài chính |
| **A** | Yuan, Z., et al. (2024). *FinLLMs.* IEEE Trans. Big Data. | **Có thể dùng để sinh câu hỏi vàng** cho phase đánh giá |
| **A** | Kertkeidkachorn, N., Nararatwong, R., Xu, Z., Ichise, R. (2023). *FinKG.* IEEE ICSC, 90–93. | Đồ thị tri thức tài chính |
| **A** | Cheng, H., Wang, K., Tan, X. (2024). *A link prediction method for Chinese financial event knowledge graph based on GAT and CNN.* EAAI 138, 109361. | Bài tiền thân của A2 |

## 25. Huấn luyện LLM — tham khảo, không dùng

A2 trích nhóm này để bối cảnh. Không dùng trực tiếp trong `etflab`.

- **A** Wei, J., et al. (2022). *Chain-of-thought prompting.* NeurIPS 35, 24824–24837.
- **A** Achiam, J., et al. (2023). *GPT-4 technical report.* arXiv:2303.08774.
- **A** Grattafiori, A., et al. (2024). *The Llama 3 herd of models.* arXiv:2407.21783. ← xem cảnh báo mục 17
- **A** Hu, E. J., et al. (2022). *LoRA.* ICLR 2022.
- **A** Chung, H. W., et al. (2023). *PaLM.* JMLR 24(240), 1–113.

## 26. Danh sách đọc và việc cần làm với danh mục

**Nếu chỉ đọc 5 mục:**

1. **A1** — dual-granularity. Lý thuyết nền của thiết kế chunk trong dự án.
2. **Jimeno Yepes et al. 2024** (mục 14.31) — *Financial report chunking*. Bài trùng miền gần
   nhất, và là chuẩn để đo xem chunking hai cấp của ta có đủ tốt không. **Đọc trước TAT-QA.**
3. **Robertson & Zaragoza 2009** (mục 14.10) — BM25. Vì `retrieval.py` hiện thực đúng công thức này.
4. **Cormack et al. 2009** (mục 14.12) — RRF. Vì `retrieval.py` dùng `rrf_k=60`.
5. **TAT-QA** (mục 14.27) — để biết bộ dữ liệu QA bảng + văn bản tài chính chuẩn trông như thế nào.

**Việc cần làm với danh mục:**

1. Tra mục **B** trước khi trích vào luận văn — chưa kiểm chứng được trong repo.
2. Bổ sung mục **B** vào `data/references/` để pipeline test được trên chúng.
3. Xác minh DOI của A1: năm **2027** trong PDF có vẻ là lỗi đánh số của nhà xuất bản.
4. Chốt tác giả + venue của FinTabNet (mục 14.23) — hiện còn tranh chấp *Poddar et al. 2022
   (WACV)* vs *Zheng et al. 2021*.
5. Thêm TEDS (mục 14.18) vào pipeline đánh giá — mục 13 khuyên dùng nhưng code chưa có.

Chi tiết thiết kế và ước lượng công sức nằm ở `archive/geometric_parser/docs/Huong_dan_ky_thuat.md` mục 15.