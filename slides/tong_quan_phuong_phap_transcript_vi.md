# Transcript thuyết minh — Methods Overview: Parsing, Chunking, and Evaluation

> Dùng kèm bản `slides/tong_quan_phuong_phap.pdf` (25 slide).
> Thời lượng gợi ý: khoảng 35–40 phút nếu nói đủ ý; có thể rút còn 25–30 phút bằng cách lược bớt các slide 12, 22–24.
> Các con số trong transcript viết theo cách đọc tiếng Việt (dấu phẩy thập phân); trên slide dùng dấu chấm thập phân.
> Khi trình bày, không cần đọc nguyên văn từng câu; dùng các đoạn dưới đây như lời dẫn và ý chính.

---

## Slide 1 — Title page

**Thời lượng:** ~45 giây

**Lời thoại:**

Kính thưa thầy cô và các bạn. Hôm nay tôi sẽ trình bày phần tổng quan phương pháp của đồ án: cách chúng ta parse dữ liệu, chia chunk, đánh giá trên bộ câu hỏi vàng, và so sánh với ColPali.

Phạm vi của phần này gồm 7 file PDF, 27 trang và 130 câu hỏi vàng. Parser được dùng xuyên suốt là `unstructured` ở chế độ `hi_res`; toàn bộ thí nghiệm chạy trên CPU, không dùng API trả phí. Điểm cần nhớ ngay từ đầu: tất cả phương pháp đều dùng chung một parser và một bộ gold, nên thứ duy nhất thay đổi là **đơn vị truy hồi**.

**Chuyển ý:** Để thấy bố cục của phần trình bày, mời thầy cô và các bạn xem slide tiếp theo.

---

## Slide 2 — Outline

**Thời lượng:** ~45 giây

**Lời thoại:**

Phần trình bày có bốn phần. Phần một là bối cảnh và tổng quan: dữ liệu, parser, các phương pháp chia chunk và kết quả chính. Phần hai đi sâu vào **Row-JSON** — phương pháp cho kết quả tốt nhất trong nhóm không dùng hình học. Phần ba đi sâu vào **Chipper** — phương pháp theo paper nhưng có một giới hạn rất quan trọng với bảng số. Phần bốn là cách chạy lại thực nghiệm: môi trường, lệnh parse, lệnh đánh giá và demo thư viện.

**Chuyển ý:** Chúng ta bắt đầu với bài toán và dữ liệu.

---

## Slide 3 — Problem and dataset

**Thời lượng:** ~2 phút

**Lời thoại:**

Bài toán của đồ án là trích xuất và truy hồi thông tin từ các bảng số không có viền trong báo cáo tài chính và báo cáo ETF tiếng Việt. Câu hỏi vàng điển hình là: “Giá trị của mã cổ phiếu X là bao nhiêu?”. Mục tiêu không phải là xây dựng một parser vạn năng, mà là so sánh các **đơn vị truy hồi** trên cùng một parser. Vì vậy, phần so sánh này cô lập được ảnh hưởng của cách chia chunk.

Bộ dữ liệu gồm 7 PDF: 3 báo cáo dự báo của công ty chứng khoán và 4 bản tin SSI; tổng cộng 27 trang và 10 bảng vàng. Có 130 câu hỏi, mỗi file từ 10 đến 40 câu. Các PDF có text layer, không phải bản scan. Gold trang được xây dựng độc lập với parser: 111 câu chỉ nằm trên một trang, 18 câu nằm trên hai trang và 1 câu nằm trên ba trang.

Điểm rất quan trọng là **độ phủ của parser**. `unstructured` giữ được 126 trên 130 giá trị, tức 96,9%, ở mức element; và 122 trên 130, tức 93,8%, ở mức chunk. Cả bốn giá trị bị thiếu đều là thiếu **giá trị trong ô**, không phải mất cả dòng. Vì vậy, khi nhìn kết quả retrieval, phải nhớ rằng trần của phương pháp bị giới hạn bởi parser.

Phần đánh giá dùng BM25 ở mức chunk, xếp hạng trong từng file; ở mức trang thì lấy điểm cao nhất theo trang trên 27 trang. Metric là Recall@k và MRR. Bộ câu hỏi vàng được sinh từ các ô bảng của parser lưu trữ và giữ cố định, để đo phần dữ liệu bị mất khi chuyển sang parser mới.

**Chuyển ý:** Vậy parser mới hoạt động như thế nào? Mời xem slide 4.

---

## Slide 4 — What parses the PDF: one pipeline

**Thời lượng:** ~2 phút

**Lời thoại:**

Slide này mô tả đường đi của dữ liệu. Từ PDF có text layer, `unstructured` chạy ở chế độ `hi_res`. Trong pipeline có YOLOX để phát hiện layout và Table Transformer để nhận diện bảng. Kết quả bảng không phải là tọa độ hay đường kẻ, mà là HTML nằm trong `metadata.text_as_html`.

Sau đó, một parser HTML nhỏ trong project đọc `text_as_html` và chuyển thành `UTable` — một cấu trúc bảng thống nhất gồm cột, dòng, số trang và caption. Từ `UTable` này, ba nhóm phương pháp chia chunk cùng chạy: Row-JSON, Chipper và Dual.

Có ba điểm cần nhấn. Thứ nhất, bảng lấy từ HTML do Table Transformer sinh ra, **không dùng tọa độ hay ruling lines**. Thứ hai, `PyMuPDF` chỉ dùng cho gold trang và render trang cho ColPali, không dùng để parse bảng. Thứ ba, `row_json` không đọc PDF; nó chỉ tiêu thụ `UTable` đã được dựng sẵn.

**Chuyển ý:** Từ HTML đến `UTable` là bước chuẩn hóa quan trọng, vì mọi phương pháp đều dùng chung đầu ra này.

---

## Slide 5 — From table HTML to UTable

**Thời lượng:** ~2 phút

**Lời thoại:**

Bước chuẩn hóa có bốn việc. Một, dùng `html.parser.HTMLParser` để đọc các thẻ `tr` và `td`. Hai, trải `colspan` và `rowspan` thành một lưới chữ nhật, để mọi dòng có cùng số cột. Ba, xác định header: ưu tiên `thead` hoặc `th`; nếu không có thì suy ra từ tỷ lệ ô số; nếu header nhiều tầng thì nối bằng dấu gạch chéo. Bốn, bỏ các cột trống ở cuối và gắn title hoặc caption gần nhất phía trước.

Đầu ra là một `UTable` chung, ví dụ gồm cột `Ticker`, `Weight`, dòng `MBB`, `7.18%`, số trang và caption. Điểm mấu chốt là **cả ba nhóm chunking đều nhận cùng một `UTable`**. Khác biệt duy nhất còn lại là cách chia bảng, không phải cách parse PDF.

Có một cảnh báo: lỗi header hoặc lỗi merge của Table Transformer sẽ đi thẳng vào key JSON. Bốn giá trị thiếu trên 130 câu là giới hạn của parser, không phải lỗi của chunking. Khi đọc kết quả, phải tách hai nguyên nhân này.

**Chuyển ý:** Với cùng đầu vào đó, chúng ta có những cách chia chunk nào?

---

## Slide 6 — Methods used

**Thời lượng:** ~2 phút

**Lời thoại:**

Slide này liệt kê các phương pháp. Có ba nhóm chunking chính. **Row-JSON** lấy mỗi dòng bảng làm một object JSON; tạo ra 332 chunk, đơn vị trả về là một dòng, phù hợp để tra cứu một ô. **Chipper** theo bốn quy tắc của paper; bảng được giữ nguyên, nên chỉ có 136 chunk, đơn vị trả về thường là cả bảng hoặc một đoạn lớn, phù hợp với câu hỏi cần ngữ cảnh bảng. **Dual** chia hai mức: câu/dòng nhỏ và đoạn lớn; tạo 935 chunk, phù hợp với chiến lược small-to-big.

Ngoài ra có hai baseline. **BM25 page** dùng text của `unstructured` nhưng không chia chunk, mỗi trang là một đơn vị; đây là baseline rất rẻ. **ColPali** dùng ảnh trang, zero-shot, mỗi trang là một đơn vị; phù hợp để lọc trang hoặc xử lý bản scan.

Có hai con số nền: cả ba phương pháp text dùng chung 809 element và 14 phần tử Table; bộ vàng có 10 bảng. Số chunk khác nhau là vì đơn vị chia khác nhau, không phải vì parser khác nhau. `chipper_prefix` chỉ là một biến thể của Chipper: thêm hai câu đầu hoặc caption vào chunk, không dùng LLM.

**Chuyển ý:** Vậy mỗi phương pháp có ưu nhược điểm gì và hợp với loại câu hỏi nào?

---

## Slide 7 — Methods: pros / cons and suitable questions

**Thời lượng:** ~2 phút

**Lời thoại:**

Không có phương pháp nào tốt nhất cho mọi câu hỏi. **Row-JSON** có độ chính xác cao, rẻ, tự chứa ngữ cảnh và dễ audit; nhưng nó mất cấu trúc toàn bảng, nên không trả lời được câu hỏi cần tổng hợp nhiều dòng. Nó hợp với câu hỏi một ô: “Trọng số của MBB là bao nhiêu?”, so sánh vài mã, hoặc tra một giá trị.

**Chipper** đơn giản, không có tham số, giữ bảng nguyên vẹn và có R@5/R@10 cao; nhưng cả bảng thành một chunk, khoảng 2,3 nghìn ký tự, nên R@1 thấp. Nó hợp với câu hỏi “bảng này nói gì?”, tóm tắt, hoặc cung cấp ngữ cảnh cho LLM.

**Dual** có hai mức và có thể rerank; nhưng nhiều tham số, và bước rerank/dense có thể làm giảm độ chính xác ở mức ô. Nó hợp với câu hỏi cần ngữ cảnh đoạn văn, sinh câu trả lời, hoặc dữ liệu prose.

**ColPali** không cần parser/OCR, nhìn được layout; nhưng đơn vị là cả trang, không trả lời trực tiếp một ô, và bản 256M zero-shot còn yếu. Nó hợp với PDF scan, chọn trang, hoặc fallback khi mất text layer. **BM25 page** chỉ nên xem là baseline đối chứng.

**Chuyển ý:** Vậy các con số thực nghiệm nói gì?

---

## Slide 8 — Results

**Thời lượng:** ~2 phút 30 giây

**Lời thoại:**

Đây là bảng kết quả chính. Cột đầu là số chunk; các cột tiếp theo là R@1, R@5, MRR ở mức chunk; hai cột cuối là R@1 và MRR ở mức trang. Các phương pháp text được xếp theo R@1 chunk; ColPali và BM25 page chỉ có kết quả mức trang.

Kết quả nổi bật nhất là **Row-JSON**: 81,5% R@1 chunk, 90,0% R@5 chunk, MRR 0,850; ở mức trang là 90,0% R@1 và MRR 0,950. Con số này cao hơn `dual_small` 15,4 điểm R@1 chunk, và cao hơn Chipper khoảng 3 lần.

**Chipper** có R@10 89,2%, gần bằng Row-JSON, nhưng R@1 chỉ 27,7%. Điều này có nghĩa là bảng chứa câu trả lời thường nằm trong top-10, nhưng hiếm khi đứng đầu. Đây là khác biệt giữa “chứa” và “trả lời đúng”.

**ColPali** đạt 13,8% R@1 trang, cao hơn BM25 page 4,6%, nhưng vẫn thấp hơn Row-JSON khoảng 6,5 lần. Vì vậy, trong bộ dữ liệu này, ColPali có ích để lọc trang nhưng chưa đủ để trả lời trực tiếp.

**Chuyển ý:** Nếu phải chọn theo mục tiêu, chúng ta chọn như thế nào?

---

## Slide 9 — Choose by goal

**Thời lượng:** ~2 phút

**Lời thoại:**

Slide này là bảng chọn nhanh theo mục tiêu. Nếu cần một giá trị đơn lẻ hoặc một mã, chọn **Row-JSON + BM25**: 81,5% R@1 chunk, 90,0% R@1 trang, mỗi chunk khoảng 491 ký tự. Nếu cần ngữ cảnh toàn bảng, dùng Row-JSON kèm parent table, hoặc Chipper; cách này giữ được small-to-big mà không làm loãng chỉ mục ô.

Nếu câu hỏi mang tính định tính hoặc prose, dùng Dual bản large hoặc một index prose riêng; tránh để chunk bảng chiếm top-1 trong câu hỏi số. Nếu PDF là bản scan hoặc mất text layer, ColPali sau fine-tuning và có GPU là hướng phù hợp; nhưng bản 256M zero-shot chỉ đạt 13,8% R@1 trang, chưa đủ để trả lời. Nếu cần baseline học thuật, dùng Chipper có prefix hoặc BM25 page vì đơn giản, tái lập được và dễ so sánh.

Bài học xuyên suốt là: **không trộn tất cả loại chunk vào một index**. BM25 không phân biệt được chunk “chứa” với chunk “trả lời”. Khi thêm prose hoặc cả bảng vào một index số, R@1 giảm mạnh: Dual còn 50,0%, Chipper còn 27,7%.

**Chuyển ý:** Bây giờ tôi đi sâu vào phương pháp tốt nhất trong nhóm text: Row-JSON.

---

## Slide 10 — Deep dive 1: Row-JSON

**Thời lượng:** ~30 giây

**Lời thoại:**

Đây là phần đi sâu thứ nhất. Câu hỏi dẫn dắt là: vì sao “mỗi dòng bảng là một chunk” lại thắng trong các câu hỏi tra cứu ô? Câu trả lời nằm ở việc dòng bảng chứa đúng thực thể, đúng giá trị và đúng metadata cần thiết.

**Chuyển ý:** Mời xem cấu trúc cụ thể của một object Row-JSON.

---

## Slide 11 — Row-JSON: one table row, one object

**Thời lượng:** ~1 phút 30 giây

**Lời thoại:**

Mỗi dòng bảng được chuyển thành một object JSON. Ví dụ object có `file`, `page`, `table`, và `row`; trong `row` có `Ticker` là MBB và `New weight` là 7,18%. Đây là ví dụ minh họa; key thật lấy từ header của bảng, thường là tiếng Việt trong bộ dữ liệu này.

Vì sao cách này hiệu quả? Thứ nhất, thực thể và giá trị nằm trong **cùng một chunk**, nên BM25 không phải nối hai chỗ. Thứ hai, header được lặp lại, nên tên cột cũng trở thành tín hiệu tìm kiếm. Thứ ba, file, trang và caption là metadata tự chứa. Thứ tư, cách làm hoàn toàn tất định, không LLM, không GPU, dễ audit.

Cần phân biệt tên gọi: trên slide gọi là **Row-JSON**, trong code là `row_json`. Đây không phải JSONL: một **dòng bảng** là một object, không phải một dòng trong file. Toàn bộ có 332 chunk trên 7 tài liệu, trung bình khoảng 491 ký tự mỗi chunk; chỉ những dòng có ít nhất một con số mới thành chunk.

**Chuyển ý:** Vậy một dòng đi từ PDF đến chunk như thế nào?

---

## Slide 12 — Path of one row

**Thời lượng:** ~2 phút

**Lời thoại:**

Đường đi gồm bốn bước. Một, `parse_unstructured.py` tạo cache element và HTML. Hai, `uelements.parse_table_html` chuyển HTML thành `UTable`. Ba, `row_json_chunks` lặp qua `table.rows`. Bốn, `table_row_json` tạo JSON cho từng dòng.

Điểm quan trọng: `row_json` **không đọc PDF và không dùng tọa độ**; nó chỉ tiêu thụ bảng đã được dựng lại. Vì vậy, nếu parser sai header hoặc gộp ô sai, lỗi đó đi thẳng vào JSON.

Phần code bên phải minh họa cách dùng thư viện: load một file cache, gọi `row_json_chunks`, và in chunk đầu tiên. Có ba lưu ý vận hành. Thứ nhất, dòng không có số bị bỏ; dòng header hoặc dòng trống không tạo chunk. Thứ hai, không chia bảng, không gộp dòng: một dòng là một sự kiện. Thứ ba, nếu `text_as_html` rỗng thì không có bảng HTML, và sẽ không có chunk.

**Chuyển ý:** Cách làm này có ưu điểm rõ, nhưng cũng có giới hạn rõ.

---

## Slide 13 — Row-JSON: pros / cons

**Thời lượng:** ~1 phút 30 giây

**Lời thoại:**

Ưu điểm lớn nhất là R@1 cao nhất trong nhóm text: 81,5% ở mức chunk và 90,0% ở mức trang. Chunk nhỏ, khoảng 491 ký tự, nên tín hiệu tập trung. Ngữ cảnh tự chứa: file, trang, caption, tên cột. Và không cần model, không cần GPU, chi phí ingest thấp.

Nhược điểm là mất cấu trúc toàn bảng: câu hỏi nhiều dòng cần parent table. Dòng không có số bị bỏ, nên có thể mất dòng mô tả. Lỗi header hoặc lỗi ô của parser đi thẳng vào key JSON. Và nó không trả lời câu hỏi định tính trong phần prose.

Tóm lại: điểm mạnh nhất là một dòng chứa đủ thực thể, giá trị và metadata; điểm yếu nhất là không có ngữ cảnh toàn bảng.

**Chuyển ý:** Vậy nên dùng Row-JSON cho những câu hỏi nào?

---

## Slide 14 — Row-JSON: suitable questions

**Thời lượng:** ~1 phút 30 giây

**Lời thoại:**

Nên dùng Row-JSON khi câu hỏi là: “Trọng số của MBB là bao nhiêu?”, “So sánh trọng số của X và Y” — lấy top-k dòng, “Giá trị của quý 2”, hoặc “Mã này có trong rổ không?”.

Không nên dùng một mình khi câu hỏi cần: “Tổng của năm mã”, “Xu hướng của cả bảng”, hoặc “Vì sao rổ chỉ số thay đổi?”. Với câu hỏi nhiều dòng, chiến lược đúng là retrieve top-k dòng, rồi mở rộng sang parent table hoặc một chunk prose. Đừng kỳ vọng một dòng có thể trả lời một phép tổng hợp.

**Chuyển ý:** Vậy kết quả định lượng và giới hạn của Row-JSON là gì?

---

## Slide 15 — Row-JSON: results and limits

**Thời lượng:** ~2 phút

**Lời thoại:**

Bảng bên trái là kết quả: R@1 81,5% ở chunk và 90,0% ở trang; R@3, R@5, R@10 đều 90,0% ở chunk và 100% ở trang; MRR 0,850 và 0,950; chars@1 là 494.

Vì sao dừng ở 90%? Độ phủ là 93,8%, tức 122 trên 130. Trong đó có 4 giá trị thiếu trong output của parser và 4 cặp thực thể/giá trị bị tách sang chunk khác. R@3 bằng R@10 bằng 90,0%, nghĩa là 8 câu nằm ngoài độ phủ; trong số còn lại, 5 câu có câu trả lời nhưng nằm ngoài top-3.

Bốn giá trị thiếu là TCB 8,75%, TPB 3.729, MBB 65.517 và MBB 24,4. Muốn vượt 90%, phải sửa parser — đặc biệt là header và merge — hoặc thêm parent-table expansion. Chỉ tinh chỉnh BM25 thì không thể lấy lại 4 giá trị vốn không có trong output của parser.

**Chuyển ý:** Sau Row-JSON, tôi chuyển sang Chipper — phương pháp có cách tiếp cận gần như ngược lại.

---

## Slide 16 — Deep dive 2: Chipper

**Thời lượng:** ~30 giây

**Lời thoại:**

Đây là phần đi sâu thứ hai: Chipper. Câu hỏi dẫn dắt là: vì sao một phương pháp làm đúng theo paper vẫn thua trên bảng số? Câu trả lời liên quan đến đơn vị chunk: Chipper giữ cả bảng, nên tín hiệu bị thô.

**Chuyển ý:** Trước hết cần phân biệt Chipper là model hay method.

---

## Slide 17 — Chipper: model or method?

**Thời lượng:** ~1 phút 30 giây

**Lời thoại:**

Có hai thứ thường bị nhầm. **Chipper** là tên một model layout của Unstructured — một vision encoder-decoder. Còn **method** trong paper là element-based chunking, tức chia chunk dựa trên các phần tử của tài liệu. Đồ án này dùng **method**, không dùng model Chipper. Vì vậy, đây là baseline tái lập được, không phải so sánh model Chipper.

Về paper: có bốn quy tắc, không có tham số, ngưỡng 2.048 ký tự; thí nghiệm trên FinanceBench với báo cáo 10-K/10-Q tiếng Anh; mục tiêu là tìm chunk size tốt mà không cần tuning. Về project: dùng cùng element stream của `unstructured`; tạo 136 chunk; bảng giữ nguyên; `chipper_prefix` thêm hai câu đầu hoặc caption; không dùng GPT-4, không gọi API.

**Chuyển ý:** Bốn quy tắc đó cụ thể là gì?

---

## Slide 18 — Chipper's four rules

**Thời lượng:** ~2 phút

**Lời thoại:**

Bốn quy tắc: một, đi theo thứ tự đọc của các element. Hai, gom tối đa 2.048 ký tự, nhưng không bao giờ cắt một element. Ba, gặp title thì bắt đầu chunk mới, title nằm trong chunk mới. Bốn, gặp table thì bắt đầu chunk mới và giữ nguyên cả bảng.

Ví dụ ở giữa slide: title và một đoạn ngắn tạo thành chunk A. Một bảng 45 dòng, khoảng 4 đến 6 nghìn ký tự, tạo thành chunk B — cả bảng. Hệ quả là bảng 45 dòng vẫn là **một chunk**, dù vượt xa ngưỡng 2.048 ký tự. Đây là quy tắc 4, không phải lỗi cài đặt.

Vì vậy, chunk của Chipper thường là “cả bảng”, và BM25 chỉ có một tín hiệu thô để so khớp. Đó là lý do R@10 cao nhưng R@1 thấp.

**Chuyển ý:** Cách áp dụng vào `unstructured` và ưu nhược điểm cụ thể ra sao?

---

## Slide 19 — Application and pros / cons

**Thời lượng:** ~1 phút 30 giây

**Lời thoại:**

Khi áp dụng lên `unstructured`, các element được dùng là title, narrative, list, table và caption. Header, footer, pagebreak, image và formula bị bỏ qua. `chipper_prefix` thêm hai câu đầu nếu là prose, hoặc caption nếu là bảng. Kết quả là 136 chunk, trung bình 557 ký tự; bản prefix là 662 ký tự.

Ưu điểm: đơn giản, không tham số, tái lập được; giữ bảng nguyên vẹn; R@5/R@10 cao; ngữ cảnh rộng, tốt cho sinh câu trả lời. Nhược điểm: cả bảng thành một chunk khoảng 2,3 nghìn ký tự nên R@1 thấp; bảng lớn vượt xa 2.048; caption và header không đủ để định vị một ô.

Vì vậy, Chipper hợp với câu hỏi “bảng này nói gì?”, tóm tắt, hoặc cung cấp ngữ cảnh cho LLM. Nó không hợp với câu hỏi “Trọng số của MBB là bao nhiêu?”.

**Chuyển ý:** Kết quả định lượng của Chipper xác nhận điều này.

---

## Slide 20 — Chipper: results

**Thời lượng:** ~2 phút

**Lời thoại:**

Bảng có hai biến thể. **Chipper** đạt 27,7% R@1, 76,9% R@5, 89,2% R@10, MRR 0,475; R@1 trang 36,1%. **Chipper prefix** đạt 36,9% R@1, 77,7% R@5, 89,2% R@10, MRR 0,535; R@1 trang 45,4%.

Chars@1 là 2.299 ký tự cho Chipper và 2.430 cho prefix, tức khoảng 4,7 đến 4,9 lần Row-JSON. R@10 gần bằng Row-JSON, nhưng R@1 chỉ khoảng một phần ba. Prefix cải thiện 9,2 điểm R@1 chunk và 9,3 điểm R@1 trang, nhưng vẫn còn cách Row-JSON 44,6 điểm.

R@10 89,2% là bằng chứng của “chứa” chứ không phải “trả lời”: bảng nằm trong top-10 nhưng hiếm khi đứng đầu. Các lỗi điển hình là bảng 45 dòng, mất caption, hai bảng gần nhau bị gộp, và số nằm trong header làm nhiễu BM25.

**Chuyển ý:** Tóm lại, Chipper là bộ xây ngữ cảnh, không phải bộ định vị ô. Bây giờ tôi chuyển sang cách chạy lại toàn bộ thực nghiệm.

---

## Slide 21 — Deep dive 3: Running the Experiments

**Thời lượng:** ~30 giây

**Lời thoại:**

Phần thứ ba là cách chạy lại thực nghiệm: parse, đánh giá, rồi ColPali. Mục tiêu là tái lập các bảng kết quả từ PDF gốc, chỉ dùng CPU.

**Chuyển ý:** Bắt đầu với môi trường và bước parse.

---

## Slide 22 — Setup and parsing

**Thời lượng:** ~2 phút

**Lời thoại:**

Có hai môi trường. Môi trường `unstr` dùng `unstructured` phiên bản 0.27.34; parse trên CPU mất khoảng 65 đến 130 giây mỗi tài liệu. Môi trường `colpali` dùng torch/transformers, e5-small, PyMuPDF và pytest. Không có API trả phí; mọi thứ chạy cục bộ.

Lệnh parse đầy đủ dùng `parse_unstructured.py` với `--force`; lệnh thử một file dùng `--only VN30 --dry-run`. Output là `data/processed_unstructured/*.json`, mỗi file chứa element và HTML bảng đã chuẩn hóa. `data/eval/gold_pages.json` chỉ cần dựng lại khi PDF hoặc gold thay đổi.

Lưu ý: `--force` parse lại toàn bộ; bỏ nó để dùng cache. `--only` và `--dry-run` để kiểm tra một file trước khi chạy cả bộ.

**Chuyển ý:** Sau khi có cache, bước tiếp theo là chạy đánh giá.

---

## Slide 23 — Run the evaluation

**Thời lượng:** ~2 phút

**Lời thoại:**

Các lệnh đánh giá gồm: chạy đầy đủ `eval_three_methods.py`; chạy smoke test với `--no-dense --limit 30`; chạy `diagnose_unstructured.py` để kiểm tra độ phủ; chạy `build_gold_pages.py` để dựng gold trang; và chạy `colpali_retrieval.py` với model ColSmol-256M.

Output gồm `three_methods.json`, `colpali_results.json`, `extraction_gap.json` và `gold_pages.json`. Có ba lưu ý. Thứ nhất, chỉ full run mới cho số cuối cùng; `--no-dense` chỉ là smoke test, vì cách gộp câu của Dual có thể khác. Thứ hai, `--limit` không đại diện cho 130 câu. Thứ ba, chỉ chạy lại `build_gold_pages` khi PDF hoặc gold thay đổi.

**Chuyển ý:** Ngoài CLI, có thể dùng trực tiếp thư viện để kiểm tra nhanh một file cache.

---

## Slide 24 — Library demo and checks

**Thời lượng:** ~2 phút

**Lời thoại:**

Đoạn code bên trái load một file cache, gọi `row_json_chunks` và `chipper_chunks`, rồi in parser, số chunk và chunk đầu tiên. Cách này hữu ích khi cần kiểm tra nhanh dữ liệu mà không chạy cả pipeline đánh giá.

Khi kiểm tra, cần xác nhận `doc["parser"]` là `unstructured hi_res + infer_table_structure`; `rows[0].text` có đủ `file`, `page`, `table`, `row`; tổng số row chunk là 332 trên 7 file. Nếu ra 0 chunk, thường là do `text_as_html` rỗng hoặc không có dòng nào chứa số.

Các vấn đề thường gặp: số liệu khác nhau giữa full run và `--no-dense`; gold trang khác phiên bản; hoặc parser khác phiên bản. Với ColPali, nếu chậm hoặc OOM, giảm batch xuống 1, dùng ColSmol-256M và 150 DPI. Cuối cùng, chạy pytest và ruff để kiểm tra code.

**Chuyển ý:** Tổng kết lại, chúng ta rút ra ba bài học chính.

---

## Slide 25 — Conclusion

**Thời lượng:** ~1 phút 30 giây

**Lời thoại:**

Ba bài học chính. Thứ nhất, **parser là trần**: 96,9% ở mức element và 93,8% ở mức chunk; 4 giá trị thiếu ở mức ô. Luôn ghi log độ phủ trước khi so sánh ranking. Thứ hai, **đơn vị truy hồi là đòn bẩy**: Row-JSON đạt 81,5% và 90,0%; Chipper có R@10 89,2% nhưng R@1 27,7%; Dual dùng cho ngữ cảnh, không phải định vị ô. Thứ ba, **visual là bộ lọc trang**: ColPali 13,8% R@1 trang cao hơn BM25 4,6%, nhưng vẫn thấp hơn Row-JSON khoảng 6,5 lần; muốn dùng thực sự cần fine-tuning và GPU.

Trong phạm vi 7 PDF này, không cần parser bảng hình học; điều cần là một parser đủ tốt và một đơn vị truy hồi khớp với loại câu hỏi. Hướng tiếp theo gồm: thêm khoảng 20 câu hỏi vàng viết tay, thử weighted RRF, thêm parent-table expansion, và chỉ chạy ColPali-3B nếu có GPU.

Tôi xin kết thúc phần trình bày ở đây. Cảm ơn thầy cô và các bạn đã chú ý; tôi sẵn sàng nhận câu hỏi.

---

## Ghi chú nhanh khi trình bày

- Nếu bị hỏi vì sao không dùng parser hình học: nhấn mạnh phạm vi so sánh là **đơn vị truy hồi**, không phải so sánh parser; mọi phương pháp dùng chung `unstructured`.
- Nếu bị hỏi vì sao Row-JSON thắng: thực thể và giá trị nằm cùng chunk, header được lặp lại, chunk nhỏ và tự chứa metadata.
- Nếu bị hỏi vì sao Chipper thua: quy tắc 4 giữ cả bảng, nên chunk quá lớn; R@10 cao nhưng R@1 thấp.
- Nếu bị hỏi vì sao ColPali thấp: đơn vị là cả trang, zero-shot, model 256M; chưa fine-tune và chưa có GPU.
- Nếu bị hỏi về 4 giá trị thiếu: đó là giới hạn parser ở mức ô, không phải lỗi chunking; muốn vượt 90% phải sửa parser hoặc thêm parent-table expansion.
