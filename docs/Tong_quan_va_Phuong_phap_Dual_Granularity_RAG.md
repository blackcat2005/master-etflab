# TỔNG QUAN HỌC THUẬT & PHƯƠNG PHÁP NGHIÊN CỨU: DUAL-GRANULARITY CHUNKING & DYNAMIC CONTEXT AUGMENTATION FOR RAG

*Dựa trên công trình nghiên cứu: "Dual-granularity chunking and dynamic context augmentation: An optimization method for retrieval-augmented generation" (Expert Systems with Applications \- ESWA, Elsevier, 2026/2027).*

---

## 1\. HỆ THỐNG THUẬT NGỮ CHUYÊN NGHÀNH (TERMINOLOGY)

1. **RAG (Retrieval-Augmented Generation):** Tạo nội dung được tăng cường bằng truy xuất – mô hình kết hợp trích xuất kiến thức bên ngoài vào ngữ cảnh đầu vào của LLM để giảm thiểu ảo giác và cập nhật tri thức chuyên ngành.  
2. **Dual-granularity Chunking (Phân đoạn hai cấp độ / hai mức độ chi tiết):** Cơ chế phân tách tài liệu thành hai cấp độ kích thước song song:  
   - **Khối nhỏ (Small chunks):** Đơn vị ngữ nghĩa nhỏ, cô đọng, dùng để tính vector embedding và truy xuất chính xác bằng chứng (evidence retrieval).  
   - **Khối lớn (Large chunks):** Đơn vị văn bản hoàn chỉnh, giàu ngữ cảnh xung quanh, được nạp vào prompt cho LLM sinh câu trả lời (generation).  
3. **Small2Big:** Tư duy phân đoạn "từ nhỏ đến lớn": truy xuất bằng đơn vị nhỏ nhưng khi cấp ngữ cảnh cho mô hình sinh thì lấy đơn vị lớn tương ứng chứa đơn vị nhỏ đó.  
4. **Dynamic Context Augmentation (Mở rộng / Tăng cường ngữ cảnh động):** Cơ chế xử lý sau truy xuất (post-retrieval) bao gồm tái xếp hạng ngữ cảnh kép và quyết định sao chép ngữ cảnh động.  
5. **Dual Re-ranking (Tái xếp hạng hai cấp độ):** Cơ chế tính điểm tương đồng tổng hợp dựa trên cả khối nhỏ và khối lớn đối với câu truy vấn.  
6. **Dynamic Context Duplication (Sao chép / Nhân bản ngữ cảnh động):** Lấy cảm hứng từ kỹ thuật *Gán nhãn giả (Pseudo-labeling)*; chủ động nhân bản đoạn ngữ cảnh xếp hạng cao nhất vào prompt để hướng sự chú ý (attention) của LLM vào thông tin cốt lõi.  
7. **Prompt Learning & Few-shot Learning:** Kỹ thuật hướng dẫn LLM phân loại/liên kết câu/đoạn văn dựa vào lời nhắc kèm ví dụ mẫu mà không cần cập nhật trọng số mạng neural.

---

## 2\. VẤN ĐỀ TỒN ĐỌNG (RESEARCH PROBLEMS & GAPS)

Nghiên cứu chỉ ra 2 hạn chế chí mạng trong các hệ thống RAG truyền thống (Naive RAG / Retrieve-Read) khi áp dụng vào tài liệu chuyên ngành sâu:

1. **Phân đoạn văn bản không phù hợp (Improper Chunking):**  
   - Các phương pháp truyền thống như *Fixed-size Chunking* (chia đoạn cố định) hoặc *Sliding Window* (cửa sổ trượt) thường cắt ngang xương nội dung, làm đứt gãy tính nhất quán ngữ nghĩa và tách rời các ý liên quan chặt chẽ (đặc biệt là bảng biểu, danh sách gạch đầu dòng kỹ thuật).  
   - Truy xuất khối quá lớn sẽ lẫn nhiều nhiễu (noise); truy xuất khối quá nhỏ thì thiếu hụt bối cảnh (context loss) khi sinh nội dung.  
2. **Sự liên kết yếu giữa nội dung sinh và bối cảnh truy xuất (Weak Context-Generation Alignment & Hallucination):**  
   - Dù tài liệu truy xuất có chứa thông tin đúng, LLM vẫn có thể bỏ qua hoặc xử lý sai lệch thông tin bối cảnh (do hiện tượng "Lost in the Middle" hoặc thiên kiến mô hình), dẫn đến ảo giác (hallucination) hoặc câu trả lời không bám sát văn bản hỗ trợ.

---

## 3\. PHƯƠNG PHÁP GIẢI QUYẾT & CHI TIẾT CÁC CÔNG THỨC TOÁN HỌC

Hệ thống đề xuất gồm 4 giai đoạn: **Lập chỉ mục (Indexing) [![\\to][image1]](https://saxarona.github.io/mathjax-viewer/?input=%5Cto#2) Truy xuất (Retrieval) [![\\to][image2]](https://saxarona.github.io/mathjax-viewer/?input=%5Cto#2) Tăng cường ngữ cảnh động (Augmentation)** [![\\to][image3]](https://saxarona.github.io/mathjax-viewer/?input=%5Cto#2) **Sinh câu trả lời (Generation)**.

### 3.1. Xây dựng Khối lớn (Large Chunk Construction)

- **Bước 1 (Quy tắc kết thúc dòng):** Phân chia tài liệu thành các đơn vị văn bản cơ bản dựa trên dấu ngắt dòng kết hợp dấu chấm câu kết thúc (dấu chấm, dấu hỏi, dấu than).  
- **Bước 2 (Prompt Engineering phân loại gộp đoạn):** Sử dụng LLM với Few-shot Prompting (Bảng 1 trong bài báo) để phân loại nhị phân: Hai đơn vị văn bản liền kề [![t\_i][image4]](https://saxarona.github.io/mathjax-viewer/?input=t_i#2) và [![t\_{i+1}][image5]](https://saxarona.github.io/mathjax-viewer/?input=t_%7Bi%2B1%7D#2) có thuộc cùng một chủ đề/đoạn văn không (đặc biệt hiệu quả gom các mục gạch đầu dòng [![1, 2, (a), (b)][image6]](https://saxarona.github.io/mathjax-viewer/?input=1%2C%202%2C%20\(a\)%2C%20\(b\)#2)).  
- **Bước 3 (Gộp khối lớn theo ngưỡng độ dài** [![\\gamma][image7]](https://saxarona.github.io/mathjax-viewer/?input=%5Cgamma#2)**):** Hợp nhất các đoạn văn liền kề nếu tổng độ dài ký tự của chúng thỏa mãn: \$\$\\text{Length}(P\_k) \+ \\text{Length}(P\_{k+1}) \\le \\gamma \\quad (\\text{v\\acute{ơ}i } \\gamma \= 1024)\$\$

### 3.2. Xây dựng Khối nhỏ (Small Chunk Construction)

- **Bước 1:** Tách khối lớn thành tập các câu độc lập dựa trên dấu ngắt câu tiếng Trung/văn bản.  
- **Bước 2 (Ghép câu ngữ nghĩa qua Cosine Similarity):** Dùng mô hình nhúng (Embedding model: `bge-large-zh-v1.5`) tính vector biểu diễn của hai câu liền kề [![s\_i][image8]](https://saxarona.github.io/mathjax-viewer/?input=s_i#2) và [![s\_{i+1}][image9]](https://saxarona.github.io/mathjax-viewer/?input=s_%7Bi%2B1%7D#2). Hai câu được ghép thành một khối nhỏ [![c][image10]](https://saxarona.github.io/mathjax-viewer/?input=c#2) nếu: \$\$c \= s\_i \+ s\_{i+1} \\quad \\text{n\\acute{\\hat{e}}u } \\cos(s\_i, s\_{i+1}) \\ge \\theta \\quad (\\text{v\\acute{ơ}i } \\theta \= 0.9)\$\$ Trong đó: \$\$\\cos(u, v) \= \\frac{u \\cdot v}{|u|\_2 |v|\_2}\$\$

### 3.3. Tái xếp hạng hai cấp độ (Dual Re-ranking Formula)

Sau khi truy xuất được top\-[![k][image11]](https://saxarona.github.io/mathjax-viewer/?input=k#2) ([![k=4][image12]](https://saxarona.github.io/mathjax-viewer/?input=k%3D4#2)) cặp khối nhỏ và khối lớn [![{(\\alpha\_h, \\beta\_h)}\_{h=1}^k][image13]](https://saxarona.github.io/mathjax-viewer/?input=%7B\(%5Calpha_h%2C%20%5Cbeta_h\)%7D_%7Bh%3D1%7D%5Ek#2) (với [![\\alpha\_h][image14]](https://saxarona.github.io/mathjax-viewer/?input=%5Calpha_h#2) là khối nhỏ, [![\\beta\_h][image15]](https://saxarona.github.io/mathjax-viewer/?input=%5Cbeta_h#2) là khối lớn tương ứng), điểm xếp hạng lại của ứng viên thứ [![h][image16]](https://saxarona.github.io/mathjax-viewer/?input=h#2) đối với câu hỏi [![Q][image17]](https://saxarona.github.io/mathjax-viewer/?input=Q#2) được tính bằng trung bình cộng độ tương đồng cosin: \$\$S\_h \= \\frac{1}{2} \\left\[ \\cos(Q, \\alpha\_h) \+ \\cos(Q, \\beta\_h) \\right\]\$\$ *Ý nghĩa:* Đánh giá toàn diện cả mức độ khớp bằng chứng trực tiếp (khối nhỏ [![\\alpha\_h][image18]](https://saxarona.github.io/mathjax-viewer/?input=%5Calpha_h#2)) lẫn mức độ hòa hợp ngữ cảnh bao quát (khối lớn [![\\beta\_h][image19]](https://saxarona.github.io/mathjax-viewer/?input=%5Cbeta_h#2)).

### 3.4. Ba chiến lược Sao chép ngữ cảnh động (Dynamic Context Duplication)

Quyết định xem có sao chép khối lớn Top 1 ([![\\beta\_{\\text{top1}}][image20]](https://saxarona.github.io/mathjax-viewer/?input=%5Cbeta_%7B%5Ctext%7Btop1%7D%7D#2)) thêm một lần vào prompt (đẩy lên vị trí ưu tiên số 1 và số 2\) hay không:

1. **Phương pháp Ngưỡng tuyệt đối (Threshold Strategy):**  
   - Điều kiện sao chép: [![S\_{\\text{top1}} \\ge \\tau\_{\\text{thresh}} \\quad (\\text{nghi\\hat{e}n c\\acute{ư}u chọn } \\tau\_{\\text{thresh}} = 0.4)][image21]](https://saxarona.github.io/mathjax-viewer/?input=S_%7B%5Ctext%7Btop1%7D%7D%20%5Cge%20%5Ctau_%7B%5Ctext%7Bthresh%7D%7D%20%5Cquad%20\(%5Ctext%7Bnghi%5Chat%7Be%7Dn%20c%5Cacute%7Bu%CC%9B%7Du%20cho%CC%A3n%20%7D%20%5Ctau_%7B%5Ctext%7Bthresh%7D%7D%20%3D%200.4\)#0)  
2. **Phương pháp Độ lệch chuẩn (Standard Deviation Strategy):**  
   - Tính giá trị trung bình [![\\mu\_S][image22]](https://saxarona.github.io/mathjax-viewer/?input=%5Cmu_S#2) và độ lệch chuẩn [![\\sigma\_S][image23]](https://saxarona.github.io/mathjax-viewer/?input=%5Csigma_S#2) của tập điểm tái xếp hạng [![{S\_1, S\_2, \\dots, S\_k}][image24]](https://saxarona.github.io/mathjax-viewer/?input=%7BS_1%2C%20S_2%2C%20%5Cdots%2C%20S_k%7D#2): \$\$\\mu\_S \= \\frac{1}{k} \\sum\_{h=1}^k S\_h, \\quad \\sigma\_S \= \\sqrt{\\frac{1}{k-1} \\sum\_{h=1}^k (S\_h \- \\mu\_S)^2}\$\$  
   - Điều kiện sao chép: [![S\_{\\text{top1}} \> \\mu\_S + \\sigma\_S][image25]](https://saxarona.github.io/mathjax-viewer/?input=S_%7B%5Ctext%7Btop1%7D%7D%20%3E%20%5Cmu_S%20%2B%20%5Csigma_S#0)  
3. **Phương pháp Chênh lệch giữa các khối (Inter-chunk Disparity Strategy):**  
   - Với mỗi khối lớn [![\\beta\_i][image26]](https://saxarona.github.io/mathjax-viewer/?input=%5Cbeta_i#2), tính độ tương đồng trung bình của nó với các khối lớn còn lại: \$\$D(\\beta\_i) \= \\frac{1}{k-1} \\sum\_{j \\neq i}^k \\cos(\\beta\_i, \\beta\_j)\$\$  
   - Xác định 2 khối có tính độc bản cao nhất (giá trị [![D][image27]](https://saxarona.github.io/mathjax-viewer/?input=D#2) thấp nhất). Nếu [![\\beta\_{\\text{top1}}][image28]](https://saxarona.github.io/mathjax-viewer/?input=%5Cbeta_%7B%5Ctext%7Btop1%7D%7D#2) thuộc nhóm này, thực hiện nhân bản.

---

## 4\. DỮ LIỆU SỬ DỤNG TRONG NGHIÊN CỨU

- **Tên tập dữ liệu:** Domain-specific Steel QA Dataset (Bộ câu hỏi \- đáp chuyên ngành luyện kim / sản xuất thép).  
- **Nguồn gốc:** Dữ liệu nội bộ riêng tư của một doanh nghiệp thương mại điện tử chuyên ngành thép.  
- **Nội dung bao phủ:** Thành phần hóa học, đặc tính cơ lý tính, môi trường ứng dụng, quy trình xử lý nhiệt (ủ, ram, tôi), tiêu chuẩn kỹ thuật (ASTM, GB) và nguyên nhân khuyết tật vật liệu thép.  
- **Quy mô:** **500 cặp câu hỏi \- câu trả lời (QA pairs)** chất lượng cao, được xây dựng và thẩm định nghiêm ngặt bởi các chuyên gia trong ngành thép.

---

## 5\. PHƯƠNG PHÁP & CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

Nghiên cứu sử dụng 4 chỉ số chuẩn để đo mức độ trùng khớp giữa câu trả lời sinh ra [![g][image29]](https://saxarona.github.io/mathjax-viewer/?input=g#2) và câu trả lời tham chiếu chuẩn [![r][image30]](https://saxarona.github.io/mathjax-viewer/?input=r#2):

### 5.1. ROUGE-n (ROUGE-1, ROUGE-2)

- Độ chính xác (Precision) [![P\_n][image31]](https://saxarona.github.io/mathjax-viewer/?input=P_n#2) và Độ bao phủ (Recall) [![R\_n][image32]](https://saxarona.github.io/mathjax-viewer/?input=R_n#2): \$\$P\_n \= \\frac{|n\\text{-gram}(g) \\cap n\\text{-gram}(r)|}{|n\\text{-gram}(g)|}, \\quad R\_n \= \\frac{|n\\text{-gram}(g) \\cap n\\text{-gram}(r)|}{|n\\text{-gram}(r)|}\$\$  
- Điểm [![F\_1][image33]](https://saxarona.github.io/mathjax-viewer/?input=F_1#2) của ROUGE\-[![n][image34]](https://saxarona.github.io/mathjax-viewer/?input=n#2): \$\$\\text{ROUGE-}n\_{F1} \= 2 \\times \\frac{P\_n \\times R\_n}{P\_n \+ R\_n}\$\$

### 5.2. ROUGE-L (Longest Common Subsequence)

- Dựa trên độ dài chuỗi con chung dài nhất [![\\text{LCS}(g, r)][image35]](https://saxarona.github.io/mathjax-viewer/?input=%5Ctext%7BLCS%7D\(g%2C%20r\)#2): \$\$P\_{\\text{LCS}} \= \\frac{\\text{LCS}(g, r)}{|g|}, \\quad R\_{\\text{LCS}} \= \\frac{\\text{LCS}(g, r)}{|r|}\$\$ \$\$\\text{ROUGE-L}{F1} \= 2 \\times \\frac{P{\\text{LCS}} \\times R\_{\\text{LCS}}}{P\_{\\text{LCS}} \+ R\_{\\text{LCS}}}\$\$

### 5.3. BLEU-4

- Đánh giá độ chính xác tích lũy n-gram từ 1 đến 4 kèm phạt câu quá ngắn (Brevity Penalty \- [![\\text{BP}][image36]](https://saxarona.github.io/mathjax-viewer/?input=%5Ctext%7BBP%7D#2)): \$\$\\text{BP} \= \\begin{cases} 1 & \\text{n\\acute{\\hat{e}}u } |g| \> |r| \\ \\exp\\left(1 \- \\frac{|r|}{|g|}\\right) & \\text{n\\acute{\\hat{e}}u } |g| \\le |r| \\end{cases}\$\$ \$\$\\text{BLEU-4} \= \\text{BP} \\times \\exp\\left( \\sum\_{n=1}^4 w\_n \\log(p\_n) \\right) \\quad \\text{v\\acute{ơ}i } w\_1 \= w\_2 \= w\_3 \= w\_4 \= 0.25\$\$

---

## 6\. CODE THỬ NGHIỆM CÓ KHÔNG?

### 6.1. Tình trạng công bố mã nguồn của bài báo

- **Không có link GitHub chính thức đi kèm bài báo:** Nhóm tác giả không đính kèm repository mã nguồn mở riêng cho kiến trúc này.  
- **Framework nền tảng:** Bài báo nêu rõ toàn bộ thí nghiệm được lập trình và chạy trên framework mã nguồn mở **FlashRAG** (Jin et al., 2025): `https://github.com/RUC-NLPIR/FlashRAG`.  
- **Phần cứng thử nghiệm:** Chạy trên 01 GPU NVIDIA RTX 6000 (24GB VRAM).  
- **Mô hình sử dụng:**  
  - Embedding: `BAAI/bge-large-zh-v1.5`  
  - Sinh phản hồi (Generators): `LLaMA-3-Chinese-8B-Instruct` và `Qwen2.5-7B-Instruct`.

### 6.2. Code tái hiện lõi thuật toán bằng Python (Implementation Core)

Dưới đây là đoạn mã nguồn Python hoàn chỉnh tái hiện chính xác 2 công thức cốt lõi của bài báo (Tái xếp hạng kép & Sao chép ngữ cảnh động):

import numpy as np

from sentence\_transformers import SentenceTransformer

\# 1\. Tải mô hình Embedding (khuyến nghị bge-large hoặc bge-m3)

model \= SentenceTransformer("BAAI/bge-large-zh-v1.5")

def cosine\_similarity(v1: np.ndarray, v2: np.ndarray) \-\> float:

    return float(np.dot(v1, v2) / (np.linalg.norm(v1) \* np.linalg.norm(v2)))

\# 2\. Cơ chế Xếp hạng lại hai cấp độ (Dual Re-ranking \- Công thức 2\)

def dual\_reranking(query: str, retrieved\_pairs: list) \-\> list:

    """

    retrieved\_pairs: list of dicts \[{'small\_chunk': str, 'large\_chunk': str}\]

    """

    q\_emb \= model.encode(query)

    reranked\_results \= \[\]

    

    for pair in retrieved\_pairs:

        s\_emb \= model.encode(pair\['small\_chunk'\])

        l\_emb \= model.encode(pair\['large\_chunk'\])

        

        sim\_small \= cosine\_similarity(q\_emb, s\_emb)

        sim\_large \= cosine\_similarity(q\_emb, l\_emb)

        

        \# S\_h \= 1/2 \* (cos(Q, a\_h) \+ cos(Q, b\_h))

        s\_score \= 0.5 \* (sim\_small \+ sim\_large)

        reranked\_results.append({

            'small\_chunk': pair\['small\_chunk'\],

            'large\_chunk': pair\['large\_chunk'\],

            'score': s\_score

        })

        

    \# Sắp xếp giảm dần theo điểm số S\_h

    reranked\_results.sort(key=lambda x: x\['score'\], reverse=True)

    return reranked\_results

\# 3\. Ba chiến lược Sao chép ngữ cảnh động (Dynamic Context Duplication)

def apply\_dynamic\_duplication(reranked\_results: list, strategy="std\_dev", threshold=0.4) \-\> list:

    """

    Quyết định nhân bản khối lớn Top 1 nếu thỏa mãn tiêu chí

    """

    scores \= \[item\['score'\] for item in reranked\_results\]

    top\_score \= scores\[0\]

    duplicate \= False

    

    if strategy \== "threshold":

        if top\_score \>= threshold:

            duplicate \= True

            

    elif strategy \== "std\_dev":

        mu \= np.mean(scores)

        sigma \= np.std(scores, ddof=1) if len(scores) \> 1 else 0.0

        if top\_score \> (mu \+ sigma):

            duplicate \= True

            

    \# Nếu thỏa mãn điều kiện, nhân bản khối lớn Top 1

    final\_contexts \= \[\]

    if duplicate:

        final\_contexts.append(reranked\_results\[0\]\['large\_chunk'\]) \# Bản gốc vị trí 1

        final\_contexts.append(reranked\_results\[0\]\['large\_chunk'\]) \# Bản sao vị trí 2

        for item in reranked\_results\[1:\]:

            final\_contexts.append(item\['large\_chunk'\])

    else:

        final\_contexts \= \[item\['large\_chunk'\] for item in reranked\_results\]

        

    return final\_contexts

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAICAYAAADwdn+XAAAAK0lEQVR4XmNgIA38RxcgFWxCFyAHUOwKEAAZMg1dkBRAkQEUeYHiQMSwHQAEVQiL7X2FMwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAICAYAAADwdn+XAAAAK0lEQVR4XmNgIA38RxcgFWxCFyAHUOwKEAAZMg1dkBRAkQEUeYHiQMSwHQAEVQiL7X2FMwAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAICAYAAADwdn+XAAAAK0lEQVR4XmNgIA38RxcgFWxCFyAHUOwKEAAZMg1dkBRAkQEUeYHiQMSwHQAEVQiL7X2FMwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAMCAYAAAC0qUeeAAAAYUlEQVR4XmNgQAX/0fh4AdGKdwHxZXRBdNAAxPsZIKaCaBAmCIh2AgjgUjwViBWQBS4C8RJkAXwA2dQzSOz3DFhsRBYwRWKDAIZiBahgAqowGBAMThj4BqWPoIjiAChOAADL5RZnfOyVIwAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAANCAYAAABcrsXuAAAAjklEQVR4XmNgQAX/0fhUB7wMdLDkDhC/RhekFgC5Hh3TDNDUcBjAZckrdAECAGdoXAHibeiCBABWg6AAqxyyYCwS+yUQv0XiIwOsBkEBVjlkQXQF6HwYwCUOAljlVjFAJNLRxMWAuBrK5gbiBiQMUo/MRwZYLcEFYIr3o4hCAD6D8MlhAHyKyZUjCRBtEACM0SrJHJnOwQAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE8AAAAQCAYAAABX7CKkAAAB7ElEQVR4Xs2YPUsDQRCG51+IhWBtJRaKXfAfCLZWgpWt/YGgYCGChZVY2IkgWImIKUTwB1haaqUi2Fnpzd3uZe9lZvbDCHlgSeaZD+9iLtkc0WTxgyLCJ4oC5lAoDFBI5J6A553a3nNMJHKHIpFFFBkcQ/xI7TmsgmfmUXgGNDrxkhfvLXi+T2UzKhSJhH87B+0YNc9YuYZoAcD194L7AGfxhSKTXRQJXKNwWOc/hQKxmiW4HnskZ5FTK5Hbf4vCsU3xWWbeTCbCMw5QGnyjCOBZs+5ROzbNa2j17LfqtVmvpXrd9NMNWm+DmUxgg/JmrNXrCKUD52Ds0byGVo8eY0ZyHWYygdz+IcnbAJ6DszD28IwctDnoMWYk12EmI5T0Wi9eyJngPON48fhSRY8xI7kOM2mAfRhraJct9nO8AM6DtTGket5wXwRxRXKd5Dq0pOYZKXcaPI9tnKUvDJzp472ebcFa5hVFgFTPX3DrQSzVMKJ/ojbh17Cfbpy0n3qgfp9fM0GNdxpS7oRGl7M/NubZPYZg/6HgQi5ROHyP1WvlTPhzoZQXFAF//Y26gyIBbZNsEd0k/xfWZcRUKBLRfp5NowBK3kElPQ3FjTUrKATGfWPgCoUA3hiwUG8MTAq5/6DYuzmF1FtSyyh+AUJrkgxUmG0aAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAJCAYAAAAPU20uAAAAM0lEQVR4XmNgQID/UFyPJobKQGODwVF0ASjAUIgMtNAF0AFe3SDQjS5AEpBFF0AHf5A5ACdiCgxU3pbzAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAJCAYAAAAGuM1UAAAAT0lEQVR4XmNgQAX/kTBBgKzoMRIbJwBpeIguiATOoAuAANFOQteNrEECjQ8GN9H4T9H4GBrcoYIgXI8mBwIYGvCBTij9GkUUD9AC4vkwDgC47BYouS6ZxgAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAJCAYAAADDylfFAAAAc0lEQVR4XmNgQAX/gTgNiU1TcAeJTXPLQBaAcCW6BBD0owvgASBHO0AxXjCfAWLhJXQJEoEDFGMAUDyhB9sWJHYDA6Y8IeAAxRjAHY2PzWBksQYcOBQiDQYOUEwy6EQXIAI4QDHJAJtPCQEHIM5FF6QbAADQORh5hxtKlgAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAHCAYAAADEUlfTAAAALklEQVR4XmNgQAX/gdgNiJOAuAFdApkN56Nw0AFOCRAgWRIuxgnlmALxEwaoSwGHaQ3Ad64iRAAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAALCAYAAABCm8wlAAAASUlEQVR4XmNgQID/UKyAJIYBQArwAsoU9DMg7IcpfASlUQSR6U4oGy6A1wqQ5EQojQGaGBASb5ElYAAkyYzExgDIgkJALAkTAwBXcxZXshuFGQAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAALCAYAAAD4OERFAAAAm0lEQVR4XmNgQID/UCyPJEZPALIbK8ApQQeA026cEjQGrxlw2D2NAYcEjcFFKI3VbpCgKJT9HIj5oGK0BqxQGqtdMEEYPR+JjQ6yGBAZgxDGB5DlsaolxhBqglVofAy7O6GCE6E0PYANGh/DXpCACZT9EogfQ9n7oTQ6oFb0IQMMtcgCIHY9lG2HJE5rgNdRINCARYxWAD1k/wMAiJNBuReTG8kAAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE0AAAAUCAYAAADIiLCPAAAB9UlEQVR4Xu2WMUsdQRCAt9PGxlIQrCSKWKRJq3YSsBXTpLBRIWARO4uzsAvaWOYXCCKIrYUBIZWWamehYiGITbBSd/JuYfze3N3evTvFhx8Mt/Pt3uzuvXM957qfPh9PlB8U8/HQKnBJ0QTv7Zf5S6GYS69/fPS6hvbWSNGSyBrWfVyxI4cNihSpdUNZJ5MUb0Ci2ps+vqg8j88UKeElaOxlaKxwJJx/28dPuDx4v3CcXq2+WvhO8cr0Iy+7UY4fV20599jfMUUFr11rzB07amI+vR651hm0rPpi6fGxRtkkeQ+NfSH/+sJ2BucY9TEEFwPrNErWZJYPjn0TPqbgYjmkcO31mVvEjKkNa7J9Z3vLCVm+KqzH3CJmTG1Yk4nL8g+Uzh4bQzjPCOtdILfgPdOGK8sPioBV+NbHCaWzxwra8z7+Z9RY9egWlMv7WL2ncO21ypJQBH5TpOgJB33sKsfFzKr2qmoLMpbjAweudRQEZNw3lQdntTUrFCmPFCVJKDTWYZwFv8D1r3+u2pqszQbGKBQxD83y8ucpnzGC1R9DQqGpWlSouqkligxOVXvLx4zKA3sULm5dRSQUJKGIhIs7U7kwgjwQs5FPyK17LCdwXVVIKLqBRQrFP9XecflHQBYJRTcz4WNA5b98DKu8CHkzdfznGZ/Vf9dsIzWmAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAJCAYAAADQHRcxAAAAfElEQVR4XmNgQAX/gfgoEBtD2WQDdM3mSGJXkSWgwAFdAAZAmtANAwGYWCSKKAODIhD7o4nBAUiTKrogA2FLsAJckvjE26C0GpocVk3IrtJClmBAVY9NL1jwLJQOgYqtAOL9DKiGhQKxOBL/MRKbZIDNVU+QxEgC2AwDAwBSPyLUUWjcogAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAPCAYAAADtc08vAAAAn0lEQVR4XmNgwA7+A7EblCYZoGsqR+PjBU/RBYDgDroAPtCJxj+KxicKpAHxfCDWYsD0Dl4gDaXNgdgLiIuB+CxCmjD4gy7AgOkCdD4KwCaJLobORwH+6AIMmBrQ+XDAgS4ABFuA+DeaWAaUxjDoOxZBdH4eEhtdDi4AokFYB0kOBpA1YRhgiC6ABSBrigNibyQ+UQDd1lwY4z2yKKkAAPg/I/OBMPOtAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAALCAYAAABCm8wlAAAATUlEQVR4XmNgQID/UIwXEFSwBl0AGbxHF0AHIOP3ALENlI0BkAVnAvEKJD4YICsAsYWR+GD7dyDxMaxAF4Dx4eLoChrQ+BgKQPyzIAYAZCUVzXhQGg0AAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAPCAYAAADQ4S5JAAAAd0lEQVR4XmNgwA3+Q/F8dAl0AFLkhUUMK8AlkcuARQ4kEIsuiARQNCxCF8ACUORBnLvIAlgAhgZ84BsDiRpA8o/QBXABOwYs8iABVST2dDQ5rABdAhTE6GIY4DUDRBEMw8BTJDZOANOQDsSzkCVwAWw2EQSB6AIA6hcnZstUhBMAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAJCAYAAADQHRcxAAAAfElEQVR4XmNgQAX/gfgoEBtD2WQDdM3mSGJXkSWgwAFdAAZAmtANAwGYWCSKKAODIhD7o4nBAUiTKrogA2FLsAJckvjE26C0GpocVk3IrtJClmBAVY9NL1jwLJQOgYqtAOL9DKiGhQKxOBL/MRKbZIDNVU+QxEgC2AwDAwBSPyLUUWjcogAAAABJRU5ErkJggg==>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAPCAYAAADtc08vAAAAn0lEQVR4XmNgwA7+A7EblCYZoGsqR+PjBU/RBYDgDroAPtCJxj+KxicKpAHxfCDWYsD0Dl4gDaXNgdgLiIuB+CxCmjD4gy7AgOkCdD4KwCaJLobORwH+6AIMmBrQ+XDAgS4ABFuA+DeaWAaUxjDoOxZBdH4eEhtdDi4AokFYB0kOBpA1YRhgiC6ABSBrigNibyQ+UQDd1lwY4z2yKKkAAPg/I/OBMPOtAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAQCAYAAAB3AH1ZAAAA4ElEQVR4XmNgwA5+AfEiIP6PLkEPgG4pOp+mIA1dgIHODsBmGTYxmgItIN4OxBIMdLY8CIntAKXp6gBslmEToxTgNPMmugADRLE+Er8BiU0uwOkAbABdcQManxyAbiYYuKELAIEPA6biBjQ+TL4BSWwmEP+Asp2AuBxJDgTQzQQDkCCyhBAaHwYakNjo8tgcAwK41KGAzVB6GhBPRJZAAw1IbHSDKHIAsaAfid0OxCpIfLId4IAugAeANO9C4tcwQPS/RRJrgNIJDJiWwaIaRTwBmUMF0IAuQG/wlIEERwAAk2c5u+hhfsQAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI8AAAAPCAYAAADd05xLAAADLElEQVR4Xu2Xz6uNQRjHp0jdBbFRfiyIhSI2t4idHQtLYuFNWcpGKZK7sUGy4B8QSysbdbdKlFJEWSilpNz8SFHI8XzPzNf5es7zHueczq1za059mpnn18wz87zzvicl+3U6adboVCpDMou66f4wCAwqlTZq8VTGphZPZWxq8VTGpqV4zmLYZzwcywLZOJwzLgfyaeaZEe1bJJsU143FiH/JGBzXFc95UQ52zMwEskkybcXTBDLPMPs2CT5Lf7HmHBz3P8WzxgXYZuwxfhgfiu6d6P2ECykusCPGW2N5GR9MuVDgq/aQ/SryKD59YYO1qv6C8cS4YuwvcuqQA3R+PsgBD4b5sT1sfDJ+G0+LnL7I5b6TgS8i41wYY11qdzz19gA8LC32Cfr1xhtjbRljLdx72GHs56YO+aC9l/KbIVr3USe7KHZRvn3F4ydFe6e0KBgEQb9xNsqW0p40vgV6xHhQ+i+Nm6XPW0ZjUnaq+KGvT5zae39sLAoEfRQPWsyFfL6KDofxU/w0xjrjtcgbkaNFDMjmAl/Fr7HNjvrVIuNDscv5NWJD2cfUy4vwPNSO8+h6dqa8x5sDe59v1gfFc9o5oloZbEdpGSDaAF3shsAG8TRB2vvDVxnlLDQPdH5zo+IBTfq3eLAe3i4aDy3WpjcrfNGuEJna+76X+Xy8HfUsGO3j1le/Rmwo07zIoOLReXguUS4+30xQPHPOkcXjD0BtFF8M71N+ItRmVWm/p5yw2mtM3ey9Kd9Afj7EVz+2d1NvI7mm5yk/XZiTeR0oLdekMdqKB3q8KhnzmDHvfBV/aG120aGyv9v5NWIzbvHg1Xi79HlpYB/4xtA91XxzvKB4OCFEmKSrcsH4FFwzXv0N1rMBV42VxhmnB/yOwfeA+rDy+a3F+QbRFDsmh5ibUj50yvjNs1DG2OTHRcbXIThkbJWxXwO/0cBG45aMTxR/feWAJuUYbAH3V2916ngLKtDjZn2Rcm70wc3OM2JMXa/GJdxj2j1K/f+Qtxs3jH0i8/m2Fs804g9lFPTWBNETWhmVJVI8M4FsWPiU8fWoMm9bGYUlUjyVaaRbPH8AGBGfExuehtwAAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAJCAYAAADQHRcxAAAAkklEQVR4XmNgQAV9QPwUTew/Gp9oANLIjcS/C8TTkfjI4AwQ34CyVZAlYADdFeh8EGAHYhkk/nokNgpA14zOB4HPaPwGND4YHAPib0j8YAaEYaZI4pJQcRB2QRJHATAFIBAIxIZA/BiID8NVYAL0MIYDbF5CBxFo/HQ0PhiEMhBnGEiNERL/PhIbDkCK+NAFSQUAnMsd9+qsB4oAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAJCAYAAAA7KqwyAAAAYUlEQVR4XmNgQAX/sWCiAbJifBr5gNgZXRBdAzofBpYiseFqFJA5UIDOxwbgavKA+DWSBAjgMgAknowuCALE+h8EOhlwqNkJxBPRBXGA6+gChIAnGh+rC/ABkAZLKI2iGQA97BvpAclliwAAAABJRU5ErkJggg==>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFgAAAAPCAYAAABgHgh7AAABwklEQVR4XtWYMS8EURDH50NoJCpKpYhC6xMInciFTqIRjcopRCPXiEKhUykkCtrTSKgU1ESlk1BKhBv33u68/8089+xz1i95uZ3f7P1vdu/2sndEOh9iMcOilwPMPxK9uvNM4ezJ4BObiqsCZlUadsDgnBOKi2LtbPlUrJx3FDXEmt3yKtbO+yh+iJU/iqKGWLPfoojBIccoM8L5jyj/CTz7HcpU/PehXydhuzI8oMxfDtu15p7C2S/Ddho3VAbtQY+xLpd+eaEyfxp6D511Dq5OXFA5+3XYonZnvYKL4oM8a8LnAPMbYjvXa/wW/mqU7EBdsIXCsUC9IYzmYgyhcHAOv+uy1rb/knEUjhHqnRHrAquR6i3eUDhiObHeILHmYM/3weiYU7FdNM6koO7dRCwcYad5RutxvQHOc4WC9AyP1uN6BWrcx5PaW1Icw24TJTPmHn0Yr8my3YMWziyicKy7R5lvMYvCMU/dX5UWM1CvQv0dUygc2uxzZbvAH3vs2PrGCtlFkRnrdXNQdXY/G/9v06DyhiCJZmc9UTfsMGx9cYAiEfkp0U4mfuflJMfsHr713BZ1FlooMqOd8Fy0UeTiEygQk4MDIvVVAAAAAElFTkSuQmCC>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHAAAAAQCAYAAADOFPsRAAACQklEQVR4Xu2YP2sUURTFL2gjKCJIwCaVlX4A29vZapFeMUIaBRsJwWJJI7YiiEUqRdFGEERMAtvYqr1FAomV+hUk3sPsS+6ceW/mzUx2M8L+4DBzz33z/t1hZnZF4hyQwP2j9JwhEwrmiXlzBggKdYZNmRewDcrGLEkVKuXPgi052fHbomzMkvDOO8WJAfBYirktciKTTalu7hrFx4GykcGC6ZVp7PSx1KIF/uNliHf+DSnmhWMbeC2PIt5xoGw0EPZZ6Rzqxa5Mr4gXTQ/ZbAnuWsxtnRMJeB2IV8kLbEyOV6R6XRPKRg0/Tb/Jqxvv+eS4JPXtSnBDjrugphF5Xcm5yS5JtQ3HAfY59oSxm5Qilot5gH2Ok3BDjrug0r+AmMcOmwn+mj6Rl1oH+0pxE8pGgrNSHetcxAuwf9kHb33g+GF6Qh535OOcc6DSvYDoa5vNBnDNVRcvTzwwdj6AH9QFZSPBaamOwbGndl6xxM2IB7yHZ/I1F/ucP39geuZilXYFPC9Ffy84kQmu9e+7kemri2O8kfj6m1A2avD942kycnGK6LzeT47Y6FDM60fpElykCxTHzjlWyZsswHV9P3jQx23TZ+fhZwk+BurgNeSgbDRwV4p9b0OXeR3iL8ZXWvgyAqkCjqVoG1DJL2Bffpm+sZnJHhsD4TsbbUBhnrr4jumlad95AO1em+5J0cb7XtMmd4w/Uvx8wJ/2t0xfStmTY9f0Top9XDF9KKenR+7GTZuhzOO/AncKNq7pHTOnJ/8ADXSqRax/GCwAAAAASUVORK5CYII=>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAPCAYAAAA71pVKAAAAlklEQVR4XmNgwA7eAHEGECcD8Wc0ObzgPxr/HhBzoIkRDUCGRaILYgPotoIANjGsYAKUXgzEUQwkaEQGWVCaaM0H0QWAwIUB04BqND4YoCsCAUcGTPEEND4YwPyLDE4xYGrGCmTRBRiwa8QQ24pFEMR3RuMj03CALAHD2MAZdAEQmIIugAPgMpQoANIMyixwsACZQwoAAGuVIr7weq3hAAAAAElFTkSuQmCC>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAALCAYAAACksgdhAAAAc0lEQVR4XnWOuQ2AQBADXQJdUAIdEBMiqqEDQgKKISKkGgqAXWTxnOyRHOx47wH+9JE1ckaWyMjknLG4hQ7a32Sxl5Jk15QysbfB/0JLIg/VSn7IblJyKyVpYS6UksivJVIGA0xXwRQwfsb7/MisnI9ni1ybECVf5ZJSNwAAAABJRU5ErkJggg==>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAQCAYAAAB3AH1ZAAAA4ElEQVR4XmNgwA5+AfEiIP6PLkEPgG4pOp+mIA1dgIHODsBmGTYxmgItIN4OxBIMdLY8CIntAKXp6gBslmEToxTgNPMmugADRLE+Er8BiU0uwOkAbABdcQManxyAbiYYuKELAIEPA6biBjQ+TL4BSWwmEP+Asp2AuBxJDgTQzQQDkCCyhBAaHwYakNjo8tgcAwK41KGAzVB6GhBPRJZAAw1IbHSDKHIAsaAfid0OxCpIfLId4IAugAeANO9C4tcwQPS/RRJrgNIJDJiWwaIaRTwBmUMF0IAuQG/wlIEERwAAk2c5u+hhfsQAAAAASUVORK5CYII=>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAJCAYAAAD+WDajAAAAQ0lEQVR4XmNgQAX/ofQbJDaKBAhwI/NBDGTJL0DcCOOAJIIQcriNxODLAvEqKHs+A6ZiOABJlME4IB1XEHK47UORAAA7zhUK/AapAQAAAABJRU5ErkJggg==>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAHCAYAAADEUlfTAAAAMElEQVR4XmNggID/QLwHyv4BxAJQMTADBK7ABKA0mH0KSQAnIE/yPgMeSZCEDrogACbBDnSrsrYVAAAAAElFTkSuQmCC>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAANCAYAAABLjFUnAAAAoElEQVR4XmNgQAVaQLwTiP8D8RcgboDi51AxI5hCUgBIIzaASxwnqGDArQkkroMuiA+ANGAzbDoDdnG8AKTBFE3MFSpOMoC5DIZPAbE0igoiwUEGMl2ADeAKL7IAyCA9dEE0MB+IZwLxOiB2AOIlKLJQEMRA2FUJUBpZHYaeTUA8mQGS2kG0H4osKgDlEuS0thKJTTLA5ioM1xELsBkGBgDI5imA4DpeSgAAAABJRU5ErkJggg==>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAANCAYAAACpUE5eAAAAlElEQVR4XmNgQAX/8WCyAS4DsIkRBUAad6MLMkDE96MLEgNwuQSXOF7gxICpMQiLGNEAFn4NUAxir0CSJxmADKjBIpaGJkY0wOY1kNh8dEFigAEDbgNB4YgMnkDFm4FYDcrGALDwQwcgMXEo+zqaODY2QaDFANGwEU38BBJ7LxKbLFCMxD4OpUlyJTpA1hwKxDkwDgAYzC9tPV7eUgAAAABJRU5ErkJggg==>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAANCAYAAACgu+4kAAAAc0lEQVR4XmNgQAX/CWCiAEihO7ogA4kGYAO96AK4AC4DcImjAHS/7kdiEwVAmj8xQDSiG0YUAGlQQeMjA4KG4pWEApxqZBjwSCIBnGpAEvfQBbEADAO8gHgVEHcD8TQg3ogqjQEwDCAVUGRAHwPEgKUwAQCKoic9DEjHwAAAAABJRU5ErkJggg==>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAHCAYAAADam2dgAAAASUlEQVR4XmNggID/UAxjqyHxwYAPSiMrBIE+JDbDKSiNohMLHwzQBdH5YIAsaIXEvw8TnA/EBTAOA0TBTiQ+XBAZNABxAgNUHACiYBQ/JVyAEAAAAABJRU5ErkJggg==>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAQCAYAAABJJRIXAAABnklEQVR4Xs2WMUoEQRBFO9TUQFAMFgMREwNBEAwEI829gKcwHBAzIz2AIIYieAITAwMDExHBA5gbGGrVThdb8/3T3ejMuA+K7fr9q7a72GUmhOniFoWh+CJx13D85DDUvqOYn0i8xLWvfZB4jWur0WB8oNATiyh4Si6vpC6CPZiPaU8o9Aw7wxi8AGMjJBoIZ2HS403iwu0ZrJ5pfbKKglEyhNSvwLAex4F7TyE/kJgFbQjY2Tobgsf8GkuwZ+T66f4ofua8yrzEQqi9OxJVDIT2Kh3CPYoZ/CA01prb/DAR3MOcYR7vZXVMKx4CLS6E1WNuoHcO8hw5L91vG8KuW+PBGNfx87GhTtB6/2to64d62/kYzxLvKALYf0zbl3jzFuQp2nyqjyBnoI55ihIv9ZQMwXLUDK/runK5gbWYG6jnck9qz2h47FKpQM5DrevjTVmJuUfzZdB9jbEnMQOasi1xKXEjsR54/wo0A72MEs+g5A6k++yptIlCIa0vS/8JvjbjUDD/K13364xPt8a/0L7Ljd9eBN9Tpo4rFIbgGzLbmZESZkfNAAAAAElFTkSuQmCC>

[image36]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAALCAYAAACQy8Z9AAAAhklEQVR4Xr2PsQ2AMAwEvQabUVBQMgYDMBAdBUswAQOwAOElB4zzMR0nfZGz+QQRkeXKGcTiZzad2bvBABdYZvWemis8KwXwA3GeXYiPSj3MHUI8K4WbnAPFx/Lx+6NmVcfIu8im5/a1obCXAlbMHCUq9SX+XOX30p64EBTl17A0z2oxo+UJcsROwjZb+UIAAAAASUVORK5CYII=>