# RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

Tài liệu implement thực tế theo paper **Sarthi et al., ICLR 2024** (`arxiv:2401.18059`).

> Nếu bạn đang dùng paper *Dual-granularity chunking + dynamic context augmentation (ESWA 2026)*, thì RAPTOR là bản nâng cấp trực tiếp ở trục **chunking**: từ 2 mức cố định → cây đa tầng đệ quy có tóm tắt trừu tượng.

---

## 1. Ý tưởng cốt lõi

RAG cổ điển chỉ retrieve các chunk ngắn, rời rạc → mất bức tranh toàn cục của document dài.

RAPTOR làm 3 bước bottom-up:

```
Docs → Chunk (100 tokens) → Embed → Cluster (GMM+BIC) → Summarize (LLM) → Lặp lại → CÂY
```

Lúc inference, retrieve trên **toàn bộ cây** (leaf + summary nodes), nên một query có thể trúng cả chi tiết lẫn tổng hợp.

```
Level 2:        [Summary toàn corpus]
                  /              \
Level 1:   [Summary A]        [Summary B]
            /     \             /     \
Level 0: [c1] [c2] [c3]     [c4] [c5] [c6]   ← leaf chunks
```

## 2. Thuật toán chi tiết

### 2.1 Chunking
- Cắt corpus thành chunk ~100 tokens (~75 từ tiếng Anh), giữ ranh giới câu khi có thể.
- Code: `research/raptor/chunking.py` → `chunk_text()`, `chunk_corpus()`.

### 2.2 Embedding
- Mặc định `sentence-transformers/all-MiniLM-L6-v2` (384-dim), normalize L2 để cosine = dot.
- Paper gốc dùng SBERT / OpenAI Ada. Implement này cho phép swap model bất kỳ.
- Code: `research/raptor/embedding.py`.

### 2.3 Clustering
Paper gốc: UMAP giảm chiều → GMM, chọn K tối ưu bằng BIC.

Implement này (để chạy offline, không cần `umap-learn`):
- Nếu `dim > 10`: `TruncatedSVD(n_components=10)` giảm chiều.
- Thử `k = 1..min(10, n//2)`, fit `GaussianMixture`, chọn k có BIC thấp nhất.
- Nếu `n < 6`: trả về 1 cluster duy nhất (tránh over-split).
- Code: `research/raptor/clustering.py` → `cluster_embeddings()`, `find_optimal_clusters()`.

### 2.4 Summarization
Interface `BaseSummarizer`. Có 2 bản:
- `ExtractiveSummarizer` (mặc định, chạy offline): chọn câu gần centroid nhất của cluster làm summary. Đủ để test pipeline, clustering, retrieval.
- `OpenAISummarizer` (stub, tự cắm API key / LLM riêng): prompt chuẩn RAPTOR — *"Summarize the following cluster of texts into a coherent summary..."*. Production nên dùng LLM thật (GPT-4, Llama-3, Qwen).
- Code: `research/raptor/summarization.py`.

### 2.5 Xây cây (Tree construction)
```
level = leaf chunks
repeat:
  if len(level) <= max_leaf_per_level (mặc định 6): break
  clusters = cluster(level.embeddings)
  parents = [summarize(cluster) for cluster in clusters]
  tree.add_level(parents)
  level = parents
until giảm còn 1 node hoặc max_levels (mặc định 5)
```
- Code: `research/raptor/tree.py` → `RaptorTreeBuilder`.

### 2.6 Retrieval
Paper so sánh 2 cách, **Collapsed Tree thắng**:
- **Collapsed**: flatten tất cả nodes, tính cosine với query, lấy top-k. Đơn giản + hiệu quả nhất.
- **Tree traversal**: đi từ root xuống, chọn top-k child mỗi tầng. Tiết kiệm token nhưng dễ miss.
- Implement mặc định Collapsed. Code: `research/raptor/retrieval.py` → `CollapsedRetriever`.

## 3. So sánh vs Dual-granularity (ESWA 2026)

| Tiêu chí | Dual-granularity (2 mức) | RAPTOR (cây đệ quy) |
|---|---|---|
| Số tầng | Fix = 2 (fine + coarse) | Động, 3–5 tầng tới khi hội tụ |
| Nội dung node cha | Concat / mở rộng context | **Tóm tắt trừu tượng** bằng LLM |
| Gom nhóm | Theo vị trí / heuristic | Theo **semantic** (GMM trên embedding) |
| Retrieval | Trên 2 tập chunk | Trên toàn cây (leaf + summary) |
| QA phức tạp | Yếu khi cần synthèse nhiều section | SOTA QuALITY (+20% vs DPR+GPT-4) |
| Chi phí | Rẻ | Đắt hơn (summarize mỗi node) |

Khi nào giữ Dual-granularity: corpus ngắn, ít liên đoạn, cần rẻ.
Khi nào dùng RAPTOR: document dài, câu hỏi tổng hợp / multi-hop / domain QA.

## 4. Cấu trúc code

```
research/raptor/
  README.md          # docs (file này)
  __init__.py        # public API
  chunking.py        # cắt text → chunks
  embedding.py       # wrapper SentenceTransformer
  clustering.py      # SVD + GMM + BIC
  summarization.py   # Extractive (offline) + OpenAI stub
  tree.py            # Node + RaptorTreeBuilder
  retrieval.py       # CollapsedRetriever
  pipeline.py        # RaptorPipeline: fit/retrieve/answer/save/load
  demo.py            # demo chạy thử offline
  requirements.txt
  test_raptor.py     # pytest
```

## 5. Cài đặt & chạy

```bash
# Base (demo + test offline):
pip install -r research/raptor/requirements.txt

# Optional (chỉ khi cần scale/production):
# pip install -r research/raptor/requirements-optional.txt  # faiss / umap-learn / openai

# Demo offline (không cần API key):
python -m research.raptor.demo

# Test:
python -m pytest research/raptor/test_raptor.py -v
```

| Cài gì | Khi nào |
|---|---|
| `requirements.txt` | Luôn luôn — torch, sentence-transformers (embedding), scikit-learn (clustering), numpy, pytest |
| `requirements-optional.txt` | `faiss-cpu`: >100k nodes; `umap-learn`: clustering đúng paper; `openai`: summarize abstractive production |

## 6. Ví dụ sử dụng

```python
from research.raptor import RaptorPipeline

docs = [
    "RAPTOR builds a tree by clustering chunks...",
    "Self-RAG uses reflection tokens...",
]

pipe = RaptorPipeline()          # dùng ExtractiveSummarizer mặc định
pipe.fit(docs)                   # build cây
print(f"Total nodes: {len(pipe.tree.all_nodes())}")

results = pipe.retrieve("What is RAPTOR?", top_k=3)
for r in results:
    print(r.score, r.node.level, r.node.text[:120])

answer_context = pipe.build_context("What is RAPTOR?", top_k=3)
# Ghép answer_context vào prompt LLM của bạn để sinh câu trả lời
```

Muốn dùng LLM thật để summarize:

```python
from research.raptor import RaptorPipeline
from research.raptor.summarization import OpenAISummarizer  # tự implement _call_llm()

pipe = RaptorPipeline(summarizer=OpenAISummarizer(api_key="..."))
pipe.fit(docs)
```

Lưu / tải cây:

```python
pipe.save("raptor_tree.json")
pipe2 = RaptorPipeline.load("raptor_tree.json")
```

## 7. Tham số khuyến nghị (theo paper)

| Tham số | Giá trị paper | Mặc định code |
|---|---|---|
| chunk_size | 100 tokens | 100 từ (~tương đương) |
| embedding | SBERT / Ada-002 | all-MiniLM-L6-v2 |
| clustering | UMAP + GMM + BIC | SVD(10) + GMM + BIC |
| summary prompt | Abstractive, ~150 từ | Extractive 2 câu (offline) |
| retrieval | collapsed, top_k=20, d=... | collapsed, top_k=5 |
| max_levels | tới khi còn 1 cluster | 5 |
| QA model | GPT-4 | tự ghép (build_context) |

## 8. Hạn chế & hướng mở rộng

1. `ExtractiveSummarizer` chỉ để test — production phải thay bằng LLM abstractive, nếu không node cha không nén được thông tin.
2. Chưa có UMAP — với corpus >10k chunks nên cài `umap-learn` và thay SVD để cluster tốt hơn.
3. Chưa có reranker — nên thêm `CrossEncoder` sau collapsed retrieval.
4. Chưa có incremental update — thêm doc mới hiện phải rebuild cây.
5. Nên kết hợp với RAFT (fine-tune) hoặc Adaptive-RAG (chọn strategy) cho domain-specific.

## 9. Trích dẫn

```bibtex
@inproceedings{sarthi2024raptor,
  title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
  author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
