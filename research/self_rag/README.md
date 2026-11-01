# Self-RAG — Learning to Retrieve, Generate, and Critique through Self-Reflection

Implement thực tế theo paper **Asai et al., ICLR 2024 Oral Top 1%** (`arxiv:2310.11511`).

> Nếu paper *Dual-granularity + dynamic context augmentation (ESWA 2026)* làm dynamic theo heuristic (lấy thêm context xung quanh chunk), thì Self-RAG làm dynamic theo hướng **trainable + có kiểm soát**: model tự học *khi nào cần retrieve*, passage có *relevant* không, câu trả lời có được *support* không.

---

## 1. Ý tưởng cốt lõi

RAG cổ điển luôn retrieve top-k cố định → tốn kém với câu dễ, thiếu với câu khó, và dễ hallucinate khi passage nhiễu.

Self-RAG chèn **reflection tokens** vào quá trình sinh, mỗi segment đi kèm 4 loại token:

```
Query → [Retrieve=Yes] → retrieve D → [IsRel] lọc → sinh candidate/segment
      → [IsSup] kiểm groundedness → [IsUse 1-5] chấm utility → rank → chốt segment
      → lặp cho segment tiếp theo hoặc dừng
```

- `Retrieve`: Yes / No / Continue — có cần retrieve cho segment này?
- `IsRel`: Relevant / Irrelevant — passage có giúp trả lời?
- `IsSup`: Fully / Partially / No support — segment có được passage chống lưng?
- `IsUse`: 1–5 — segment hữu ích tổng thể ra sao?

Inference dùng soft re-rank (cộng điểm) hoặc hard constraint (bỏ passage No-support).

## 2. Training trong paper (để hiểu, không cần chạy lại)

1. **Retriever** có sẵn (Contriever).
2. **Critic**: prompt GPT-4 gán reflection tokens cho dữ liệu instruction, rồi distill vào model nhỏ.
3. **Generator**: train LM (Llama-2 7B/13B) bằng next-token prediction trên dữ liệu đã chèn reflection tokens + retrieved passages (passage bị mask loss).
4. Inference: segment-level beam search, re-rank bằng xác suất reflection tokens.

Kết quả paper: thắng ChatGPT và Llama2-chat+RAG trên OD-QA, reasoning, fact verification; citation accuracy cao hơn cho long-form.

## 3. Implement offline trong folder này

Vì không train lại Llama-2, folder này xấp xỉ Critic bằng embedding + lexical (minh bạch, chạy không cần API key):

| Module | Vai trò | File |
|---|---|---|
| Schema token | `Retrieve/IsRel/IsSup/IsUse`, `SegmentCritique.total` | `reflection.py` |
| Retriever | dense cosine, in-memory | `retriever.py` |
| Critic | `decide_retrieve` (rule) + `score_relevance` (cosine) + `critique` (cosine + word-F1) | `critic.py` |
| Generator | `ExtractiveGenerator` (chọn câu giống query nhất) + `LlmGenerator` stub | `generator.py` |
| Pipeline | adaptive retrieve → filter → draft song song → rank → citations | `pipeline.py` |

`SegmentCritique.total = 1.0*rel + 1.0*sup + 0.5*use`, tương ứng trọng số relevance + groundedness + utility của paper.

Production: truyền `call_llm(prompt)->str` vào `LlmGenerator` / `LlmCritic` để thay heuristic bằng LLM thật (GPT-4, Llama-3, Qwen) với prompt kiểu paper.

## 4. So sánh vs dynamic context augmentation (ESWA 2026)

| Tiêu chí | Dynamic context (heuristic) | Self-RAG |
|---|---|---|
| Khi nào retrieve | Luôn retrieve top-k | Học `Retrieve=Yes/No` theo query |
| Lọc nhiễu | Không / rerank nông | `IsRel` từng passage |
| Chống hallucination | Không kiểm | `IsSup` từng segment, bỏ No-support |
| Long-form | 1-shot concat | Sinh từng segment, critique từng đoạn |
| Citation | Không | Có `[1],[2]` kèm trace |
| Chi phí | Cố định, cao với câu dễ | Rẻ với câu dễ (skip), sâu với câu khó |

## 5. Cấu trúc code

```
research/self_rag/
  README.md          # file này
  __init__.py        # public API
  reflection.py      # schema token + SegmentCritique
  retriever.py       # InMemoryRetriever (cosine)
  critic.py          # EmbeddingCritic + LlmCritic stub
  generator.py       # ExtractiveGenerator + LlmGenerator stub
  pipeline.py        # SelfRAGPipeline: answer/fit/trace
  demo.py            # demo offline
  requirements.txt
  test_selfrag.py    # pytest
```

## 6. Cài đặt & chạy

```bash
# Base (demo + test offline):
pip install -r research/self_rag/requirements.txt

# Optional (chỉ khi thay heuristic bằng LLM thật):
# pip install -r research/self_rag/requirements-optional.txt  # transformers / openai

# Demo offline (không cần API key):
python -m research.self_rag.demo

# Test:
python -m pytest research/self_rag/test_selfrag.py -v
```

| Cài gì | Khi nào |
|---|---|
| `requirements.txt` | Luôn luôn — torch, sentence-transformers (retriever + critic), numpy, pytest |
| `requirements-optional.txt` | `transformers`: LlmGenerator/LlmCritic local; `openai`: distill kiểu paper qua API |

## 7. Ví dụ sử dụng

```python
from research.self_rag import SelfRAGPipeline

pipe = SelfRAGPipeline().fit([
    "Self-RAG predicts Retrieve Yes/No, checks IsRel, verifies IsSup...",
    "RAFT fine-tunes with oracle and distractor documents...",
])

res = pipe.answer("How does Self-RAG decide when to retrieve?", top_k=4, max_segments=2)
print(res.answer)   # "... [1] ..."
print(res.trace())  # Retrieve=Yes, rel/sup/use từng segment

# Chitchat -> skip retrieval:
res2 = pipe.answer("Hello")
print(res2.retrieve)  # Retrieve.NO
```

Cắm LLM thật:

```python
from research.self_rag import SelfRAGPipeline
from research.self_rag.generator import LlmGenerator

def my_llm(prompt: str) -> str:
    return your_client.generate(prompt)

pipe = SelfRAGPipeline(generator=LlmGenerator(call_llm=my_llm))
```

## 8. Tham số khuyến nghị

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `top_k` | 5 | paper retrieve 5–10 rồi lọc |
| `max_segments` | 2 | long-form tăng lên 4–8 |
| `rel_threshold` | 0.25 (cosine rescale [0,1]) | tăng nếu corpus nhiễu |
| `sup_full / sup_part` | 0.55 / 0.30 | hard constraint bỏ `< sup_part` khi `enforce_supported=True` |
| `enforce_supported` | True | đặt False nếu muốn soft re-rank giữ lại Partially |

## 9. Hạn chế & hướng mở rộng

1. Critic heuristic yếu hơn Critic distill từ GPT-4 — production nên thay bằng LLM.
2. Chưa có segment-level beam search + threshold decoding đầy đủ như paper (mới greedy theo total).
3. Chưa học `Retrieve` bằng classifier — mới rule chitchat; nên train classifier nhỏ như Adaptive-RAG.
4. Nên kết hợp với RAPTOR (cây hierarchical) ở tầng retriever và RAFT ở tầng fine-tune cho domain.

## 10. Trích dẫn

```bibtex
@inproceedings{asai2024selfrag,
  title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
