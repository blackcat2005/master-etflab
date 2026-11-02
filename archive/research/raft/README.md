# RAFT — Retrieval Augmented FineTuning for Domain-Specific RAG

Implement thực tế theo paper **Zhang et al., UC Berkeley, COLM 2024** (`arxiv:2403.10131`).

> RAPTOR cải thiện **chunking**, Self-RAG cải thiện **dynamic retrieval + critique**. RAFT đánh vào điểm yếu còn lại của RAG domain: **model không biết bỏ qua tài liệu nhiễu**. RAFT dạy model bằng fine-tune: cho đọc cả tài liệu đúng + tài liệu nhiễu, bắt trích dẫn verbatim + reasoning CoT.

---

## 1. Ý tưởng cốt lõi (analogy đề thi trong paper)

- Fine-tune thường = thi **closed-book**: học thuộc rồi憶, lúc thi không được mở sách.
- RAG thường = thi **open-book** nhưng chưa từng luyện dùng sách: dễ chép nhầm trang nhiễu.
- RAFT = luyện **cách dùng sách**: lúc học đã cho đọc cả trang đúng + trang nhiễu, ép chỉ ra đúng đoạn + giải thích.

Công thức data:

```
P% data:     Q + D* (oracle) + D1..Dk (distractors) -> A* (CoT + ##begin_quote##...##end_quote##)
(1-P)% data: Q + D1..Dk (chỉ distractors)            -> A* (ép memorize kiến thức domain)
```

## 2. Thuật toán chi tiết

### 2.1 Distractors
Mix 2 loại để mô phỏng retriever thật:
- **Hard negatives**: docs giống query (Jaccard/embedding cao) nhưng không chứa đáp án.
- **Random**: docs ngẫu nhiên còn lại.
- Code: `data.pick_distractors()`.

### 2.2 Target answer (CoT + verbatim citation)
Format ép model grounding:
```
## Reasoning
... nêu doc nào chứa đáp án, bỏ qua distractors ...
##begin_quote##
<span chép nguyên văn từ oracle>
##end_quote##
## Answer
<đáp án cuối>
```
- Code: `reasoning.format_raft_answer()`, `verify_quotes()` kiểm tra quote là substring verbatim.

### 2.3 SFT
Paper SFT Llama-2-7B trên data trên (mask loss phần prompt, học phần completion). Kết quả: thắng `Llama2+RAG` và `DSF+RAG` trên PubMed, HotpotQA, Gorilla APIBench.
- Sinh JSONL: `data.to_jsonl()` → `{prompt, completion, has_oracle}`.
- Train thật: xem §6 (cần GPU).

### 2.4 Inference
1. Score từng retrieved doc với query (0.7 embedding cosine + 0.3 word-F1).
2. Top-1 vượt threshold → oracle: trích câu verbatim, trả lời CoT. Dưới threshold → fallback parametric (đúng tinh thần (1-P)% lúc train).
3. Distractors còn lại bị ignore tường minh trong reasoning.
- Code: `inference.RAFTInference`.

## 3. So sánh vs baseline domain RAG

| Cách làm | Điểm yếu | RAFT khắc phục |
|---|---|---|
| RAG prompting (frozen LLM) | Chép nhầm doc giống-chứ-không-đúng, similar ≠ relevant | Dạy ignore distractor bằng data |
| Domain fine-tune (DSF, closed-book) | Quên khi test open-book, không dùng được context mới | Train ngay trong setting open-book |
| RAG + reranker | Chỉ xếp lại thứ tự, vẫn có thể sinh sai | Ép citation verbatim + CoT |

Khi nào dùng RAFT: domain hẹp (y tế, pháp lý, API docs nội bộ), retriever nhiễu cao, cần citation kiểm chứng được.

## 4. Cấu trúc code

```
research/raft/
  README.md        # file này
  __init__.py      # public API
  data.py          # QAItem, RAFTExample, build_raft_examples, to_jsonl
  reasoning.py     # format/extract/verify CoT + quotes
  inference.py     # RAFTInference (offline, lọc nhiễu + trích verbatim)
  evaluate.py      # citation_fidelity + answer_accuracy
  training.py      # stub SFT (cần transformers + GPU)
  demo.py          # demo offline
  requirements.txt
  test_raft.py     # pytest
```

## 5. Cài đặt & chạy

```bash
# Base (demo + test + sinh data SFT, offline):
pip install -r research/raft/requirements.txt

# Optional (CHỈ máy có GPU, khi train SFT thật):
# pip install -r research/raft/requirements-optional.txt  # transformers / peft / accelerate / datasets

# Demo offline (không cần GPU/API key):
python -m research.raft.demo

# Test:
python -m pytest research/raft/test_raft.py -v
```

| Cài gì | Khi nào |
|---|---|
| `requirements.txt` | Luôn luôn — torch, sentence-transformers (scoring + trích quote), numpy, pytest |
| `requirements-optional.txt` | SFT Llama-2 thật: `transformers` + `peft` (LoRA) + `accelerate` + `datasets` |

## 6. Train SFT thật (máy có GPU)

```python
from research.raft.data import QAItem, build_raft_examples, to_jsonl
import json

qa = [QAItem(question="...", answer="...", oracle_doc="...")]
rows = to_jsonl(build_raft_examples(qa, corpus, n_distractors=4, p_with_oracle=0.8))
with open("raft_train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

Rồi SFT Llama-2-7B (paper: epochs=2, lr=2e-5, top-k test: NQ=3, Hotpot=1+gold):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
# dataset text = prompt + completion, mask prompt loss nếu cần
```

Tham số paper: `P=80%` có oracle là điểm ngọt cho cả 3 dataset; tăng distractors lúc train giúp robust nhưng quá nhiều sẽ hại.

## 7. Ví dụ sử dụng (offline inference)

```python
from research.raft import build_raft_examples, QAItem, RAFTInference, evaluate_dataset

qa = [QAItem(question="Which paper introduced the Transformer?",
             answer="Attention Is All You Need",
             oracle_doc="Attention Is All You Need introduced the Transformer...")]
ex = build_raft_examples(qa, corpus, n_distractors=3)[0]

pred = RAFTInference().predict("Which paper introduced the Transformer?", ex.context_docs)
print(pred.answer)       # ## Reasoning ... ##begin_quote## ... ## Answer ...
print(pred.used_doc_idx) # index doc được chọn, còn lại là distractors bị ignore
```

## 8. Tham số khuyến nghị

| Tham số | Mặc định code | Paper |
|---|---|---|
| `n_distractors` | 3 | 4–5 (NQ/Hotpot/Gorilla) |
| `p_with_oracle` | 0.8 | 0.8 tốt nhất thực nghiệm |
| `relevance_threshold` | 0.30 | tune theo retriever/domain |
| top-k test | tự chọn | NQ=3+gold, Hotpot=1+gold |

## 9. Hạn chế & hướng mở rộng

1. `RAFTInference` offline là extractive — model SFT thật mới reasoning trừu tượng được.
2. Distractor sampling mới dùng Jaccard — production nên dùng embedding retriever thật để lấy hard negatives.
3. Chưa xử lý multi-oracle (HotpotQA cần 2 docs) — mở rộng `oracle_doc: list[str]`.
4. Nên kết hợp: RAPTOR (chunk tree) → retriever → RAFT-SFT model → Self-RAG critique.

## 10. Trích dẫn

```bibtex
@inproceedings{raft2024,
  title={RAFT: Adapting Language Model to Domain Specific RAG},
  author={Tianjun Zhang and Shishir G. Patil and Naman Jain and Sheng Shen and Matei Zaharia and Ion Stoica and Joseph E. Gonzalez},
  year={2024},
  journal={arXiv preprint arXiv:2403.10131}
}
```
