# Research

Thư mục chứa code + docs thử nghiệm các paper RAG. Mỗi paper một folder riêng, tự chứa code, requirements, README.

## Danh sách

| Folder | Paper | Venue | Yêu cầu |
|---|---|---|---|
| `raptor/` | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | ICLR 2024 | `raptor/requirements.txt` (+ `requirements-optional.txt` cho faiss/umap/openai) |
| `self_rag/` | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | ICLR 2024 Oral | `self_rag/requirements.txt` (+ `requirements-optional.txt` cho transformers/openai) |
| `raft/` | RAFT: Adapting Language Model to Domain Specific RAG | COLM 2024 | `raft/requirements.txt` (+ `requirements-optional.txt` cho SFT GPU) |

Cài 1 lần cho tất cả: `pip install -r research/requirements-all.txt` (hợp của các base, đã verify).

## Quy ước

- Mỗi folder có `README.md` (docs + cách chạy), `requirements.txt` (base: demo+test offline), `requirements-optional.txt` (chỉ khi cần production/GPU), `demo.py`, `test_*.py`.
- Version trong `requirements.txt` pin `==` theo môi trường đã verify (Python 3.13, torch 2.14.0, sentence-transformers 6.1.0, numpy 2.5.3) — không dùng `>=` trôi nổi.
- Import theo dạng `from research.<paper> import ...`, chạy demo `python -m archive.research.<paper>.demo`.
- Không để code research lẫn với code app/server chính.
