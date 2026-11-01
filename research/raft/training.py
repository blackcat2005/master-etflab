"""RAFT SFT recipe (paper dùng Llama-2-7B). File này chỉ chứa công thức + stub.

Paper train: supervised fine-tuning, input = Q + D* + distractors,
target = CoT answer có citation (xem data.to_jsonl). P~80% có oracle.

Để chạy full SFT cần GPU + transformers/peft (không có sẵn ở env này),
nên hàm train() import lazy và báo lỗi rõ ràng thay vì crash lúc import.
Dùng data.py để sinh JSONL rồi train ở máy có GPU theo snippet trong README.
"""

from __future__ import annotations


def train(
    train_jsonl: str,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str = "raft-llama2-7b",
    epochs: int = 2,
    lr: float = 2e-5,
) -> str:
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Full RAFT SFT cần `transformers` (+ GPU). "
            "Cài bằng `pip install -r research/raft/requirements-optional.txt` rồi chạy lại. "
            "Dữ liệu train sinh bằng research.raft.data.to_jsonl()."
        ) from e
    # Placeholder: wire Trainer ở máy có GPU theo README (không fake-train ở đây).
    _ = (train_jsonl, base_model, output_dir, epochs, lr)
    raise NotImplementedError(
        "Xem snippet Trainer trong research/raft/README.md §6 để chạy SFT thật."
    )
