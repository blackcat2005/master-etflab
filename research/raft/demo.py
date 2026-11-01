"""Offline RAFT demo (no GPU/API key). Run: python -m research.raft.demo"""

from .data import QAItem, build_raft_examples, to_jsonl
from .evaluate import evaluate_dataset
from .inference import RAFTInference

CORPUS = [
    "Metformin is a first-line medication for type 2 diabetes. It decreases hepatic glucose production.",
    "Attention Is All You Need introduced the Transformer architecture with self-attention in 2017.",
    "Adam is an optimizer combining momentum and RMSProp, widely used for training neural networks.",
    "GloVe learns word vectors from global co-occurrence statistics of a corpus.",
    "ResNet uses residual connections to train very deep convolutional networks for image recognition.",
]

QA = [
    QAItem(
        question="What is the first-line medication for type 2 diabetes?",
        answer="Metformin",
        oracle_doc=CORPUS[0],
    ),
    QAItem(
        question="Which paper introduced the Transformer?",
        answer="Attention Is All You Need",
        oracle_doc=CORPUS[1],
    ),
]


def main() -> None:
    # 1. Build RAFT training data (P% có oracle, còn lại chỉ distractors).
    examples = build_raft_examples(QA, CORPUS, n_distractors=3, p_with_oracle=0.8, seed=42)
    rows = to_jsonl(examples)
    print(f"Built {len(rows)} RAFT examples (has_oracle={[r['has_oracle'] for r in rows]})")
    print("--- sample training target ---")
    print(rows[0]["completion"][:400])

    # 2. Inference với retriever nhiễu (oracle + 3 distractors, xáo trộn).
    infer = RAFTInference()
    eval_items = []
    for item in QA:
        docs = [CORPUS[2], item.oracle_doc, CORPUS[3], CORPUS[4]]  # oracle lẫn trong nhiễu
        pred = infer.predict(item.question, docs)
        print(f"\nQ: {item.question}")
        print(f"doc_scores={[round(s, 2) for s in pred.doc_scores]} used_doc={pred.used_doc_idx}")
        print(pred.answer[:500])
        eval_items.append({"pred_answer": pred.answer, "context_docs": docs, "gold": item.answer})

    print("\nMetrics:", evaluate_dataset(eval_items))


if __name__ == "__main__":
    main()
