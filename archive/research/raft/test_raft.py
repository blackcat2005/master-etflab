"""pytest suite for the RAFT implementation (offline)."""

from research.raft.data import QAItem, build_raft_examples, pick_distractors, to_jsonl
from research.raft.evaluate import answer_contains, citation_fidelity, evaluate_dataset
from research.raft.inference import RAFTInference
from research.raft.reasoning import extract_quotes, format_raft_answer, verify_quotes

CORPUS = [
    "Metformin is a first-line medication for type 2 diabetes.",
    "Attention Is All You Need introduced the Transformer.",
    "Adam is an optimizer combining momentum and RMSProp.",
    "GloVe learns word vectors from co-occurrence statistics.",
]


def test_quote_format_and_verify():
    ans = format_raft_answer("reasoning here", "Metformin is great", "Metformin")
    assert extract_quotes(ans) == ["Metformin is great"]
    assert verify_quotes(ans, ["xxx Metformin is great yyy"]) is True
    assert verify_quotes(ans, ["unrelated doc"]) is False
    assert verify_quotes("no quotes here", CORPUS) is False


def test_distractors_exclude_oracle():
    import random

    d = pick_distractors("diabetes medication?", CORPUS[0], CORPUS, 2, random.Random(0))
    assert CORPUS[0] not in d
    assert len(d) == 2


def test_builder_p_split():
    import random

    qa = [QAItem(question=f"Q{i}?", answer="A", oracle_doc=CORPUS[0]) for i in range(20)]
    full = build_raft_examples(qa, CORPUS, n_distractors=2, p_with_oracle=1.0, seed=1)
    assert all(e.has_oracle for e in full)
    none = build_raft_examples(qa, CORPUS, n_distractors=2, p_with_oracle=0.0, seed=1)
    assert not any(e.has_oracle for e in none)
    assert all(e.oracle_doc not in e.context_docs for e in none)
    mixed = build_raft_examples(qa, CORPUS, n_distractors=2, p_with_oracle=0.5, seed=7)
    assert 0 < sum(e.has_oracle for e in mixed) < 20
    # Mọi example có oracle đều có target quote verbatim trong context.
    for e in full:
        assert verify_quotes(e.answer, e.context_docs) is True
    rows = to_jsonl(full)
    assert set(rows[0]) == {"prompt", "completion", "has_oracle"}


def test_inference_ignores_distractors():
    infer = RAFTInference.__new__(RAFTInference)
    infer.relevance_threshold = 0.30
    infer._embedding_model = None
    docs = [CORPUS[2], CORPUS[0], CORPUS[3]]  # oracle ở giữa
    pred = infer.predict("What is first-line for type 2 diabetes?", docs)
    assert pred.used_doc_idx == 1
    assert "Metformin" in pred.answer
    assert verify_quotes(pred.answer, docs) is True


def test_inference_fallback_without_docs():
    infer = RAFTInference.__new__(RAFTInference)
    infer.relevance_threshold = 0.30
    infer._embedding_model = None
    pred = infer.predict("Anything?", [])
    assert pred.used_doc_idx == -1


def test_metrics():
    items = [
        {"pred_answer": format_raft_answer("r", "Metformin is great", "Metformin"), "context_docs": ["Metformin is great"], "gold": "Metformin"},
        {"pred_answer": "no quotes", "context_docs": ["whatever"], "gold": "Metformin"},
    ]
    assert citation_fidelity(items[0]["pred_answer"], items[0]["context_docs"]) == 1.0
    assert answer_contains(items[0]["pred_answer"], "Metformin") == 1.0
    m = evaluate_dataset(items)
    assert m == {"n": 2, "citation_fidelity": 0.5, "answer_accuracy": 0.5}
    assert evaluate_dataset([])["n"] == 0
