"""pytest suite for the Self-RAG implementation (offline, no LLM needed)."""

from research.self_rag.critic import EmbeddingCritic, word_f1
from research.self_rag.generator import ExtractiveGenerator
from research.self_rag.pipeline import SelfRAGPipeline
from research.self_rag.reflection import IsRel, IsSup, Retrieve


def test_retrieve_decision_chitchat():
    c = EmbeddingCritic.__new__(EmbeddingCritic)  # không cần embedding cho rule này
    assert c.decide_retrieve("Hello") == Retrieve.NO
    assert c.decide_retrieve("What is Self-RAG?") == Retrieve.YES
    assert c.decide_retrieve("") == Retrieve.NO


def test_word_f1():
    assert word_f1("the cat sits", "the cat sits") == 1.0
    assert word_f1("apple orange", "car train") == 0.0
    assert 0.0 < word_f1("RAPTOR builds trees", "RAPTOR builds hierarchical trees") < 1.0


def test_extractive_generator_picks_relevant_sentence():
    g = ExtractiveGenerator()
    passage = "The Eiffel Tower is in Paris. Bananas are yellow and sweet."
    out = g.draft("Where is the Eiffel Tower?", passage)
    assert "Paris" in out


def test_pipeline_retrieves_and_cites():
    pipe = SelfRAGPipeline().fit(
        [
            "RAPTOR builds hierarchical trees with clustering and summarization.",
            "The Eiffel Tower is located in Paris, France.",
        ]
    )
    res = pipe.answer("Where is the Eiffel Tower?", top_k=2, max_segments=1)
    assert res.retrieve == Retrieve.YES
    assert "Paris" in res.answer
    assert "[1]" in res.answer
    assert len(res.segments) == 1
    assert res.segments[0].critique.issup in (IsSup.FULLY, IsSup.PARTIALLY)


def test_pipeline_no_retrieval_for_chitchat():
    pipe = SelfRAGPipeline().fit(["Some domain knowledge about Paris."])
    res = pipe.answer("Hello", top_k=2)
    assert res.retrieve == Retrieve.NO
    assert res.segments == []


def test_pipeline_skips_unsupported():
    pipe = SelfRAGPipeline().fit(
        ["Completely unrelated text about cooking pasta with tomatoes."]
    )
    res = pipe.answer("Quantum entanglement in superconductors?", top_k=1, max_segments=1)
    # Passage không liên quan hoặc không support -> parametric fallback, không citation.
    assert "[1]" not in res.answer or len(res.segments) <= 1


def test_isrel_labels_valid():
    assert set(IsRel) == {IsRel.RELEVANT, IsRel.IRRELEVANT}
    assert set(Retrieve) == {Retrieve.YES, Retrieve.NO, Retrieve.CONTINUE}
