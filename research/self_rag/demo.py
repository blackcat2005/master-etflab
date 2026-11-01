"""Offline Self-RAG demo (no API key). Run: python -m research.self_rag.demo"""

from .pipeline import SelfRAGPipeline

CORPUS = [
    "RAPTOR builds a hierarchical tree from documents. Leaf chunks of about 100 tokens are embedded, "
    "clustered with Gaussian Mixture Models, and summarized by a language model bottom-up. "
    "It improves QA over long documents and reaches state of the art on the QuALITY benchmark.",
    "Self-RAG trains a language model to retrieve on demand and critique with reflection tokens. "
    "It predicts Retrieve Yes or No, checks passage relevance IsRel, verifies support IsSup, "
    "and scores utility IsUse. It outperforms ChatGPT and Llama2-chat on open-domain QA and factuality.",
    "RAFT fine-tunes language models for domain-specific RAG with oracle and distractor documents. "
    "The model cites the right span verbatim and ignores irrelevant passages using chain-of-thought answers. "
    "It improves results on PubMed, HotpotQA and Gorilla APIBench.",
    "The Eiffel Tower is located in Paris, France. It was constructed in 1889 for the World's Fair.",
]


def main() -> None:
    pipe = SelfRAGPipeline().fit(CORPUS)

    queries = [
        "How does Self-RAG decide when to retrieve?",
        "Where is the Eiffel Tower?",
        "Hello",  # chitchat -> Retrieve=No
        "Which method handles noisy retrieval in domain-specific RAG?",
    ]
    for q in queries:
        res = pipe.answer(q, top_k=4, max_segments=2)
        print(f"\nQ: {q}")
        print(f"A: {res.answer}")
        print(res.trace())


if __name__ == "__main__":
    main()
