"""Offline RAPTOR demo (no API key needed). Run: python -m research.raptor.demo"""

from .pipeline import RaptorPipeline

CORPUS = [
    "RAPTOR builds a hierarchical tree from documents. Leaf chunks of about 100 tokens are embedded, "
    "clustered with Gaussian Mixture Models, and summarized by a language model. The process repeats "
    "bottom-up until a root summary remains. This preserves both details and global context.",
    "Self-RAG trains a language model to retrieve on demand and critique its own output with reflection "
    "tokens. It decides when retrieval is needed, checks passage relevance, and verifies whether the answer "
    "is supported. It outperforms ChatGPT and Llama2-chat on open-domain QA and long-form factuality.",
    "RAFT adapts language models to domain-specific RAG by fine-tuning with oracle documents and distractor "
    "documents. The model learns to cite the right span verbatim and ignore irrelevant passages, using a "
    "chain-of-thought style answer. It improves results on PubMed, HotpotQA and Gorilla APIBench.",
    "Adaptive-RAG uses a small classifier to predict question complexity, then picks no-retrieval, single-step "
    "or multi-step retrieval. Simple questions skip retrieval to save cost, while complex ones get iterative "
    "reasoning. It balances accuracy and efficiency on diverse QA benchmarks.",
    "LightRAG indexes a knowledge graph with dual-level retrieval over entities and relations. It suits domains "
    "with dense connections such as medicine, law and engineering, where multi-hop reasoning over structured "
    "relations beats flat chunk retrieval.",
    "Dual-granularity chunking splits documents into fine and coarse chunks and augments context dynamically. "
    "It is cheap and effective for short documents, but fixed at two levels and lacks abstractive summaries, "
    "so it struggles on long documents requiring synthesis across many sections.",
]


def main() -> None:
    pipe = RaptorPipeline(chunk_size=60, max_levels=3)
    nodes = pipe.fit(CORPUS)
    print(f"Built tree: {len(nodes)} nodes, levels={pipe.level_counts()}")

    queries = [
        "How does RAPTOR handle long documents?",
        "Which method is best for domain-specific RAG with noisy retrieval?",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        for r in pipe.retrieve(q, top_k=3):
            print(f"  [{r.score:.3f}|L{r.node.level}] {r.node.text[:140]}...")
        print("--- context preview ---")
        print(pipe.build_context(q, top_k=3, max_chars=800)[:800])

    pipe.save("raptor_tree.json")
    print("\nSaved tree to raptor_tree.json")


if __name__ == "__main__":
    main()
