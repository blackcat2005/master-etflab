"""pytest suite for the RAPTOR implementation."""

import numpy as np

from research.raptor.chunking import chunk_corpus, chunk_text
from research.raptor.clustering import cluster_embeddings
from research.raptor.retrieval import CollapsedRetriever
from research.raptor.summarization import ExtractiveSummarizer
from research.raptor.tree import Node, RaptorTreeBuilder


def test_chunk_text_basic():
    chunks = chunk_text("Hello world. " * 200, chunk_size=20)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 25 for c in chunks)


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_corpus([]) == []


def test_clustering_small():
    rng = np.random.RandomState(0)
    embs = rng.rand(4, 8).astype(np.float32)
    labels = cluster_embeddings(embs)
    assert len(labels) == 4
    assert set(labels.tolist()) == {0}  # too few -> single cluster


def test_clustering_splits():
    rng = np.random.RandomState(1)
    a = rng.rand(10, 8).astype(np.float32)
    b = rng.rand(10, 8).astype(np.float32) + 5.0
    embs = np.vstack([a, b])
    labels = cluster_embeddings(embs)
    assert len(set(labels.tolist())) >= 2


def test_tree_build_offline():
    texts = [f"Document chunk number {i} about retrieval augmented generation." for i in range(12)]
    rng = np.random.RandomState(2)
    embs = rng.rand(12, 16).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    builder = RaptorTreeBuilder(summarizer=ExtractiveSummarizer(), max_levels=3)
    nodes = builder.build(texts, embs, embed_fn=None)  # fallback mean embedding
    assert len(nodes) > len(texts)  # leaves + at least one parent
    assert any(n.level > 0 for n in nodes)
    # children ids must reference existing nodes
    ids = {n.id for n in nodes}
    for n in nodes:
        for c in n.children:
            assert c in ids


def test_collapsed_retrieval_ranking():
    e0 = np.array([1.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0], dtype=np.float32)
    nodes = [Node(id=0, text="cats", embedding=e0, level=0), Node(id=1, text="dogs", embedding=e1, level=0)]
    r = CollapsedRetriever(nodes)
    res = r.search(np.array([1.0, 0.1], dtype=np.float32), top_k=2)
    assert res[0].node.text == "cats"
    assert res[0].score > res[1].score


def test_node_serialization():
    n = Node(id=5, text="hello", embedding=np.ones(4, dtype=np.float32), level=1, children=[1, 2])
    d = n.to_dict()
    n2 = Node.from_dict(d)
    assert n2.id == 5 and n2.level == 1 and n2.children == [1, 2]
    assert np.allclose(n2.embedding, 1.0)
