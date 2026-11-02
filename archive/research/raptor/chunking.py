"""Chunking: split documents into ~100-token contiguous chunks (RAPTOR level 0)."""

from __future__ import annotations

import re


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter (no nltk dependency).
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 0) -> list[str]:
    """Split a single document into word-based chunks.

    Args:
        text: input document.
        chunk_size: max words per chunk (~100 tokens in the paper).
        overlap: number of overlapping words between consecutive chunks.
    """
    text = text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        # Very long single sentence: hard-cut it.
        if len(words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            for i in range(0, len(words), chunk_size - overlap if chunk_size > overlap else chunk_size):
                step = chunk_size - overlap if chunk_size > overlap else chunk_size
                chunks.append(" ".join(words[i : i + chunk_size]))
            continue
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            if overlap > 0:
                current = current[-overlap:] + words
                current_len = len(current)
            else:
                current, current_len = words, len(words)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_corpus(docs: list[str], chunk_size: int = 100, overlap: int = 0) -> list[str]:
    """Chunk a list of documents, skipping empties."""
    out: list[str] = []
    for doc in docs:
        out.extend(chunk_text(doc, chunk_size=chunk_size, overlap=overlap))
    return out
