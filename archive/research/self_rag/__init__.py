"""Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (ICLR 2024 Oral)."""

from .critic import EmbeddingCritic
from .generator import ExtractiveGenerator
from .pipeline import SelfRAGPipeline, SelfRAGResult
from .reflection import IsRel, IsSup, IsUse, Retrieve
from .retriever import InMemoryRetriever

__all__ = [
    "Retrieve",
    "IsRel",
    "IsSup",
    "IsUse",
    "InMemoryRetriever",
    "EmbeddingCritic",
    "ExtractiveGenerator",
    "SelfRAGPipeline",
    "SelfRAGResult",
]

__version__ = "1.0.0"
