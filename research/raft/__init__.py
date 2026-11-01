"""RAFT: Adapting Language Model to Domain Specific RAG (COLM 2024, Berkeley/Meta/MS)."""

from .data import QAItem, RAFTExample, build_raft_examples
from .evaluate import evaluate_dataset
from .inference import RAFTInference
from .reasoning import extract_quotes, format_raft_answer, verify_quotes

__all__ = [
    "QAItem",
    "RAFTExample",
    "build_raft_examples",
    "format_raft_answer",
    "extract_quotes",
    "verify_quotes",
    "RAFTInference",
    "evaluate_dataset",
]

__version__ = "1.0.0"
