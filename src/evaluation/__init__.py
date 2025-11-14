"""Evaluation metrics and utilities"""
from .metrics import (
    compute_similarity,
    evaluate_retrieval,
    evaluate_classification,
    evaluate_semantic_similarity,
    compute_embedding_statistics
)

__all__ = [
    "compute_similarity",
    "evaluate_retrieval",
    "evaluate_classification",
    "evaluate_semantic_similarity",
    "compute_embedding_statistics"
]
