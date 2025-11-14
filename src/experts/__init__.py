"""
Mixture of Experts components.

This module contains:
- Expert networks (specialized feed-forward networks)
- Gating mechanisms (learned routing with Top-K selection)
- MoE layers to replace standard feed-forward layers in the transformer
- Load balancing mechanisms to ensure experts are used evenly
"""

from .expert import Expert, SparseExpert
from .gating import TopKGating, SoftmaxGating, compute_load_balancing_loss
from .moe_layer import MoELayer, EfficientMoELayer

__all__ = [
    "Expert",
    "SparseExpert",
    "TopKGating",
    "SoftmaxGating",
    "compute_load_balancing_loss",
    "MoELayer",
    "EfficientMoELayer"
]
