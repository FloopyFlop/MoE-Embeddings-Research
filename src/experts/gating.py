"""Gating mechanisms for routing inputs to experts"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGating(nn.Module):
    """
    Top-K gating mechanism that routes each token to the top-K experts.
    Implements the gating strategy from "Switch Transformers" and similar papers.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2, noise_std: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of input
            num_experts: Total number of experts
            top_k: Number of experts to route each token to
            noise_std: Standard deviation of noise added during training (for load balancing)
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Linear layer to compute gating scores
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Initialize with small weights for stable training
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, training: bool = True) -> tuple:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] or [batch_size * seq_len, hidden_dim]
            training: Whether in training mode (adds noise)

        Returns:
            Tuple of:
                - gate_weights: [batch_size, seq_len, top_k] - weights for selected experts
                - selected_experts: [batch_size, seq_len, top_k] - indices of selected experts
                - router_logits: [batch_size, seq_len, num_experts] - raw routing scores (for loss)
        """
        original_shape = x.shape
        # Flatten if needed: [batch * seq_len, hidden_dim]
        if len(x.shape) == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = x.view(-1, hidden_dim)
        else:
            x_flat = x
            batch_size, seq_len = None, None

        # Compute gating logits: [batch * seq_len, num_experts]
        logits = self.gate(x_flat)

        # Add noise during training for load balancing
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Get top-k experts and their weights
        # top_k_logits: [batch * seq_len, top_k]
        # top_k_indices: [batch * seq_len, top_k]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax over top-k to get weights
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Reshape back if needed
        if batch_size is not None:
            top_k_weights = top_k_weights.view(batch_size, seq_len, self.top_k)
            top_k_indices = top_k_indices.view(batch_size, seq_len, self.top_k)
            logits = logits.view(batch_size, seq_len, self.num_experts)

        return top_k_weights, top_k_indices, logits


class SoftmaxGating(nn.Module):
    """
    Softmax gating that assigns weights to all experts.
    Simpler than Top-K but less sparse (all experts are used).
    """

    def __init__(self, hidden_dim: int, num_experts: int, temperature: float = 1.0):
        """
        Args:
            hidden_dim: Input dimension
            num_experts: Number of experts
            temperature: Temperature for softmax (lower = more peaked)
        """
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor

        Returns:
            Tuple of (weights, indices, logits)
            - weights: [batch, seq_len, num_experts]
            - indices: None (all experts used)
            - logits: [batch, seq_len, num_experts]
        """
        logits = self.gate(x) / self.temperature
        weights = F.softmax(logits, dim=-1)

        return weights, None, logits


def compute_load_balancing_loss(router_logits: torch.Tensor, selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute load balancing loss to encourage equal usage of experts.
    Based on "Switch Transformers" auxiliary loss.

    Args:
        router_logits: [batch, seq_len, num_experts] - raw router scores
        selected_experts: [batch, seq_len, top_k] - indices of selected experts
        num_experts: Total number of experts

    Returns:
        Scalar loss value
    """
    # Compute fraction of tokens assigned to each expert
    # expert_counts: [num_experts]
    batch_size, seq_len, top_k = selected_experts.shape
    total_tokens = batch_size * seq_len * top_k

    # Count assignments to each expert
    expert_counts = torch.zeros(num_experts, device=selected_experts.device)
    for expert_id in range(num_experts):
        expert_counts[expert_id] = (selected_experts == expert_id).sum()

    # Fraction of tokens assigned to each expert
    expert_fractions = expert_counts / total_tokens

    # Compute average router probability for each expert
    # router_probs: [batch, seq_len, num_experts]
    router_probs = F.softmax(router_logits, dim=-1)
    avg_router_probs = router_probs.mean(dim=[0, 1])  # [num_experts]

    # Load balancing loss: encourages both to be uniform (1/num_experts)
    # This is the "auxiliary loss" from Switch Transformers
    load_balancing_loss = num_experts * torch.sum(expert_fractions * avg_router_probs)

    return load_balancing_loss
