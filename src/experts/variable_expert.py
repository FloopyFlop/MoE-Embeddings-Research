"""Variable-Size Experts for adaptive computation"""
import torch
import torch.nn as nn
from typing import List, Optional
from .gating import TopKGating, compute_load_balancing_loss


class VariableSizeExpert(nn.Module):
    """Expert with configurable size"""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class VariableSizeExpertLayer(nn.Module):
    """
    Layer with experts of different sizes for adaptive computation.

    Key innovation: Experts have different capacities (FFN sizes).
    - Small experts: Fast, for easy tokens
    - Large experts: Powerful, for complex tokens

    The gating network learns to route:
    - Simple tokens → Small experts (less compute)
    - Complex tokens → Large experts (more compute)

    Benefits:
    - Adaptive computation based on token difficulty
    - Better efficiency (not all tokens need max compute)
    - Load balancing across different expert sizes
    """

    def __init__(
        self,
        hidden_dim: int,
        expert_ff_dims: List[int] = None,
        top_k: int = 2,
        dropout: float = 0.1,
        noise_std: float = 0.1,
        load_balancing_weight: float = 0.01,
        size_penalty_weight: float = 0.001
    ):
        """
        Args:
            hidden_dim: Input/output dimension
            expert_ff_dims: List of FF dimensions for each expert
                           e.g., [256, 512, 768, 1024, 1024, 768, 512, 256]
                           Creates experts with varying sizes
            top_k: Number of experts to use per token
            dropout: Dropout rate
            noise_std: Noise for gating
            load_balancing_weight: Weight for load balancing
            size_penalty_weight: Weight to encourage using smaller experts
        """
        super().__init__()

        if expert_ff_dims is None:
            # Default: pyramid structure (small → large → small)
            expert_ff_dims = [256, 384, 512, 768, 768, 512, 384, 256]

        self.hidden_dim = hidden_dim
        self.num_experts = len(expert_ff_dims)
        self.expert_ff_dims = expert_ff_dims
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight
        self.size_penalty_weight = size_penalty_weight

        # Create experts with different sizes
        self.experts = nn.ModuleList([
            VariableSizeExpert(hidden_dim, ff_dim, dropout=dropout)
            for ff_dim in expert_ff_dims
        ])

        # Gating network
        self.gate = TopKGating(hidden_dim, self.num_experts, top_k, noise_std)

        # Store expert sizes for penalty calculation
        self.register_buffer(
            'expert_sizes',
            torch.tensor([expert.count_parameters() for expert in self.experts], dtype=torch.float32)
        )

        # Normalize sizes to [0, 1] range for penalty
        self.register_buffer(
            'expert_size_normalized',
            (self.expert_sizes - self.expert_sizes.min()) /
            (self.expert_sizes.max() - self.expert_sizes.min() + 1e-8)
        )

        self.aux_loss = None

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            return_aux_loss: Whether to compute auxiliary loss

        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # Get routing decisions
        gate_weights, selected_experts, router_logits = self.gate(x_flat, training=self.training)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process through experts
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx)

            for k in range(self.top_k):
                token_mask = expert_mask[:, k]

                if token_mask.any():
                    tokens_for_expert = x_flat[token_mask]
                    expert_output = self.experts[expert_idx](tokens_for_expert)
                    weights = gate_weights[token_mask, k:k+1]
                    output[token_mask] += expert_output * weights

        # Reshape back
        output = output.view(batch_size, seq_len, hidden_dim)

        # Compute auxiliary losses
        if return_aux_loss and self.training:
            router_logits_reshaped = router_logits.view(batch_size, seq_len, self.num_experts)
            selected_experts_reshaped = selected_experts.view(batch_size, seq_len, self.top_k)

            # Standard load balancing loss
            load_balance_loss = compute_load_balancing_loss(
                router_logits_reshaped,
                selected_experts_reshaped,
                self.num_experts
            )

            # Size penalty: encourage using smaller experts
            # Compute average size of selected experts
            router_probs = torch.softmax(router_logits, dim=-1)  # [batch*seq, num_experts]
            avg_expert_size = torch.sum(router_probs * self.expert_size_normalized, dim=-1).mean()
            size_penalty = avg_expert_size

            # Total aux loss
            self.aux_loss = (
                self.load_balancing_weight * load_balance_loss +
                self.size_penalty_weight * size_penalty
            )
        else:
            self.aux_loss = None

        return output

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """Get auxiliary loss"""
        return self.aux_loss

    def get_expert_stats(self) -> dict:
        """Get statistics about expert sizes and usage"""
        stats = {
            f"expert_{i}_params": self.experts[i].count_parameters()
            for i in range(self.num_experts)
        }
        stats["total_params"] = sum(stats.values())
        stats["min_expert_params"] = min(stats.values())
        stats["max_expert_params"] = max(stats.values())
        stats["avg_expert_params"] = stats["total_params"] / self.num_experts

        return stats

    def count_parameters(self) -> dict:
        """Count parameters"""
        expert_params = [expert.count_parameters() for expert in self.experts]
        gate_params = sum(p.numel() for p in self.gate.parameters())

        return {
            "expert_params": expert_params,
            "total_expert_params": sum(expert_params),
            "gate_params": gate_params,
            "total": sum(expert_params) + gate_params,
            "min_expert": min(expert_params),
            "max_expert": max(expert_params),
            "avg_expert": sum(expert_params) / len(expert_params)
        }
