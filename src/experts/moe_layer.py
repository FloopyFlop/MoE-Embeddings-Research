"""Mixture of Experts layer implementation"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .expert import Expert, SparseExpert
from .gating import TopKGating, compute_load_balancing_loss


class MoELayer(nn.Module):
    """
    Mixture of Experts layer that replaces the standard feed-forward layer.

    Key features:
    - Top-K routing (only k experts are active per token)
    - Load balancing to ensure experts are used evenly
    - Sparse computation for efficiency
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_type: str = "standard",  # "standard" or "sparse"
        dropout: float = 0.1,
        noise_std: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        """
        Args:
            hidden_dim: Input/output dimension
            ff_dim: Feed-forward dimension for each expert
            num_experts: Total number of experts
            top_k: Number of experts to route each token to
            expert_type: Type of expert ("standard" or "sparse")
            dropout: Dropout probability
            noise_std: Noise for gating (load balancing)
            load_balancing_weight: Weight for load balancing loss
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Create experts
        if expert_type == "sparse":
            self.experts = nn.ModuleList([
                SparseExpert(hidden_dim, ff_dim, bottleneck_factor=0.5, dropout=dropout)
                for _ in range(num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                Expert(hidden_dim, ff_dim, dropout=dropout)
                for _ in range(num_experts)
            ])

        # Gating network
        self.gate = TopKGating(hidden_dim, num_experts, top_k, noise_std)

        # Store aux loss
        self.aux_loss = None

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            return_aux_loss: Whether to compute and store auxiliary loss

        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        original_shape = x.shape

        # Flatten for easier processing: [batch * seq_len, hidden_dim]
        x_flat = x.view(-1, hidden_dim)

        # Get routing decisions
        # gate_weights: [batch * seq_len, top_k]
        # selected_experts: [batch * seq_len, top_k]
        # router_logits: [batch * seq_len, num_experts]
        gate_weights, selected_experts, router_logits = self.gate(x_flat, training=self.training)

        # Reshape for computation
        gate_weights = gate_weights.view(batch_size * seq_len, self.top_k)
        selected_experts = selected_experts.view(batch_size * seq_len, self.top_k)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        # This is a simple but inefficient implementation
        # In production, you'd use more sophisticated batching
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (selected_experts == expert_idx)  # [batch * seq_len, top_k]

            # Get tokens for this expert
            for k in range(self.top_k):
                # Tokens where this expert is in position k
                token_mask = expert_mask[:, k]  # [batch * seq_len]

                if token_mask.any():
                    # Get tokens assigned to this expert in position k
                    tokens_for_expert = x_flat[token_mask]  # [num_tokens, hidden_dim]

                    # Process through expert
                    expert_output = self.experts[expert_idx](tokens_for_expert)

                    # Get weights for these tokens
                    weights = gate_weights[token_mask, k:k+1]  # [num_tokens, 1]

                    # Add weighted output
                    output[token_mask] += expert_output * weights

        # Reshape back
        output = output.view(batch_size, seq_len, hidden_dim)

        # Compute load balancing loss
        if return_aux_loss and self.training:
            router_logits_reshaped = router_logits.view(batch_size, seq_len, self.num_experts)
            selected_experts_reshaped = selected_experts.view(batch_size, seq_len, self.top_k)

            self.aux_loss = compute_load_balancing_loss(
                router_logits_reshaped,
                selected_experts_reshaped,
                self.num_experts
            ) * self.load_balancing_weight
        else:
            self.aux_loss = None

        return output

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """Get the auxiliary load balancing loss"""
        return self.aux_loss

    def get_expert_usage(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get statistics on expert usage.

        Args:
            x: Input tensor

        Returns:
            Tensor of shape [num_experts] with usage counts
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        with torch.no_grad():
            _, selected_experts, _ = self.gate(x_flat, training=False)

            # Count usage
            usage = torch.zeros(self.num_experts, device=x.device)
            for expert_idx in range(self.num_experts):
                usage[expert_idx] = (selected_experts == expert_idx).sum()

        return usage


class EfficientMoELayer(nn.Module):
    """
    More efficient MoE implementation using grouped computation.
    Better for production use.
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        capacity_factor: float = 1.25
    ):
        """
        Args:
            hidden_dim: Input/output dimension
            ff_dim: Expert hidden dimension
            num_experts: Number of experts
            top_k: Experts per token
            dropout: Dropout rate
            capacity_factor: Capacity factor for expert buffer (1.0 = exact, >1.0 = overflow buffer)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Experts
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ff_dim, dropout=dropout)
            for _ in range(num_experts)
        ])

        # Gate
        self.gate = TopKGating(hidden_dim, num_experts, top_k, noise_std=0.1)

        self.aux_loss = None

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True) -> torch.Tensor:
        """Forward pass with batched expert computation"""
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)

        # Route
        gate_weights, selected_experts, router_logits = self.gate(x_flat, training=self.training)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process by expert (batched)
        for expert_idx in range(self.num_experts):
            # Find all tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)

            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = x_flat[expert_mask]

                # Process
                expert_outputs = self.experts[expert_idx](expert_inputs)

                # Get weights and combine
                # This is simplified - in practice you'd handle multiple k positions
                expert_weight_mask = (selected_experts == expert_idx)
                weights = gate_weights[expert_mask][expert_weight_mask[expert_mask]].unsqueeze(-1)

                # Accumulate
                output[expert_mask] += expert_outputs * weights

        output = output.reshape(batch_size, seq_len, hidden_dim)

        # Aux loss
        if return_aux_loss and self.training:
            router_logits_reshaped = router_logits.reshape(batch_size, seq_len, self.num_experts)
            selected_experts_reshaped = selected_experts.reshape(batch_size, seq_len, self.top_k)
            self.aux_loss = compute_load_balancing_loss(
                router_logits_reshaped,
                selected_experts_reshaped,
                self.num_experts
            ) * 0.01
        else:
            self.aux_loss = None

        return output

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self.aux_loss
