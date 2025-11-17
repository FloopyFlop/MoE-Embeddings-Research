"""Mixture of Recurrent Experts (MoRE) - Novel architecture using expert recurrence"""
import torch
import torch.nn as nn
from typing import Optional
from .expert import Expert
from .gating import TopKGating, compute_load_balancing_loss


class MoRELayer(nn.Module):
    """
    Mixture of Recurrent Experts (MoRE)

    Key innovation: Instead of having many experts (e.g., 16),
    use fewer experts (e.g., 4) applied recurrently multiple times.

    Example: 4 recurrences × 4 experts = same capacity as 16 experts,
    but with parameter sharing and recurrent processing.

    Benefits:
    - Fewer parameters (4 experts vs 16 experts)
    - Recurrent processing allows refinement
    - Each token can visit different experts across recurrences
    - Temporal dynamics in expert routing
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_experts: int = 4,
        num_recurrences: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        noise_std: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        """
        Args:
            hidden_dim: Input/output dimension
            ff_dim: Feed-forward dimension for each expert
            num_experts: Number of base experts
            num_recurrences: Number of times to apply experts
            top_k: Number of experts to route each token to per recurrence
            dropout: Dropout probability
            noise_std: Noise for gating
            load_balancing_weight: Weight for load balancing loss
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_recurrences = num_recurrences
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Base experts (shared across recurrences)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ff_dim, dropout=dropout)
            for _ in range(num_experts)
        ])

        # Separate gating network for each recurrence
        # Allows different routing at each step
        self.gates = nn.ModuleList([
            TopKGating(hidden_dim, num_experts, top_k, noise_std)
            for _ in range(num_recurrences)
        ])

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

        # Store all aux losses
        total_aux_loss = 0.0
        aux_count = 0

        # Apply experts recurrently
        current = x
        for recurrence_idx in range(self.num_recurrences):
            # Flatten for processing
            current_flat = current.view(-1, hidden_dim)

            # Get routing for this recurrence
            gate_weights, selected_experts, router_logits = self.gates[recurrence_idx](
                current_flat, training=self.training
            )

            # Initialize output for this recurrence
            recurrence_output = torch.zeros_like(current_flat)

            # Process through experts
            for expert_idx in range(self.num_experts):
                expert_mask = (selected_experts == expert_idx)

                for k in range(self.top_k):
                    token_mask = expert_mask[:, k]

                    if token_mask.any():
                        tokens_for_expert = current_flat[token_mask]
                        expert_output = self.experts[expert_idx](tokens_for_expert)
                        weights = gate_weights[token_mask, k:k+1]
                        recurrence_output[token_mask] += expert_output * weights

            # Reshape back
            recurrence_output = recurrence_output.view(batch_size, seq_len, hidden_dim)

            # Residual connection within recurrence
            current = current + recurrence_output

            # Compute load balancing loss for this recurrence
            if return_aux_loss and self.training:
                router_logits_reshaped = router_logits.view(batch_size, seq_len, self.num_experts)
                selected_experts_reshaped = selected_experts.view(batch_size, seq_len, self.top_k)

                aux_loss = compute_load_balancing_loss(
                    router_logits_reshaped,
                    selected_experts_reshaped,
                    self.num_experts
                )
                total_aux_loss += aux_loss
                aux_count += 1

        # Average aux loss across recurrences
        if return_aux_loss and self.training and aux_count > 0:
            self.aux_loss = (total_aux_loss / aux_count) * self.load_balancing_weight
        else:
            self.aux_loss = None

        return current

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """Get the auxiliary load balancing loss"""
        return self.aux_loss

    def count_parameters(self) -> dict:
        """Count parameters in MoRE"""
        expert_params = sum(p.numel() for expert in self.experts for p in expert.parameters())
        gate_params = sum(p.numel() for gate in self.gates for p in gate.parameters())

        return {
            "expert_params": expert_params,
            "gate_params": gate_params,
            "total": expert_params + gate_params,
            "num_experts": self.num_experts,
            "num_recurrences": self.num_recurrences,
            "effective_capacity": self.num_experts * self.num_recurrences
        }
