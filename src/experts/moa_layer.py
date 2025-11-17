"""Mixture of Attention (MoA) - Route to different attention mechanisms"""
import torch
import torch.nn as nn
import math
from typing import Optional
from .gating import TopKGating, compute_load_balancing_loss


class AttentionExpert(nn.Module):
    """Single attention expert with potentially different configuration"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_type: str = "standard"
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head (default: hidden_dim // num_heads)
            dropout: Dropout rate
            attention_type: Type of attention ("standard", "linear", "local")
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else hidden_dim // num_heads
        self.attention_type = attention_type
        self.scale = math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        if self.attention_type == "standard":
            # Standard scaled dot-product attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        elif self.attention_type == "linear":
            # Linear attention (approximate)
            q = torch.nn.functional.elu(q) + 1
            k = torch.nn.functional.elu(k) + 1
            kv = torch.matmul(k.transpose(-2, -1), v)
            attn_output = torch.matmul(q, kv)
            attn_output = attn_output / (torch.matmul(q, k.sum(dim=2, keepdim=True).transpose(-2, -1)) + 1e-6)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return self.out_proj(attn_output)

        elif self.attention_type == "local":
            # Local attention (window size 64)
            window_size = min(64, seq_len)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

            # Create local mask
            seq_indices = torch.arange(seq_len, device=x.device)
            distance = seq_indices.unsqueeze(0) - seq_indices.unsqueeze(1)
            local_mask = (distance.abs() <= window_size // 2)
            attn_scores = attn_scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Replace NaN from all-masked rows with zeros
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

        attn_probs = self.dropout(attn_probs)

        # Apply to values
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


class MixtureOfAttentionLayer(nn.Module):
    """
    Mixture of Attention (MoA) Layer

    Key innovation: Instead of one attention mechanism, use multiple
    attention "experts" with different characteristics:
    - Standard attention (global)
    - Linear attention (efficient)
    - Local attention (nearby tokens)
    - Different head configurations

    The gating network learns which attention mechanism is best for each token.

    Benefits:
    - Adaptive attention based on context
    - Efficiency (can use linear/local for simple patterns)
    - Diversity in attention patterns
    """

    def __init__(
        self,
        hidden_dim: int,
        num_attention_experts: int = 4,
        attention_configs: Optional[list] = None,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_attention_experts: Number of attention experts
            attention_configs: List of configs for each expert
                              [(num_heads, type), ...]
            top_k: Number of attention experts to use per token
            dropout: Dropout rate
            load_balancing_weight: Load balancing weight
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_attention_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Default attention configurations if not provided
        if attention_configs is None:
            attention_configs = [
                (8, "standard"),   # Expert 0: Standard attention, 8 heads
                (12, "standard"),  # Expert 1: Standard attention, 12 heads (more heads)
                (8, "linear"),     # Expert 2: Linear attention (efficient)
                (8, "local"),      # Expert 3: Local attention (nearby)
            ]

        # Pad or trim configs to match num_experts
        while len(attention_configs) < num_attention_experts:
            attention_configs.append((8, "standard"))
        attention_configs = attention_configs[:num_attention_experts]

        # Create attention experts
        self.attention_experts = nn.ModuleList([
            AttentionExpert(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_type=attn_type
            )
            for num_heads, attn_type in attention_configs
        ])

        # Gating network for routing
        self.gate = TopKGating(hidden_dim, num_attention_experts, top_k, noise_std=0.1)

        self.aux_loss = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            return_aux_loss: Whether to compute aux loss

        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # Get routing decisions
        gate_weights, selected_experts, router_logits = self.gate(x_flat, training=self.training)

        # Reshape routing info
        gate_weights = gate_weights.view(batch_size, seq_len, self.top_k)
        selected_experts = selected_experts.view(batch_size, seq_len, self.top_k)

        # Process through all attention experts
        # Note: For attention, we need to process full sequences, not individual tokens
        # Stack all expert outputs: [num_experts, batch_size, seq_len, hidden_dim]
        expert_outputs = torch.stack([
            self.attention_experts[expert_idx](x, attention_mask)
            for expert_idx in range(self.num_experts)
        ], dim=0)

        # Combine expert outputs based on gating decisions
        # Create gather indices: [batch_size, seq_len, top_k, hidden_dim]
        expert_indices = selected_experts.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)

        # Gather selected expert outputs: [batch_size, seq_len, top_k, hidden_dim]
        selected_outputs = torch.gather(
            expert_outputs.permute(1, 2, 0, 3),  # [batch, seq, experts, hidden]
            dim=2,
            index=expert_indices
        )

        # Apply gating weights: [batch_size, seq_len, top_k, 1]
        gate_weights_expanded = gate_weights.unsqueeze(-1)

        # Weighted sum: [batch_size, seq_len, hidden_dim]
        output = (selected_outputs * gate_weights_expanded).sum(dim=2)

        # Compute load balancing loss
        if return_aux_loss and self.training:
            router_logits_reshaped = router_logits.view(batch_size, seq_len, self.num_experts)
            selected_experts_reshaped = selected_experts

            self.aux_loss = compute_load_balancing_loss(
                router_logits_reshaped,
                selected_experts_reshaped,
                self.num_experts
            ) * self.load_balancing_weight
        else:
            self.aux_loss = None

        return output

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """Get auxiliary loss"""
        return self.aux_loss

    def count_parameters(self) -> dict:
        """Count parameters"""
        expert_params = [
            sum(p.numel() for p in expert.parameters())
            for expert in self.attention_experts
        ]
        gate_params = sum(p.numel() for p in self.gate.parameters())

        return {
            "expert_params": expert_params,
            "total_expert_params": sum(expert_params),
            "gate_params": gate_params,
            "total": sum(expert_params) + gate_params,
        }
