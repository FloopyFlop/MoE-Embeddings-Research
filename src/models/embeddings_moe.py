"""MoE-enabled embedding model"""
import torch
import torch.nn as nn
from .encoder_moe import TransformerEncoderMoE
from .pooling import Pooler
from typing import Dict, Optional, Literal


class EmbeddingModelMoE(nn.Module):
    """
    Sentence embedding model with Mixture of Experts.

    Key differences from standard model:
    - Uses MoE layers instead of standard feed-forward
    - Has auxiliary load balancing loss
    - Sparse computation (only top-k experts active per token)
    - More total parameters but same active parameters
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pooling_mode: Literal["mean", "max", "cls", "mean_max"] = "mean",
        pad_token_id: int = 0,
        normalize_embeddings: bool = True
    ):
        """
        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension PER EXPERT
            num_experts: Total number of experts per layer
            top_k: Number of experts activated per token
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pooling_mode: Pooling strategy
            pad_token_id: Padding token ID
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.normalize_embeddings = normalize_embeddings
        self.num_experts = num_experts
        self.top_k = top_k

        # MoE Encoder
        self.encoder = TransformerEncoderMoE(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_experts=num_experts,
            top_k=top_k,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id
        )

        # Pooler
        self.pooler = Pooler(
            pooling_mode=pooling_mode,
            hidden_dim=hidden_dim
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_dict: Whether to return dictionary

        Returns:
            Dictionary with embeddings, hidden_states, and aux_loss
        """
        # Get sequence of hidden states
        hidden_states = self.encoder(input_ids, attention_mask)

        # Pool to sentence embedding
        embeddings = self.pooler(hidden_states, attention_mask)

        # L2 normalization
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Get auxiliary loss
        aux_loss = self.encoder.get_aux_loss()

        if return_dict:
            return {
                "embeddings": embeddings,
                "hidden_states": hidden_states,
                "aux_loss": aux_loss
            }
        else:
            return embeddings

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convenience method for encoding"""
        return self.forward(input_ids, attention_mask, return_dict=False)

    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Batch encoding for evaluation"""
        self.eval()
        device = next(self.parameters()).device

        all_embeddings = []
        num_samples = input_ids.size(0)

        for i in range(0, num_samples, batch_size):
            batch_input_ids = input_ids[i:i + batch_size].to(device)
            batch_mask = attention_mask[i:i + batch_size].to(device) if attention_mask is not None else None

            embeddings = self.encode(batch_input_ids, batch_mask)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def get_expert_usage_stats(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """Get expert usage statistics"""
        return self.encoder.get_expert_usage_stats(input_ids, attention_mask)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters.

        Returns:
            Dictionary with total, active, and inactive parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())

        # Estimate active parameters (rough calculation)
        # Active = non-expert params + (top_k / num_experts) * expert params
        expert_params = sum(p.numel() for name, p in self.named_parameters() if 'experts' in name)
        non_expert_params = total_params - expert_params

        # Fraction of experts active per forward pass
        active_fraction = self.top_k / self.num_experts
        active_expert_params = int(expert_params * active_fraction)
        active_params = non_expert_params + active_expert_params

        return {
            "total": total_params,
            "active": active_params,
            "non_expert": non_expert_params,
            "expert_total": expert_params,
            "expert_active": active_expert_params,
            "sparsity": 1.0 - active_fraction
        }
