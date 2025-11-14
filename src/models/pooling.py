"""Pooling strategies for converting sequence outputs to fixed embeddings"""
import torch
import torch.nn as nn
from typing import Literal


class Pooler(nn.Module):
    """
    Pooling layer to convert sequence of hidden states to a single embedding vector.
    Supports multiple pooling strategies, modular for MoE integration.
    """

    def __init__(
        self,
        pooling_mode: Literal["mean", "max", "cls", "mean_max"] = "mean",
        hidden_dim: int = 256
    ):
        """
        Args:
            pooling_mode: Strategy for pooling
                - "mean": Mean pooling over all tokens (excluding padding)
                - "max": Max pooling over all tokens
                - "cls": Use [CLS] token (first token) representation
                - "mean_max": Concatenate mean and max pooling
            hidden_dim: Dimension of hidden states
        """
        super().__init__()
        self.pooling_mode = pooling_mode
        self.hidden_dim = hidden_dim

        # For mean_max, we concatenate, so output dim is 2x hidden_dim
        # Add a projection layer to bring it back to hidden_dim
        if pooling_mode == "mean_max":
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.projection = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] (1 for valid, 0 for padding)
        Returns:
            [batch_size, hidden_dim] - pooled embeddings
        """
        if self.pooling_mode == "cls":
            # Use first token ([CLS] token)
            pooled = hidden_states[:, 0, :]

        elif self.pooling_mode == "max":
            # Max pooling over sequence
            if attention_mask is not None:
                # Mask padding tokens with large negative value
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states.clone()
                hidden_states[mask_expanded == 0] = -1e9
            pooled = torch.max(hidden_states, dim=1)[0]

        elif self.pooling_mode == "mean":
            # Mean pooling over sequence (excluding padding)
            if attention_mask is not None:
                # Expand mask to match hidden states dimensions
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                # Sum of hidden states, masked
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                # Count of valid tokens per sequence
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = torch.mean(hidden_states, dim=1)

        elif self.pooling_mode == "mean_max":
            # Concatenate mean and max pooling
            # Mean pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_hidden / sum_mask
            else:
                mean_pooled = torch.mean(hidden_states, dim=1)

            # Max pooling
            if attention_mask is not None:
                hidden_states_copy = hidden_states.clone()
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states_copy[mask_expanded == 0] = -1e9
                max_pooled = torch.max(hidden_states_copy, dim=1)[0]
            else:
                max_pooled = torch.max(hidden_states, dim=1)[0]

            # Concatenate and project
            pooled = torch.cat([mean_pooled, max_pooled], dim=1)
            pooled = self.projection(pooled)

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        return pooled
