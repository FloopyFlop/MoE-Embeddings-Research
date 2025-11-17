"""Complete embedding model combining encoder and pooling"""
import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .pooling import Pooler
from typing import Dict, Optional, Literal


class EmbeddingModel(nn.Module):
    """
    Complete sentence embedding model.
    Modular architecture designed for easy MoE integration in the encoder or pooling layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pooling_mode: Literal["mean", "max", "cls", "mean_max"] = "mean",
        pad_token_id: int = 0,
        normalize_embeddings: bool = True
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Dimension of hidden states and final embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pooling_mode: Pooling strategy for final embeddings
            pad_token_id: Token ID for padding
            normalize_embeddings: Whether to L2-normalize final embeddings
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.normalize_embeddings = normalize_embeddings

        # Encoder: processes input tokens to contextualized representations
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id
        )

        # Pooler: converts sequence to fixed-size embedding
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
            input_ids: [batch_size, seq_len] - token indices
            attention_mask: [batch_size, seq_len] - attention mask (1 for valid, 0 for padding)
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing:
                - embeddings: [batch_size, hidden_dim] - final sentence embeddings
                - hidden_states: [batch_size, seq_len, hidden_dim] - sequence outputs
        """
        # Get sequence of hidden states from encoder
        hidden_states = self.encoder(input_ids, attention_mask)

        # Pool to get sentence embedding
        embeddings = self.pooler(hidden_states, attention_mask)

        # L2 normalization for cosine similarity
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if return_dict:
            return {
                "embeddings": embeddings,
                "hidden_states": hidden_states
            }
        else:
            return embeddings

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method for encoding that returns only embeddings.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            [batch_size, hidden_dim] - sentence embeddings
        """
        return self.forward(input_ids, attention_mask, return_dict=False)

    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Get embeddings in evaluation mode with batching.
        Useful for encoding large datasets.

        Args:
            input_ids: [num_samples, seq_len]
            attention_mask: [num_samples, seq_len]
            batch_size: Batch size for encoding

        Returns:
            [num_samples, hidden_dim] - embeddings for all samples
        """
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

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in the model.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "total": total_params,
            "active": total_params  # Dense model uses all parameters
        }
