"""Transformer encoder with MoE layers"""
import torch
import torch.nn as nn
from .encoder import MultiHeadAttention
from ..experts import MoELayer


class TransformerLayerMoE(nn.Module):
    """
    Transformer layer with MoE feed-forward instead of standard FFN.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # MoE layer instead of standard feed-forward
        self.moe_layer = MoELayer(
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        # Self-attention with residual
        attn_output = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attn_output)

        # MoE with residual
        moe_output = self.moe_layer(self.norm2(x))
        x = x + self.dropout(moe_output)

        return x

    def get_aux_loss(self):
        """Get auxiliary loss from MoE layer"""
        return self.moe_layer.get_aux_loss()


class TransformerEncoderMoE(nn.Module):
    """
    Transformer encoder with MoE layers.
    Designed for fair comparison with standard encoder.
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
        pad_token_id: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.num_experts = num_experts
        self.top_k = top_k

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)

        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # MoE Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayerMoE(hidden_dim, num_heads, ff_dim, num_experts, top_k, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)

        x = token_embeds + position_embeds
        x = self.dropout(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)

        return x

    def get_aux_loss(self):
        """
        Get total auxiliary loss from all MoE layers.
        Should be added to main loss during training.
        """
        aux_loss = 0.0
        count = 0
        for layer in self.layers:
            layer_aux_loss = layer.get_aux_loss()
            if layer_aux_loss is not None:
                aux_loss += layer_aux_loss
                count += 1

        return aux_loss if count > 0 else None

    def get_expert_usage_stats(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> dict:
        """
        Get statistics on expert usage across all layers.

        Returns:
            Dictionary with usage statistics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Forward pass to get hidden states
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        x = token_embeds + position_embeds

        stats = {}
        for layer_idx, layer in enumerate(self.layers):
            # Get expert usage for this layer
            usage = layer.moe_layer.get_expert_usage(x)
            stats[f"layer_{layer_idx}"] = usage.cpu().numpy()

            # Forward through layer for next iteration
            x = layer(x, attention_mask)

        return stats
