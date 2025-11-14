"""Expert network implementation for Mixture of Experts"""
import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    Single expert network - a specialized feed-forward network.
    Each expert can specialize in different linguistic patterns or domains.
    """

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Input/output dimension
            ff_dim: Hidden dimension of the expert
            dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] or [batch_size * seq_len, hidden_dim]
        Returns:
            [same shape as input]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SparseExpert(nn.Module):
    """
    Sparse expert with smaller capacity for more experts.
    Uses a bottleneck architecture to reduce parameters.
    """

    def __init__(self, hidden_dim: int, ff_dim: int, bottleneck_factor: float = 0.5, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Input/output dimension
            ff_dim: Intermediate dimension
            bottleneck_factor: Reduction factor for bottleneck (e.g., 0.5 = half size)
            dropout: Dropout probability
        """
        super().__init__()
        bottleneck_dim = int(ff_dim * bottleneck_factor)

        self.fc1 = nn.Linear(hidden_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
        Returns:
            Output tensor (same shape as input)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
