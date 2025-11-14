"""Loss functions for training embedding models"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning embeddings.
    Brings similar pairs closer and pushes dissimilar pairs apart.
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for negative pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings1: [batch_size, embedding_dim]
            embeddings2: [batch_size, embedding_dim]
            labels: [batch_size] - 1 for similar pairs, 0 for dissimilar

        Returns:
            Scalar loss value
        """
        # Euclidean distance
        distances = F.pairwise_distance(embeddings1, embeddings2)

        # Loss for positive pairs (label=1): minimize distance
        positive_loss = labels * torch.pow(distances, 2)

        # Loss for negative pairs (label=0): maximize distance up to margin
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        loss = torch.mean(positive_loss + negative_loss)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss: ensures anchor is closer to positive than to negative by a margin.
    L = max(d(a,p) - d(a,n) + margin, 0)
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Minimum distance between positive and negative pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: [batch_size, embedding_dim]
            positive: [batch_size, embedding_dim]
            negative: [batch_size, embedding_dim]

        Returns:
            Scalar loss value
        """
        # Distances
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)

        # Triplet loss
        loss = torch.mean(torch.clamp(pos_distance - neg_distance + self.margin, min=0.0))
        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss (InfoNCE / NT-Xent).
    Uses in-batch negatives for efficient contrastive learning.
    This is the loss used by sentence-transformers and SimCLR.

    For each positive pair (a, p), all other examples in the batch serve as negatives.
    """

    def __init__(self, temperature: float = 0.05, use_cosine: bool = True):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            use_cosine: Use cosine similarity (True) or dot product (False)
        """
        super().__init__()
        self.temperature = temperature
        self.use_cosine = use_cosine
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings1: [batch_size, embedding_dim] - anchors
            embeddings2: [batch_size, embedding_dim] - positives

        Returns:
            Scalar loss value
        """
        batch_size = embeddings1.size(0)

        # Compute similarity matrix
        if self.use_cosine:
            # Normalize embeddings for cosine similarity
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Similarity matrix: [batch_size, batch_size]
        # similarities[i,j] = similarity between embeddings1[i] and embeddings2[j]
        similarities = torch.matmul(embeddings1, embeddings2.t()) / self.temperature

        # Labels: diagonal elements are positives (i matches with i)
        labels = torch.arange(batch_size, device=embeddings1.device)

        # Cross-entropy loss
        # For each anchor i, positive is at position i, all others are negatives
        loss = self.cross_entropy(similarities, labels)

        return loss


class CosineSimilarityLoss(nn.Module):
    """
    Simple cosine similarity loss.
    Maximizes cosine similarity for positive pairs.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings1: [batch_size, embedding_dim]
            embeddings2: [batch_size, embedding_dim]
            labels: [batch_size] - target similarity scores (e.g., 0-1)

        Returns:
            Scalar loss value
        """
        # Cosine similarity
        cosine_sim = F.cosine_similarity(embeddings1, embeddings2)

        # MSE loss between predicted similarity and target labels
        loss = F.mse_loss(cosine_sim, labels)

        return loss
