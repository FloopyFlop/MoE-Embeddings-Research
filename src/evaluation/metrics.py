"""Evaluation metrics for embedding models"""
import torch
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F


def compute_similarity(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compute pairwise similarity between embeddings.

    Args:
        embeddings1: [n, dim]
        embeddings2: [m, dim]
        metric: "cosine" or "euclidean"

    Returns:
        [n, m] similarity matrix
    """
    if metric == "cosine":
        return cosine_similarity(embeddings1, embeddings2)
    elif metric == "euclidean":
        # Negative euclidean distance (higher is more similar)
        diff = embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return -distances
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    relevance_labels: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance using various metrics.

    Args:
        query_embeddings: [num_queries, dim]
        doc_embeddings: [num_docs, dim]
        relevance_labels: [num_queries, num_docs] - binary relevance matrix
        k_values: List of k values for top-k metrics

    Returns:
        Dictionary with metrics (precision@k, recall@k, mrr, map)
    """
    # Compute similarity matrix
    similarities = compute_similarity(query_embeddings, doc_embeddings, metric="cosine")

    # Get rankings (indices sorted by similarity)
    rankings = np.argsort(-similarities, axis=1)  # Descending order

    results = {}

    # Precision@k and Recall@k
    for k in k_values:
        precisions = []
        recalls = []

        for i in range(len(query_embeddings)):
            top_k = rankings[i, :k]
            relevant = relevance_labels[i]
            num_relevant = np.sum(relevant)

            if num_relevant == 0:
                continue

            retrieved_relevant = np.sum(relevant[top_k])

            precision = retrieved_relevant / k
            recall = retrieved_relevant / num_relevant

            precisions.append(precision)
            recalls.append(recall)

        results[f"precision@{k}"] = np.mean(precisions)
        results[f"recall@{k}"] = np.mean(recalls)

    # Mean Reciprocal Rank (MRR)
    mrr_scores = []
    for i in range(len(query_embeddings)):
        relevant_docs = np.where(relevance_labels[i] == 1)[0]
        if len(relevant_docs) == 0:
            continue

        # Find rank of first relevant document
        for rank, doc_idx in enumerate(rankings[i], start=1):
            if doc_idx in relevant_docs:
                mrr_scores.append(1.0 / rank)
                break

    results["mrr"] = np.mean(mrr_scores) if mrr_scores else 0.0

    # Mean Average Precision (MAP)
    ap_scores = []
    for i in range(len(query_embeddings)):
        relevant_docs = np.where(relevance_labels[i] == 1)[0]
        num_relevant = len(relevant_docs)

        if num_relevant == 0:
            continue

        # Calculate average precision for this query
        precisions_at_relevant = []
        num_retrieved_relevant = 0

        for rank, doc_idx in enumerate(rankings[i], start=1):
            if doc_idx in relevant_docs:
                num_retrieved_relevant += 1
                precision_at_rank = num_retrieved_relevant / rank
                precisions_at_relevant.append(precision_at_rank)

        ap = np.mean(precisions_at_relevant) if precisions_at_relevant else 0.0
        ap_scores.append(ap)

    results["map"] = np.mean(ap_scores) if ap_scores else 0.0

    return results


def evaluate_classification(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    k: int = 5
) -> Dict[str, float]:
    """
    Evaluate embeddings using k-NN classification.

    Args:
        train_embeddings: [n_train, dim]
        train_labels: [n_train]
        test_embeddings: [n_test, dim]
        test_labels: [n_test]
        k: Number of neighbors for k-NN

    Returns:
        Dictionary with classification metrics
    """
    from sklearn.neighbors import KNeighborsClassifier

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_embeddings, train_labels)

    # Predict
    predictions = knn.predict(test_embeddings)

    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted'
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_semantic_similarity(
    model: torch.nn.Module,
    sentence_pairs: List[Tuple[str, str]],
    similarity_scores: List[float],
    tokenizer,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate model on semantic similarity task (e.g., STS benchmark).

    Args:
        model: Embedding model
        sentence_pairs: List of (sentence1, sentence2) pairs
        similarity_scores: Ground truth similarity scores (0-1 or 0-5)
        tokenizer: Tokenizer
        device: Device to run on

    Returns:
        Dictionary with correlation metrics
    """
    from scipy.stats import pearsonr, spearmanr

    model.eval()
    predicted_similarities = []

    with torch.no_grad():
        for sent1, sent2 in sentence_pairs:
            # Encode sentences
            encoding1 = tokenizer.encode(sent1, return_tensors="pt")
            encoding2 = tokenizer.encode(sent2, return_tensors="pt")

            input_ids1 = encoding1["input_ids"].to(device)
            mask1 = encoding1["attention_mask"].to(device)
            input_ids2 = encoding2["input_ids"].to(device)
            mask2 = encoding2["attention_mask"].to(device)

            # Get embeddings
            emb1 = model(input_ids1, mask1)
            emb2 = model(input_ids2, mask2)

            if isinstance(emb1, dict):
                emb1 = emb1["embeddings"]
            if isinstance(emb2, dict):
                emb2 = emb2["embeddings"]

            # Compute cosine similarity
            sim = F.cosine_similarity(emb1, emb2).item()
            predicted_similarities.append(sim)

    # Compute correlations
    pearson_corr, _ = pearsonr(similarity_scores, predicted_similarities)
    spearman_corr, _ = spearmanr(similarity_scores, predicted_similarities)

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics about embeddings (for analysis).

    Args:
        embeddings: [n, dim]

    Returns:
        Dictionary with statistics
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1)

    # Compute pairwise similarities
    similarities = cosine_similarity(embeddings)

    # Remove diagonal (self-similarity)
    n = similarities.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diagonal_sims = similarities[mask]

    return {
        "mean_norm": np.mean(norms),
        "std_norm": np.std(norms),
        "mean_similarity": np.mean(off_diagonal_sims),
        "std_similarity": np.std(off_diagonal_sims),
        "min_similarity": np.min(off_diagonal_sims),
        "max_similarity": np.max(off_diagonal_sims)
    }
