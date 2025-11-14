"""Utility functions for embeddings model"""
import torch
import numpy as np
from typing import List, Tuple
import pickle
import os


def save_model(model, tokenizer, save_dir: str, model_name: str = "model"):
    """
    Save model and tokenizer to disk.

    Args:
        model: The embedding model
        tokenizer: The tokenizer
        save_dir: Directory to save to
        model_name: Base name for saved files
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)

    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, f"{model_name}_tokenizer.pkl")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")


def load_model(model_class, model_config: dict, save_dir: str, model_name: str = "model", device: str = "cpu"):
    """
    Load model and tokenizer from disk.

    Args:
        model_class: The model class (e.g., EmbeddingModel)
        model_config: Configuration dictionary for model initialization
        save_dir: Directory to load from
        model_name: Base name for saved files
        device: Device to load model to

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load model
    model = model_class(**model_config)
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = os.path.join(save_dir, f"{model_name}_tokenizer.pkl")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    print(f"Model loaded from {model_path}")
    print(f"Tokenizer loaded from {tokenizer_path}")

    return model, tokenizer


def encode_texts(texts: List[str], model, tokenizer, device: str = "cpu", batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of texts to embeddings.

    Args:
        texts: List of text strings
        model: Embedding model
        tokenizer: Tokenizer
        device: Device to run on
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape [len(texts), embedding_dim]
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Encode batch
            encoded = tokenizer.encode(batch_texts, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Get embeddings
            output = model(input_ids, attention_mask)
            embeddings = output["embeddings"] if isinstance(output, dict) else output

            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def compute_cosine_similarity(text1: str, text2: str, model, tokenizer, device: str = "cpu") -> float:
    """
    Compute cosine similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        model: Embedding model
        tokenizer: Tokenizer
        device: Device to run on

    Returns:
        Cosine similarity score (0-1)
    """
    model.eval()

    with torch.no_grad():
        # Encode texts
        enc1 = tokenizer.encode(text1, return_tensors="pt")
        enc2 = tokenizer.encode(text2, return_tensors="pt")

        emb1 = model(enc1["input_ids"].to(device), enc1["attention_mask"].to(device))
        emb2 = model(enc2["input_ids"].to(device), enc2["attention_mask"].to(device))

        if isinstance(emb1, dict):
            emb1 = emb1["embeddings"]
        if isinstance(emb2, dict):
            emb2 = emb2["embeddings"]

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

    return similarity


def find_similar_texts(
    query: str,
    candidates: List[str],
    model,
    tokenizer,
    device: str = "cpu",
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find most similar texts to a query.

    Args:
        query: Query text
        candidates: List of candidate texts
        model: Embedding model
        tokenizer: Tokenizer
        device: Device to run on
        top_k: Number of results to return

    Returns:
        List of (text, similarity_score) tuples, sorted by similarity
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Encode query
    query_emb = encode_texts([query], model, tokenizer, device)

    # Encode candidates
    candidate_embs = encode_texts(candidates, model, tokenizer, device)

    # Compute similarities
    similarities = cosine_similarity(query_emb, candidate_embs)[0]

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = [(candidates[idx], similarities[idx]) for idx in top_indices]

    return results


def create_embeddings_index(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cpu",
    batch_size: int = 32
) -> Tuple[np.ndarray, List[str]]:
    """
    Create an embedding index for fast similarity search.

    Args:
        texts: List of texts to index
        model: Embedding model
        tokenizer: Tokenizer
        device: Device to run on
        batch_size: Batch size for encoding

    Returns:
        Tuple of (embeddings_array, texts_list)
    """
    print(f"Creating index for {len(texts)} texts...")
    embeddings = encode_texts(texts, model, tokenizer, device, batch_size)
    print(f"Index created with shape {embeddings.shape}")

    return embeddings, texts


def search_index(
    query: str,
    index_embeddings: np.ndarray,
    index_texts: List[str],
    model,
    tokenizer,
    device: str = "cpu",
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Search an embedding index.

    Args:
        query: Query text
        index_embeddings: Pre-computed embeddings array
        index_texts: Corresponding texts
        model: Embedding model
        tokenizer: Tokenizer
        device: Device to run on
        top_k: Number of results

    Returns:
        List of (text, similarity_score) tuples
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Encode query
    query_emb = encode_texts([query], model, tokenizer, device)

    # Compute similarities
    similarities = cosine_similarity(query_emb, index_embeddings)[0]

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = [(index_texts[idx], similarities[idx]) for idx in top_indices]

    return results


def batch_similarity_matrix(
    texts1: List[str],
    texts2: List[str],
    model,
    tokenizer,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute pairwise similarity matrix between two lists of texts.

    Args:
        texts1: First list of texts
        texts2: Second list of texts
        model: Embedding model
        tokenizer: Tokenizer
        device: Device to run on

    Returns:
        Similarity matrix of shape [len(texts1), len(texts2)]
    """
    from sklearn.metrics.pairwise import cosine_similarity

    embs1 = encode_texts(texts1, model, tokenizer, device)
    embs2 = encode_texts(texts2, model, tokenizer, device)

    return cosine_similarity(embs1, embs2)


def visualize_embeddings_2d(
    texts: List[str],
    model,
    tokenizer,
    labels: List[str] = None,
    device: str = "cpu",
    method: str = "tsne"
):
    """
    Visualize embeddings in 2D.

    Args:
        texts: List of texts
        model: Embedding model
        tokenizer: Tokenizer
        labels: Optional labels for coloring
        device: Device to run on
        method: "tsne" or "pca"
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Get embeddings
    embeddings = encode_texts(texts, model, tokenizer, device)

    # Reduce to 2D
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))

    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

        for i, (text, label) in enumerate(zip(texts, labels)):
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                       c=[label_to_color[label]], s=100, alpha=0.6)
            plt.annotate(text[:30], (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=8, alpha=0.7)

        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=label_to_color[label], markersize=10)
                  for label in unique_labels]
        plt.legend(handles, unique_labels)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.6)
        for i, text in enumerate(texts):
            plt.annotate(text[:30], (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=8, alpha=0.7)

    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.title("2D Visualization of Embeddings")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
