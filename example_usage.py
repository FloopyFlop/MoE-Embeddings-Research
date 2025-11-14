"""
Example usage of the embeddings model for common tasks
"""
import torch
from src.models import EmbeddingModel
from src.data import SimpleTokenizer
from src.utils import (
    compute_cosine_similarity,
    find_similar_texts,
    encode_texts,
    visualize_embeddings_2d
)


def main():
    print("=" * 70)
    print("Example Usage of Language Embeddings Model")
    print("=" * 70)

    # 1. Setup: Create and train a simple model
    print("\n1. Setting up a simple model...")

    # Sample data
    training_texts = [
        "I love machine learning",
        "Deep learning is fascinating",
        "Neural networks are powerful",
        "Python is a great programming language",
        "I enjoy coding in Python",
        "The weather is nice today",
        "It's sunny outside",
        "Birds are singing",
        "Pizza is delicious",
        "I like Italian food",
    ]

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=200, max_length=64)
    tokenizer.fit(training_texts)
    print(f"   ✓ Tokenizer created with vocab size: {len(tokenizer)}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingModel(
        vocab_size=len(tokenizer),
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        pooling_mode="mean",
        pad_token_id=tokenizer.pad_token_id
    ).to(device)

    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   ✓ Using device: {device}")

    # 2. Compute similarity between two sentences
    print("\n2. Computing similarity between sentences...")

    text1 = "I love programming"
    text2 = "I enjoy coding"
    text3 = "The weather is nice"

    sim_12 = compute_cosine_similarity(text1, text2, model, tokenizer, device)
    sim_13 = compute_cosine_similarity(text1, text3, model, tokenizer, device)

    print(f"   Similarity between:")
    print(f"   '{text1}' and")
    print(f"   '{text2}': {sim_12:.4f}")
    print()
    print(f"   Similarity between:")
    print(f"   '{text1}' and")
    print(f"   '{text3}': {sim_13:.4f}")

    # 3. Find similar texts
    print("\n3. Finding similar texts to a query...")

    query = "I like programming languages"
    candidates = training_texts

    results = find_similar_texts(
        query, candidates, model, tokenizer, device, top_k=5
    )

    print(f"\n   Query: '{query}'")
    print(f"\n   Top 5 most similar texts:")
    for i, (text, score) in enumerate(results, 1):
        print(f"   {i}. [{score:.4f}] {text}")

    # 4. Encode multiple texts at once
    print("\n4. Batch encoding multiple texts...")

    texts_to_encode = [
        "Machine learning is fun",
        "The sky is blue",
        "Python programming"
    ]

    embeddings = encode_texts(texts_to_encode, model, tokenizer, device)
    print(f"   Encoded {len(texts_to_encode)} texts")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Each embedding has {embeddings.shape[1]} dimensions")

    # 5. Demonstrate clustering effect (even without training)
    print("\n5. Demonstrating semantic clustering...")

    tech_texts = [
        "Machine learning algorithms",
        "Neural network training",
        "Programming in Python"
    ]

    nature_texts = [
        "Beautiful sunny weather",
        "Trees and flowers",
        "Birds in the sky"
    ]

    food_texts = [
        "Delicious pizza",
        "Italian cuisine",
        "Tasty food"
    ]

    # Compute within-cluster and across-cluster similarities
    all_texts = tech_texts + nature_texts + food_texts
    all_embeddings = encode_texts(all_texts, model, tokenizer, device)

    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(all_embeddings)

    # Average within-cluster similarity
    tech_sim = sim_matrix[0:3, 0:3].mean()
    nature_sim = sim_matrix[3:6, 3:6].mean()
    food_sim = sim_matrix[6:9, 6:9].mean()

    # Average across-cluster similarity
    tech_nature_sim = sim_matrix[0:3, 3:6].mean()
    tech_food_sim = sim_matrix[0:3, 6:9].mean()
    nature_food_sim = sim_matrix[3:6, 6:9].mean()

    print(f"\n   Within-cluster similarities:")
    print(f"   - Tech cluster: {tech_sim:.4f}")
    print(f"   - Nature cluster: {nature_sim:.4f}")
    print(f"   - Food cluster: {food_sim:.4f}")

    print(f"\n   Across-cluster similarities:")
    print(f"   - Tech <-> Nature: {tech_nature_sim:.4f}")
    print(f"   - Tech <-> Food: {tech_food_sim:.4f}")
    print(f"   - Nature <-> Food: {nature_food_sim:.4f}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
    print("\nNote: This model is untrained, so similarities are random.")
    print("To get meaningful results, train the model using:")
    print("  - The Jupyter notebook: notebooks/demo_training_evaluation.ipynb")
    print("  - Or your own training script with real data")
    print("\nNext steps:")
    print("  1. Train on sentence pairs or triplets")
    print("  2. Evaluate on similarity benchmarks")
    print("  3. Implement MoE components for specialization")


if __name__ == "__main__":
    main()
