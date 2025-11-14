"""Quick test script to verify the embedding model pipeline works"""
import torch
import numpy as np
from src.models import EmbeddingModel
from src.data import SimpleTokenizer, PairDataset
from src.training import MultipleNegativesRankingLoss, EmbeddingTrainer
from src.evaluation import compute_similarity

def test_pipeline():
    """Test the complete pipeline"""
    print("=" * 60)
    print("Testing Language Embeddings Model Pipeline")
    print("=" * 60)

    # 1. Test tokenizer
    print("\n1. Testing Tokenizer...")
    sentences = [
        "The computer is fast",
        "I love programming",
        "Machine learning is interesting"
    ]

    tokenizer = SimpleTokenizer(vocab_size=100, max_length=32)
    tokenizer.fit(sentences)

    encoded = tokenizer.encode(sentences[0])
    print(f"   ✓ Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"   ✓ Encoded shape: {encoded['input_ids'].shape}")

    # 2. Test model initialization
    print("\n2. Testing Model Initialization...")
    model = EmbeddingModel(
        vocab_size=len(tokenizer),
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        ff_dim=128,
        max_seq_len=32,
        dropout=0.1,
        pooling_mode="mean",
        pad_token_id=tokenizer.pad_token_id
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {num_params:,} parameters")

    # 3. Test forward pass
    print("\n3. Testing Forward Pass...")
    model.eval()
    with torch.no_grad():
        output = model(encoded['input_ids'], encoded['attention_mask'])

    print(f"   ✓ Output embeddings shape: {output['embeddings'].shape}")
    print(f"   ✓ Hidden states shape: {output['hidden_states'].shape}")
    print(f"   ✓ Embedding L2 norm: {torch.norm(output['embeddings'][0]).item():.4f}")

    # 4. Test dataset
    print("\n4. Testing Dataset...")
    pairs = [
        ("sentence one", "sentence two"),
        ("another sentence", "different sentence"),
    ]
    dataset = PairDataset(pairs, tokenizer)
    print(f"   ✓ Dataset created with {len(dataset)} samples")

    # 5. Test loss function
    print("\n5. Testing Loss Function...")
    loss_fn = MultipleNegativesRankingLoss()

    # Create dummy embeddings
    emb1 = torch.randn(4, 64)
    emb2 = torch.randn(4, 64)
    loss = loss_fn(emb1, emb2)

    print(f"   ✓ Loss computed: {loss.item():.4f}")

    # 6. Test evaluation metrics
    print("\n6. Testing Evaluation Metrics...")
    emb_array1 = np.random.randn(5, 64)
    emb_array2 = np.random.randn(5, 64)

    similarities = compute_similarity(emb_array1, emb_array2)
    print(f"   ✓ Similarity matrix shape: {similarities.shape}")
    print(f"   ✓ Mean similarity: {similarities.mean():.4f}")

    # 7. Test training (1 epoch mini test)
    print("\n7. Testing Training Loop...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = EmbeddingTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cpu'
    )

    # Train for 1 epoch
    model.train()
    for batch in train_loader:
        output1 = model(batch["input_ids_1"], batch["attention_mask_1"])
        output2 = model(batch["input_ids_2"], batch["attention_mask_2"])

        loss = loss_fn(output1["embeddings"], output2["embeddings"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"   ✓ Training step completed, loss: {loss.item():.4f}")
        break  # Just test one batch

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
    print("\nThe pipeline is working correctly. You can now:")
    print("1. Run the Jupyter notebook for full training demo")
    print("2. Start implementing MoE components in src/experts/")
    print("3. Train on larger datasets")

if __name__ == "__main__":
    test_pipeline()
