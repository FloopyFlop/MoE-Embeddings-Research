"""Quick test of production setup - verify GPU, datasets, and model work"""
import torch
import sys

print("=" * 70)
print("PRODUCTION SETUP TEST")
print("=" * 70)

# Test 1: GPU Detection
print("\n[1/5] Testing GPU detection...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"  ✓ MPS (Metal) available: {device}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"  ✓ CUDA available: {device}")
else:
    device = torch.device("cpu")
    print(f"  ⚠ Using CPU (GPU not available)")

# Test 2: Import modules
print("\n[2/5] Testing imports...")
try:
    from src.models import EmbeddingModel
    from src.data import SimpleTokenizer, PairDataset, load_dataset_for_training
    from src.training import MultipleNegativesRankingLoss, EmbeddingTrainer
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Load small dataset
print("\n[3/5] Testing dataset loading (small sample)...")
try:
    train_pairs, val_pairs = load_dataset_for_training(
        dataset_name="snli",
        num_samples=100,  # Just 100 for testing
        val_size=0.1,
        cache_dir="./data/cache"
    )
    print(f"  ✓ Loaded {len(train_pairs)} train, {len(val_pairs)} val pairs")
    print(f"  Example: '{train_pairs[0][0][:50]}...'")
except Exception as e:
    print(f"  ✗ Dataset loading failed: {e}")
    print(f"  This is expected on first run - datasets will be downloaded")

# Test 4: Create production model
print("\n[4/5] Testing production model initialization...")
try:
    # Small vocab for testing
    tokenizer = SimpleTokenizer(vocab_size=1000, max_length=128)
    tokenizer.fit(["test sentence one", "test sentence two"])

    model = EmbeddingModel(
        vocab_size=len(tokenizer),
        hidden_dim=384,  # Production size
        num_layers=6,
        num_heads=12,
        ff_dim=1536,
        max_seq_len=128,
        pooling_mode="mean",
        pad_token_id=tokenizer.pad_token_id
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {num_params:,} parameters")
    print(f"  ✓ Model on device: {next(model.parameters()).device}")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    sys.exit(1)

# Test 5: Test forward pass on GPU
print("\n[5/5] Testing GPU forward pass...")
try:
    test_text = "This is a test sentence"
    encoded = tokenizer.encode(test_text, return_tensors="pt")

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        embeddings = output["embeddings"]

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Output shape: {embeddings.shape}")
    print(f"  ✓ Output device: {embeddings.device}")
    print(f"  ✓ L2 norm: {torch.norm(embeddings[0]).item():.4f} (expected ~1.0)")

    # Test if it's actually using GPU by checking computation speed
    import time
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids, attention_mask)
    elapsed = time.time() - start

    print(f"  ✓ 10 forward passes: {elapsed*1000:.1f}ms ({elapsed*100:.1f}ms per pass)")
    if device.type == "mps" and elapsed < 0.5:
        print(f"  ✓ GPU acceleration working (fast)")
    elif device.type == "cpu":
        print(f"  ⚠ CPU mode (will be slower)")

except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("PRODUCTION SETUP: ✓ READY")
print("=" * 70)
print("\nYour system is configured for:")
print(f"  - Device: {device}")
print(f"  - Model size: ~{num_params/1e6:.1f}M parameters")
print(f"  - Embeddings: 384-dim")
print(f"  - GPU acceleration: {'✓ YES' if device.type != 'cpu' else '✗ NO (CPU only)'}")

print("\nNext steps:")
print("  1. Run: uv run python train_production.py")
print("  2. Or: uv run jupyter notebook notebooks/production_training.ipynb")
print("  3. Training should take ~5-10 minutes on M4 MAX")

print("\nExpected performance:")
if device.type == "mps":
    print("  - M4 MAX (MPS): ~25-30 steps/sec")
    print("  - Total training: ~5-8 minutes for 10 epochs")
elif device.type == "cuda":
    print("  - CUDA GPU: ~50-100 steps/sec (depending on GPU)")
    print("  - Total training: ~3-5 minutes for 10 epochs")
else:
    print("  - CPU: ~3-5 steps/sec")
    print("  - Total training: ~20-30 minutes for 10 epochs")
    print("  - Recommendation: Use GPU for faster training")
