"""
Evaluate all trained models to verify they work correctly.
This script loads checkpoints and computes similarity scores.
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from tqdm import tqdm
import pickle
import os

from src.models import EmbeddingModel, EmbeddingModelMoE
from src.data import SimpleTokenizer

print("=" * 70)
print("Model Evaluation Script")
print("=" * 70)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}\n")

# Test sentences
test_pairs = [
    ("The cat sat on the mat", "A feline rested on the rug"),
    ("I love programming", "I enjoy coding"),
    ("The weather is nice today", "It's sunny outside"),
    ("Machine learning is fascinating", "AI is interesting"),
    ("Python is a programming language", "JavaScript is used for web development"),
]

def load_model_and_evaluate(model_name, checkpoint_path, tokenizer_path):
    """Load a model checkpoint and evaluate it"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name.upper()}")
    print(f"{'='*70}")

    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print(f"  Model needs to be trained first!")
        return None

    if not os.path.exists(tokenizer_path):
        print(f"⚠ Tokenizer not found: {tokenizer_path}")
        return None

    try:
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)

        print(f"Vocab size: {len(tokenizer.token2id)}")

        # Check training history
        if 'history' in checkpoint:
            history = checkpoint['history']
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])

            if train_loss:
                final_train = train_loss[-1]
                print(f"Final train loss: {final_train:.4f}")

                if np.isnan(final_train):
                    print(f"  ✗ ERROR: Training loss is NaN!")
                    print(f"  This model was trained with corrupted gradients.")
                    return {
                        'model_name': model_name,
                        'status': 'CORRUPTED',
                        'train_loss': np.nan,
                        'val_loss': np.nan,
                        'similarity': np.nan
                    }

            if val_loss:
                final_val = val_loss[-1]
                print(f"Final val loss: {final_val:.4f}")

        # Get config
        config = checkpoint.get('config', {})

        # Create model based on type
        if model_name == 'dense':
            model = EmbeddingModel(
                vocab_size=len(tokenizer.token2id),
                hidden_dim=config.get('hidden_dim', 384),
                num_layers=config.get('num_layers', 6),
                num_heads=config.get('num_heads', 12),
                ff_dim=config.get('ff_dim', 1024),
                max_seq_len=config.get('max_seq_len', 128),
                dropout=config.get('dropout', 0.1),
                pooling_mode=config.get('pooling_mode', 'mean'),
                pad_token_id=tokenizer.pad_token_id
            )
        elif model_name == 'moe':
            model = EmbeddingModelMoE(
                vocab_size=len(tokenizer.token2id),
                hidden_dim=config.get('hidden_dim', 384),
                num_layers=config.get('num_layers', 6),
                num_heads=config.get('num_heads', 12),
                ff_dim=config.get('ff_dim', 1024),
                num_experts=config.get('num_experts', 8),
                top_k=config.get('top_k', 2),
                max_seq_len=config.get('max_seq_len', 128),
                dropout=config.get('dropout', 0.1),
                pooling_mode=config.get('pooling_mode', 'mean'),
                pad_token_id=tokenizer.pad_token_id
            )
        else:
            print(f"⚠ Unknown model type: {model_name}")
            print(f"  Skipping specialized models (MoRE, Variable-Size, MoA)")
            return None

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print(f"Model loaded successfully")
        param_counts = model.count_parameters()
        print(f"Parameters: {param_counts['total']:,}")

        # Evaluate on test pairs
        print(f"\nEvaluating on {len(test_pairs)} test pairs...")
        similarities = []

        with torch.no_grad():
            for s1, s2 in test_pairs:
                # Encode
                tokens1 = tokenizer.encode(s1)
                tokens2 = tokenizer.encode(s2)

                input_ids1 = tokens1["input_ids"].to(device)
                input_ids2 = tokens2["input_ids"].to(device)
                mask1 = tokens1["attention_mask"].to(device)
                mask2 = tokens2["attention_mask"].to(device)

                # Get embeddings
                emb1 = model(input_ids1, mask1, return_dict=False)
                emb2 = model(input_ids2, mask2, return_dict=False)

                # Check for NaN
                if torch.isnan(emb1).any() or torch.isnan(emb2).any():
                    print(f"  ✗ NaN detected in embeddings!")
                    print(f"    Pair: '{s1[:30]}...' <-> '{s2[:30]}...'")
                    similarities.append(np.nan)
                    continue

                # Compute similarity
                emb1_np = emb1.cpu().numpy()
                emb2_np = emb2.cpu().numpy()
                sim = np.dot(emb1_np[0], emb2_np[0])
                similarities.append(sim)

        # Check results
        has_nan = any(np.isnan(s) for s in similarities)

        if has_nan:
            print(f"\n✗ FAILED: Model produces NaN embeddings")
            nan_count = sum(1 for s in similarities if np.isnan(s))
            print(f"  NaN count: {nan_count}/{len(similarities)}")
            return {
                'model_name': model_name,
                'status': 'FAILED_NAN',
                'train_loss': train_loss[-1] if train_loss else np.nan,
                'val_loss': val_loss[-1] if val_loss else np.nan,
                'similarity': np.nan,
                'nan_count': nan_count
            }

        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        print(f"\n✓ SUCCESS")
        print(f"  Average similarity: {avg_sim:.4f} ± {std_sim:.4f}")
        print(f"  Similarity range: [{min(similarities):.4f}, {max(similarities):.4f}]")

        # Show individual similarities
        print(f"\nIndividual similarities:")
        for (s1, s2), sim in zip(test_pairs, similarities):
            print(f"  {sim:.4f}: '{s1[:30]}...' <-> '{s2[:30]}...'")

        return {
            'model_name': model_name,
            'status': 'SUCCESS',
            'train_loss': train_loss[-1] if train_loss else None,
            'val_loss': val_loss[-1] if val_loss else None,
            'similarity': avg_sim,
            'similarity_std': std_sim,
            'similarities': similarities
        }

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model_name': model_name,
            'status': 'ERROR',
            'error': str(e)
        }

# Models to evaluate
models = [
    ('dense', 'checkpoints/dense_epoch_10.pt', 'checkpoints/dense_tokenizer.pkl'),
    ('moe', 'checkpoints/moe_epoch_10.pt', 'checkpoints/moe_tokenizer.pkl'),
    # Note: MoRE, Variable-Size, and MoA would need custom model classes
]

# Evaluate all models
results = []
for model_name, checkpoint_path, tokenizer_path in models:
    result = load_model_and_evaluate(model_name, checkpoint_path, tokenizer_path)
    if result:
        results.append(result)

# Check for specialized models (may not have loaders yet)
specialized_models = [
    ('more', 'checkpoints/more_epoch_10.pt', 'checkpoints/more_tokenizer.pkl'),
    ('variable_experts', 'checkpoints/variable_experts_epoch_10.pt', 'checkpoints/variable_experts_tokenizer.pkl'),
    ('moa', 'checkpoints/moa_epoch_10.pt', 'checkpoints/moa_tokenizer.pkl'),
]

print(f"\n{'='*70}")
print("Checking specialized models...")
print(f"{'='*70}")

for model_name, checkpoint_path, tokenizer_path in specialized_models:
    print(f"\n{model_name.upper()}:")
    if not os.path.exists(checkpoint_path):
        print(f"  ⚠ Not found - needs training")
    else:
        # Just check if checkpoint is corrupted
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            history = checkpoint.get('history', {})
            train_loss = history.get('train_loss', [])

            if train_loss:
                final_train = train_loss[-1]
                if np.isnan(final_train):
                    print(f"  ✗ CORRUPTED (NaN loss from training)")
                    print(f"  → Delete checkpoint and retrain!")
                else:
                    print(f"  ✓ Checkpoint exists (train loss: {final_train:.4f})")
                    print(f"  → Use notebook to evaluate")
        except Exception as e:
            print(f"  ✗ Error loading: {e}")

# Summary
print(f"\n{'='*70}")
print("EVALUATION SUMMARY")
print(f"{'='*70}\n")

if results:
    print(f"{'Model':<15} {'Status':<15} {'Train Loss':<12} {'Val Loss':<12} {'Similarity':<12}")
    print("-" * 70)

    for result in results:
        status = result['status']
        train_loss = f"{result.get('train_loss', np.nan):.4f}" if result.get('train_loss') is not None else 'N/A'
        val_loss = f"{result.get('val_loss', np.nan):.4f}" if result.get('val_loss') is not None else 'N/A'
        similarity = f"{result.get('similarity', np.nan):.4f}" if result.get('similarity') is not None else 'N/A'

        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {result['model_name']:<13} {status:<15} {train_loss:<12} {val_loss:<12} {similarity:<12}")

    # Check for any failures
    failures = [r for r in results if r['status'] != 'SUCCESS']
    if failures:
        print(f"\n⚠ {len(failures)} model(s) failed evaluation")
        for r in failures:
            print(f"  - {r['model_name']}: {r['status']}")
    else:
        print(f"\n✓ All evaluated models passed!")
else:
    print("No models were successfully evaluated.")

print("\n" + "=" * 70)
