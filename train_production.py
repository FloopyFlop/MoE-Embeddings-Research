"""Production training script for real embeddings model"""
import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add configs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'configs'))

from configs.production_config import (
    MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG, LOSS_CONFIG,
    SCHEDULER_CONFIG, PATHS, get_device, print_config, estimate_training_time
)

from src.models import EmbeddingModel
from src.data import SimpleTokenizer, PairDataset, load_dataset_for_training
from src.training import MultipleNegativesRankingLoss, TripletLoss, ContrastiveLoss, EmbeddingTrainer
from src.evaluation import compute_similarity, evaluate_retrieval
from src.utils import save_model


def main():
    print("\n" + "=" * 70)
    print("PRODUCTION EMBEDDINGS MODEL TRAINING")
    print("=" * 70)

    # Print configuration
    print_config()

    # Get device
    device = get_device()

    # Create directories
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

    # Step 1: Load datasets
    print("\n" + "=" * 70)
    print("STEP 1: Loading Real Datasets")
    print("=" * 70)

    train_pairs, val_pairs = load_dataset_for_training(
        dataset_name=DATA_CONFIG["dataset_name"],
        num_samples=DATA_CONFIG["num_train_samples"],
        val_size=DATA_CONFIG["val_size"],
        cache_dir=DATA_CONFIG["cache_dir"]
    )

    print(f"\nDataset loaded successfully!")
    print(f"  Training pairs: {len(train_pairs):,}")
    print(f"  Validation pairs: {len(val_pairs):,}")

    # Step 2: Build tokenizer
    print("\n" + "=" * 70)
    print("STEP 2: Building Tokenizer")
    print("=" * 70)

    # Extract all unique sentences
    all_sentences = []
    for s1, s2 in train_pairs[:50000]:  # Use subset for vocab building
        all_sentences.extend([s1, s2])

    print(f"Building vocabulary from {len(all_sentences):,} sentences...")

    tokenizer = SimpleTokenizer(
        vocab_size=DATA_CONFIG["tokenizer_vocab_size"],
        max_length=MODEL_CONFIG["max_seq_len"]
    )
    tokenizer.fit(all_sentences)

    print(f"Tokenizer built with {len(tokenizer):,} tokens")

    # Update model config with actual vocab size
    MODEL_CONFIG["vocab_size"] = len(tokenizer)
    MODEL_CONFIG["pad_token_id"] = tokenizer.pad_token_id

    # Step 3: Create datasets
    print("\n" + "=" * 70)
    print("STEP 3: Creating PyTorch Datasets")
    print("=" * 70)

    train_dataset = PairDataset(train_pairs, tokenizer)
    val_dataset = PairDataset(val_pairs, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,  # Set to 0 for MPS compatibility
        pin_memory=False  # Disable for MPS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"DataLoaders created:")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")

    # Step 4: Initialize model
    print("\n" + "=" * 70)
    print("STEP 4: Initializing Model")
    print("=" * 70)

    model = EmbeddingModel(**MODEL_CONFIG).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model initialized:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    print(f"  Model size: ~{num_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Hidden dim: {MODEL_CONFIG['hidden_dim']}")
    print(f"  Num layers: {MODEL_CONFIG['num_layers']}")
    print(f"  Num heads: {MODEL_CONFIG['num_heads']}")

    # Step 5: Setup training
    print("\n" + "=" * 70)
    print("STEP 5: Setting Up Training")
    print("=" * 70)

    # Loss function
    if LOSS_CONFIG["loss_type"] == "multiple_negatives_ranking":
        loss_fn = MultipleNegativesRankingLoss(temperature=LOSS_CONFIG["temperature"])
    elif LOSS_CONFIG["loss_type"] == "triplet":
        loss_fn = TripletLoss(margin=LOSS_CONFIG["margin"])
    else:
        loss_fn = ContrastiveLoss(margin=LOSS_CONFIG["margin"])

    print(f"Loss function: {loss_fn.__class__.__name__}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )

    # Scheduler
    if SCHEDULER_CONFIG["scheduler_type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAINING_CONFIG["num_epochs"],
            eta_min=SCHEDULER_CONFIG["eta_min"]
        )
    elif SCHEDULER_CONFIG["scheduler_type"] == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=SCHEDULER_CONFIG.get("warmup_steps", 1000)
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=SCHEDULER_CONFIG["patience"],
            factor=SCHEDULER_CONFIG["factor"]
        )

    print(f"Optimizer: AdamW (lr={TRAINING_CONFIG['learning_rate']})")
    print(f"Scheduler: {scheduler.__class__.__name__}")

    # Trainer
    trainer = EmbeddingTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    # Estimate training time
    estimate_training_time(
        len(train_pairs),
        TRAINING_CONFIG["batch_size"],
        TRAINING_CONFIG["num_epochs"]
    )

    # Step 6: Train!
    print("\n" + "=" * 70)
    print("STEP 6: Training Model")
    print("=" * 70)
    print("Starting training...\n")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=TRAINING_CONFIG["num_epochs"],
        eval_every=TRAINING_CONFIG["eval_every"],
        save_best=True,
        save_path=os.path.join(PATHS["model_save_dir"], "best_model.pt")
    )

    # Step 7: Save final model
    print("\n" + "=" * 70)
    print("STEP 7: Saving Model")
    print("=" * 70)

    save_model(
        model,
        tokenizer,
        PATHS["model_save_dir"],
        model_name="production_model"
    )

    # Save config and history
    import pickle
    with open(os.path.join(PATHS["model_save_dir"], "config.pkl"), "wb") as f:
        pickle.dump({
            "model_config": MODEL_CONFIG,
            "data_config": DATA_CONFIG,
            "training_config": TRAINING_CONFIG,
            "history": history
        }, f)

    print("\nModel and configuration saved!")

    # Step 8: Quick evaluation
    print("\n" + "=" * 70)
    print("STEP 8: Quick Evaluation")
    print("=" * 70)

    model.eval()
    test_pairs = [
        ("A man is playing guitar", "A person is playing a musical instrument"),
        ("A dog is running in the park", "A canine is jogging outdoors"),
        ("The weather is sunny", "It's a beautiful day"),
        ("I love programming", "I enjoy writing code"),
        ("The car is fast", "Pizza is delicious"),
    ]

    print("\nSimilarity tests:")
    with torch.no_grad():
        for s1, s2 in test_pairs:
            enc1 = tokenizer.encode(s1, return_tensors="pt")
            enc2 = tokenizer.encode(s2, return_tensors="pt")

            emb1 = model(enc1["input_ids"].to(device), enc1["attention_mask"].to(device))["embeddings"]
            emb2 = model(enc2["input_ids"].to(device), enc2["attention_mask"].to(device))["embeddings"]

            sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()

            print(f"\nSimilarity: {sim:.4f}")
            print(f"  S1: {s1}")
            print(f"  S2: {s2}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nModel saved to: {PATHS['model_save_dir']}")
    print(f"Best model: {os.path.join(PATHS['model_save_dir'], 'best_model.pt')}")
    print(f"Final model: {os.path.join(PATHS['model_save_dir'], 'production_model.pt')}")
    print("\nNext steps:")
    print("  1. Run the evaluation notebook")
    print("  2. Test on downstream tasks")
    print("  3. Implement MoE components")


if __name__ == "__main__":
    main()
