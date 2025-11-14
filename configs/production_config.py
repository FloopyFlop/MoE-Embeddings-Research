"""Production configuration for real embeddings training"""

# Device configuration
DEVICE_CONFIG = {
    "use_mps": True,  # Use Metal Performance Shaders for Mac M4
    "use_cuda": False,  # CUDA for NVIDIA GPUs
    "fallback_cpu": True,  # Fallback to CPU if GPU unavailable
}

# Model architecture - scaled up for real training
MODEL_CONFIG = {
    "vocab_size": None,  # Will be set after tokenizer is built
    "hidden_dim": 384,  # Larger embeddings (SBERT uses 384-768)
    "num_layers": 6,  # 6 transformer layers
    "num_heads": 12,  # 12 attention heads (divisible by hidden_dim)
    "ff_dim": 1536,  # 4x hidden_dim (standard transformer ratio)
    "max_seq_len": 128,  # Max sequence length
    "dropout": 0.1,
    "pooling_mode": "mean",  # Mean pooling works best
    "pad_token_id": 0,
    "normalize_embeddings": True,
}

# Dataset configuration
DATA_CONFIG = {
    "dataset_name": "combined",  # "snli", "stsb", "quora", or "combined"
    "num_train_samples": 100000,  # 100K training pairs (increase for better results)
    "val_size": 0.1,  # 10% validation
    "cache_dir": "./data/cache",
    "tokenizer_vocab_size": 30000,  # Larger vocabulary
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 64,  # Large batch for contrastive learning (use as much as fits in memory)
    "num_epochs": 10,  # 10 epochs should be enough
    "learning_rate": 2e-5,  # Standard for fine-tuning
    "weight_decay": 0.01,
    "warmup_steps": 1000,  # Warmup for stable training
    "gradient_clip_norm": 1.0,
    "eval_every": 1,  # Evaluate every epoch
    "save_every": 1,  # Save checkpoints
}

# Loss configuration
LOSS_CONFIG = {
    "loss_type": "multiple_negatives_ranking",  # Most effective
    "temperature": 0.05,  # Temperature for contrastive loss
    "margin": 1.0,  # For triplet loss (if used)
}

# Scheduler configuration
SCHEDULER_CONFIG = {
    "scheduler_type": "cosine",  # "cosine", "linear", or "plateau"
    "T_max": None,  # Will be set to num_epochs
    "eta_min": 1e-7,
    "patience": 3,  # For ReduceLROnPlateau
    "factor": 0.5,  # For ReduceLROnPlateau
}

# Paths
PATHS = {
    "model_save_dir": "./models",
    "checkpoint_dir": "./checkpoints",
    "logs_dir": "./logs",
}


def get_device():
    """Get the best available device"""
    import torch

    if DEVICE_CONFIG["use_mps"] and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Metal) device: {device}")
    elif DEVICE_CONFIG["use_cuda"] and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
    elif DEVICE_CONFIG["fallback_cpu"]:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
    else:
        raise RuntimeError("No suitable device available")

    return device


def print_config():
    """Print all configuration"""
    print("=" * 70)
    print("PRODUCTION CONFIGURATION")
    print("=" * 70)
    print("\n[MODEL]")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k:25s}: {v}")

    print("\n[DATA]")
    for k, v in DATA_CONFIG.items():
        print(f"  {k:25s}: {v}")

    print("\n[TRAINING]")
    for k, v in TRAINING_CONFIG.items():
        print(f"  {k:25s}: {v}")

    print("\n[LOSS]")
    for k, v in LOSS_CONFIG.items():
        print(f"  {k:25s}: {v}")

    print("=" * 70)


# Estimate training time
def estimate_training_time(num_samples, batch_size, num_epochs):
    """Rough estimate of training time"""
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * num_epochs

    # Rough estimates (will vary based on hardware)
    # M4 MAX with MPS: ~20-30 steps/sec
    # CPU: ~3-5 steps/sec
    steps_per_sec_mps = 25
    steps_per_sec_cpu = 4

    time_mps = total_steps / steps_per_sec_mps / 60  # minutes
    time_cpu = total_steps / steps_per_sec_cpu / 60  # minutes

    print(f"\nEstimated training time:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  On M4 MAX (MPS): ~{time_mps:.1f} minutes")
    print(f"  On CPU: ~{time_cpu:.1f} minutes")

    return time_mps
