"""Checkpoint utilities for saving/restoring model training"""
import torch
import os
import pickle
from typing import Optional, Dict, Any


def save_training_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    history: dict,
    epoch: int,
    config: dict,
    filepath: str,
    model_name: str = "checkpoint"
):
    """
    Save complete training checkpoint.

    Args:
        model: The model
        tokenizer: The tokenizer
        optimizer: Optimizer state
        scheduler: Scheduler state
        history: Training history
        epoch: Current epoch
        config: Model configuration
        filepath: Directory to save
        model_name: Base name for checkpoint
    """
    os.makedirs(filepath, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "history": history,
        "config": config,
    }

    checkpoint_path = os.path.join(filepath, f"{model_name}_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save tokenizer separately
    tokenizer_path = os.path.join(filepath, f"{model_name}_tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"✓ Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_training_checkpoint(
    model,
    tokenizer_class,
    optimizer,
    scheduler,
    filepath: str,
    model_name: str = "checkpoint",
    epoch: Optional[int] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        model: Model to load state into
        tokenizer_class: Tokenizer class
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        filepath: Directory containing checkpoint
        model_name: Base name of checkpoint
        epoch: Specific epoch to load (None = latest)
        device: Device to load to

    Returns:
        Dictionary with loaded states and metadata
    """
    # Find checkpoint file
    if epoch is not None:
        checkpoint_path = os.path.join(filepath, f"{model_name}_epoch_{epoch}.pt")
    else:
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(filepath) if f.startswith(f"{model_name}_epoch_")]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for {model_name} in {filepath}")

        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split(".pt")[0]))
        checkpoint_path = os.path.join(filepath, checkpoints[-1])

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load tokenizer
    tokenizer_path = os.path.join(filepath, f"{model_name}_tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "history": checkpoint.get("history", {}),
        "epoch": checkpoint.get("epoch", 0),
        "config": checkpoint.get("config", {})
    }


def checkpoint_exists(filepath: str, model_name: str) -> bool:
    """Check if checkpoint exists"""
    if not os.path.exists(filepath):
        return False

    checkpoints = [f for f in os.listdir(filepath) if f.startswith(f"{model_name}_epoch_")]
    return len(checkpoints) > 0


def get_latest_checkpoint_epoch(filepath: str, model_name: str) -> Optional[int]:
    """Get the epoch number of the latest checkpoint"""
    if not checkpoint_exists(filepath, model_name):
        return None

    checkpoints = [f for f in os.listdir(filepath) if f.startswith(f"{model_name}_epoch_")]
    checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split(".pt")[0]))

    latest = checkpoints[-1]
    epoch = int(latest.split("_epoch_")[1].split(".pt")[0])

    return epoch


def prompt_resume_training(filepath: str, model_name: str) -> bool:
    """
    Prompt user to resume from checkpoint if it exists.

    Returns:
        True if should resume, False otherwise
    """
    if not checkpoint_exists(filepath, model_name):
        return False

    latest_epoch = get_latest_checkpoint_epoch(filepath, model_name)

    print("\n" + "="*70)
    print("CHECKPOINT FOUND!")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Latest epoch: {latest_epoch}")
    print(f"Location: {filepath}")

    while True:
        response = input("\nResume from checkpoint? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def auto_save_checkpoint_wrapper(trainer, save_dir: str, model_name: str, save_every: int = 1):
    """
    Wrapper to automatically save checkpoints during training.

    Args:
        trainer: EmbeddingTrainer instance
        save_dir: Directory to save checkpoints
        model_name: Name for checkpoints
        save_every: Save every N epochs
    """
    original_train = trainer.train

    def train_with_checkpoints(*args, **kwargs):
        # Get training parameters
        train_loader = args[0] if args else kwargs.get('train_loader')
        val_loader = args[1] if len(args) > 1 else kwargs.get('val_loader')
        num_epochs = args[2] if len(args) > 2 else kwargs.get('num_epochs', 10)

        # Run original training but save after each epoch
        for epoch in range(1, num_epochs + 1):
            # Train one epoch
            history = original_train(
                train_loader,
                val_loader,
                num_epochs=epoch,
                **{k: v for k, v in kwargs.items() if k not in ['train_loader', 'val_loader', 'num_epochs']}
            )

            # Save checkpoint
            if epoch % save_every == 0:
                save_training_checkpoint(
                    model=trainer.model,
                    tokenizer=None,  # Will be saved separately
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    history=trainer.history,
                    epoch=epoch,
                    config={},
                    filepath=save_dir,
                    model_name=model_name
                )

        return history

    return train_with_checkpoints
