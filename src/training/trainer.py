"""Training loop for embedding models"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
from tqdm import tqdm
import time


class EmbeddingTrainer:
    """
    Trainer class for embedding models.
    Supports different loss functions and training strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Args:
            model: Embedding model
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Optional learning rate scheduler
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass - handle different batch formats
            if "input_ids_1" in batch and "input_ids_2" in batch:
                # Pair-based training
                output1 = self.model(batch["input_ids_1"], batch["attention_mask_1"])
                output2 = self.model(batch["input_ids_2"], batch["attention_mask_2"])

                embeddings1 = output1["embeddings"] if isinstance(output1, dict) else output1
                embeddings2 = output2["embeddings"] if isinstance(output2, dict) else output2

                # Compute loss (assumes contrastive loss or similar)
                loss = self.loss_fn(embeddings1, embeddings2)

                # Add auxiliary loss if present (for MoE models)
                if isinstance(output1, dict) and "aux_loss" in output1 and output1["aux_loss"] is not None:
                    loss = loss + output1["aux_loss"]
                if isinstance(output2, dict) and "aux_loss" in output2 and output2["aux_loss"] is not None:
                    loss = loss + output2["aux_loss"]

            elif "input_ids_anchor" in batch:
                # Triplet-based training
                output_anchor = self.model(batch["input_ids_anchor"], batch["attention_mask_anchor"])
                output_positive = self.model(batch["input_ids_positive"], batch["attention_mask_positive"])
                output_negative = self.model(batch["input_ids_negative"], batch["attention_mask_negative"])

                emb_anchor = output_anchor["embeddings"] if isinstance(output_anchor, dict) else output_anchor
                emb_positive = output_positive["embeddings"] if isinstance(output_positive, dict) else output_positive
                emb_negative = output_negative["embeddings"] if isinstance(output_negative, dict) else output_negative

                loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)

                # Add auxiliary loss if present (for MoE models)
                for output in [output_anchor, output_positive, output_negative]:
                    if isinstance(output, dict) and "aux_loss" in output and output["aux_loss"] is not None:
                        loss = loss + output["aux_loss"]

            else:
                raise ValueError("Unknown batch format")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            if "input_ids_1" in batch and "input_ids_2" in batch:
                output1 = self.model(batch["input_ids_1"], batch["attention_mask_1"])
                output2 = self.model(batch["input_ids_2"], batch["attention_mask_2"])

                embeddings1 = output1["embeddings"] if isinstance(output1, dict) else output1
                embeddings2 = output2["embeddings"] if isinstance(output2, dict) else output2

                loss = self.loss_fn(embeddings1, embeddings2)

                # Add auxiliary loss if present (for MoE models)
                if isinstance(output1, dict) and "aux_loss" in output1 and output1["aux_loss"] is not None:
                    loss = loss + output1["aux_loss"]
                if isinstance(output2, dict) and "aux_loss" in output2 and output2["aux_loss"] is not None:
                    loss = loss + output2["aux_loss"]

            elif "input_ids_anchor" in batch:
                output_anchor = self.model(batch["input_ids_anchor"], batch["attention_mask_anchor"])
                output_positive = self.model(batch["input_ids_positive"], batch["attention_mask_positive"])
                output_negative = self.model(batch["input_ids_negative"], batch["attention_mask_negative"])

                emb_anchor = output_anchor["embeddings"] if isinstance(output_anchor, dict) else output_anchor
                emb_positive = output_positive["embeddings"] if isinstance(output_positive, dict) else output_positive
                emb_negative = output_negative["embeddings"] if isinstance(output_negative, dict) else output_negative

                loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)

                # Add auxiliary loss if present (for MoE models)
                for output in [output_anchor, output_positive, output_negative]:
                    if isinstance(output, dict) and "aux_loss" in output and output["aux_loss"] is not None:
                        loss = loss + output["aux_loss"]

            else:
                raise ValueError("Unknown batch format")

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        eval_every: int = 1,
        save_best: bool = True,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
            save_best: Whether to save the best model
            save_path: Path to save the best model

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')

        print(f"Training for {num_epochs} epochs on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["learning_rate"].append(current_lr)

            # Validate
            val_loss = None
            if val_loader is not None and epoch % eval_every == 0:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)

                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"Saved best model to {save_path}")

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print epoch summary
            epoch_time = time.time() - start_time
            log_msg = f"Epoch {epoch}/{num_epochs} - {epoch_time:.1f}s - train_loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - val_loss: {val_loss:.4f}"
            log_msg += f" - lr: {current_lr:.6f}"
            print(log_msg)

        print("Training completed!")
        return self.history

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        print(f"Checkpoint loaded from {path}")
        return checkpoint.get("epoch", 0)
