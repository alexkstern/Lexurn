import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from dotenv import load_dotenv

from model import UrnTransformerDecoder
from dataloader import create_dataloaders
from utils import load_model_config, count_parameters

# Load environment variables
load_dotenv()


class LexurnTrainer:
    def __init__(self, config_path="configs/dummy.config", lex=False, device=None, training_urns=None):
        """
        Initialize trainer for Lexurn experiments.

        Args:
            config_path: Path to configuration file
            lex: Whether to use lexical invariance (True) or normal mode (False)
            device: Device to use (auto-detected if None)
        """
        self.config = load_model_config(config_path)
        self.lex = lex
        self.training_urns = training_urns
        
        # Early stopping parameters
        training_config = self.config["training"]
        self.early_stopping = training_config.get("early_stopping", False)
        self.early_stopping_patience = training_config.get("early_stopping_patience", 10)
        self.early_stopping_min_delta = training_config.get("early_stopping_min_delta", 1e-4)
        
        # Early stopping state
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")
        print(f"Lexical invariance: {self.lex}")

        # Initialize model
        model_config = self.config["model"].copy()
        model_config["lex"] = self.lex
        self.model = UrnTransformerDecoder(**model_config).to(self.device)

        # Print model info
        total_params, trainable_params, frozen_params = count_parameters(self.model)
        print(
            f"Model parameters: {total_params:,} total, {trainable_params:,} trainable, {frozen_params:,} frozen"
        )

        # Initialize optimizer
        training_config = self.config["training"]
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )

        # Loss function for causal language modeling
        self.criterion = nn.CrossEntropyLoss()

        # Training tracking
        self.train_losses = []
        self.eval_losses = []
        self.step = 0
        
        # Initialize wandb if enabled
        self.use_wandb = self.config["training"]["wandb"]
        self.wandb_run = None
        if self.use_wandb:
            # Try to set up wandb authentication with proper fallback order
            wandb_key = self._setup_wandb_auth()
            
            import wandb

            wandb.login(key=wandb_key)
            self.wandb_run = wandb.init(
                project="lexurn",
                config={
                    **self.config,
                    "lex": self.lex,
                    "model_type": "lexical" if self.lex else "normal"
                },
                name=f"{'lexical' if self.lex else 'normal'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print(f"W&B initialized: {self.wandb_run.name}")

    def _setup_wandb_auth(self):
        """
        Set up wandb authentication with proper fallback order:
        1. First try Google Colab userdata
        2. Then try .env file
        3. Finally let wandb handle interactive login
        """
        import os
        
        # Try Google Colab userdata first
        try:
            from google.colab import userdata
            token = userdata.get('WANDB_API_KEY')
            return token
        except ImportError:
            # Not in Colab environment
            pass
        except Exception as e:
            # Colab userdata failed
            print(f"Could not get wandb key from Colab userdata: {e}")
        
        # Check if WANDB_API_KEY is set from .env file
        env_token = os.getenv('WANDB_API_KEY')
        if env_token:
            print("Using wandb API key from .env file")
            return env_token
        
        # If we get here, let wandb handle interactive login
        print("No wandb API key found, wandb will prompt for authentication")
        return None

    def train_step(self, batch):
        """
        Single training step with causal language modeling loss.

        Args:
            batch: Sequences of shape (batch_size, context_len)

        Returns:
            loss: Training loss for this batch
        """
        self.model.train()

        # Move to device
        sequences = batch.to(self.device)
        batch_size, seq_len = sequences.shape

        # Forward pass
        logits = self.model(sequences)  # (batch_size, seq_len, vocab_size)

        # Causal language modeling: predict next token
        # Input: positions 0 to seq_len-2, Target: positions 1 to seq_len-1
        input_logits = logits[
            :, :-1, :
        ].contiguous()  # (batch_size, seq_len-1, vocab_size)
        targets = sequences[:, 1:].contiguous()  # (batch_size, seq_len-1)

        # Reshape for loss calculation
        input_logits = input_logits.view(
            -1, self.model.vocab_size
        )  # (batch_size * (seq_len-1), vocab_size)
        targets = targets.view(-1)  # (batch_size * (seq_len-1))

        # Calculate loss
        loss = self.criterion(input_logits, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        max_grad_norm = self.config["training"]["max_grad_norm"]
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()
        
        # Log to wandb if enabled (step-based)
        if self.use_wandb:
            self.wandb_run.log({
                "train_loss": loss.item()
            }, step=self.step)

        return loss.item()

    def evaluate(self, test_loader):
        """
        Evaluate model on test set.

        Args:
            test_loader: DataLoader for test data

        Returns:
            avg_loss: Average loss on test set
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                sequences = batch.to(self.device)

                # Forward pass
                logits = self.model(sequences)

                # Causal language modeling loss
                input_logits = logits[:, :-1, :].contiguous()
                targets = sequences[:, 1:].contiguous()

                input_logits = input_logits.view(-1, self.model.vocab_size)
                targets = targets.view(-1)

                loss = self.criterion(input_logits, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def train(self, train_loader, test_loader, num_epochs=None):
        """
        Main training loop.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            num_epochs: Number of epochs (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config["training"]["num_epochs"]

        eval_frequency = self.config["evaluation"]["eval_frequency"]

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Evaluation frequency: every {eval_frequency} steps")

        for epoch in range(num_epochs):
            epoch_losses = []

            # Training loop
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.step += 1

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Evaluation
                if self.step % eval_frequency == 0:
                    eval_loss = self.evaluate(test_loader)
                    self.eval_losses.append(eval_loss)
                    
                    # Log eval loss to wandb (step-based)
                    if self.use_wandb:
                        self.wandb_run.log({
                            "eval_loss": eval_loss
                        }, step=self.step)

                    print(
                        f"\nStep {self.step}: Train Loss = {loss:.4f}, Eval Loss = {eval_loss:.4f}"
                    )
                    
                    # Early stopping check
                    if self.early_stopping:
                        if eval_loss < self.best_eval_loss - self.early_stopping_min_delta:
                            self.best_eval_loss = eval_loss
                            self.patience_counter = 0
                            print(f"New best eval loss: {eval_loss:.6f}")
                        else:
                            self.patience_counter += 1
                            print(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                            
                            if self.patience_counter >= self.early_stopping_patience:
                                print(f"\nEarly stopping triggered after {self.patience_counter} steps without improvement!")
                                print(f"Best eval loss was: {self.best_eval_loss:.6f}")
                                self.early_stopped = True
                                return  # Exit training immediately

            # Record epoch loss
            avg_epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_epoch_loss)
            
            # Log epoch metrics to wandb (optional, for multi-epoch training)
            if self.use_wandb and num_epochs > 1:
                self.wandb_run.log({
                    "epoch_loss": avg_epoch_loss,
                    "epoch": epoch + 1
                }, step=self.step)

            print(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")

    def save_model(self, path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "lex": self.lex,
            "step": self.step,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "training_urns": self.training_urns,  # Save the urns used for training
            # Key model parameters for easy access without config
            "context_len": self.config["model"]["context_len"],
            "vocab_size": self.config["model"]["vocab_size"],
            "d_model": self.config["model"]["d_model"],
            "n_layers": self.config["model"]["n_layers"],
            "n_heads": self.config["model"]["n_heads"],
            # Key training parameters
            "n_steps": self.config["dataset"]["n_steps"],
            "batch_size": self.config["training"]["batch_size"],
            "num_epochs": self.config["training"]["num_epochs"],
            "learning_rate": self.config["training"]["learning_rate"],
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
        if self.training_urns is not None:
            print(f"Saved {len(self.training_urns)} training urns in checkpoint")

    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.train_losses = checkpoint["train_losses"]
        self.eval_losses = checkpoint["eval_losses"]

        print(f"Model loaded from {path}")

    def plot_losses(self):
        """Plot training and evaluation losses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Training loss per epoch
        ax1.plot(self.train_losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        # Evaluation loss
        if self.eval_losses:
            eval_steps = (
                np.arange(len(self.eval_losses))
                * self.config["evaluation"]["eval_frequency"]
            )
            ax2.plot(eval_steps, self.eval_losses)
            ax2.set_title("Evaluation Loss")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Loss")
            ax2.grid(True)

        plt.tight_layout()
        plt.show()
    
    def finish_wandb(self):
        """Finish wandb run if active."""
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
            print("W&B run finished")


def run_experiment(
    config_path="configs/dummy.config",
    lex=False,
    train_steps=None,
    test_steps=256,
    num_epochs=10,
):
    """
    Run a complete training experiment.

    Args:
        config_path: Path to config file
        lex: Whether to use lexical invariance
        train_steps: Number of training steps (None = use config)
        test_steps: Number of test samples
        num_epochs: Number of training epochs

    Returns:
        trainer: Trained model
    """
    print("=" * 60)
    print(f"LEXURN EXPERIMENT - Lexical Invariance: {lex}")
    print("=" * 60)

    # Create data loaders
    train_loader, test_loader, test_dataset, config = create_dataloaders(
        config_path=config_path, train_steps=train_steps, test_steps=test_steps
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize trainer
    trainer = LexurnTrainer(config_path=config_path, lex=lex)

    # Train model
    trainer.train(train_loader, test_loader, num_epochs=num_epochs)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lexurn_{'lex' if lex else 'normal'}_{timestamp}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    trainer.save_model(f"checkpoints/{model_name}")

    return trainer


if __name__ == "__main__":
    # Run both experiments for comparison
    print("Running Lexurn training experiments...")

    # Quick test with small datasets
    train_steps = 1000
    test_steps = 256
    num_epochs = 5

    # Normal model
    print("\n" + "=" * 60)
    print("TRAINING NORMAL MODEL")
    print("=" * 60)
    normal_trainer = run_experiment(
        lex=False, train_steps=train_steps, test_steps=test_steps, num_epochs=num_epochs
    )

    # Lexical invariance model
    print("\n" + "=" * 60)
    print("TRAINING LEXICAL INVARIANCE MODEL")
    print("=" * 60)
    lex_trainer = run_experiment(
        lex=True, train_steps=train_steps, test_steps=test_steps, num_epochs=num_epochs
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Normal model final loss: {normal_trainer.train_losses[-1]:.4f}")
    print(f"Lexical model final loss: {lex_trainer.train_losses[-1]:.4f}")
