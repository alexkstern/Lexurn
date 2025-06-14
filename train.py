import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from model import UrnTransformerDecoder
from dataset import create_dataloaders
from utils import load_model_config, count_parameters

class LexurnTrainer:
    def __init__(self, config_path="configs/dummy.config", lex=False, device=None):
        """
        Initialize trainer for Lexurn experiments.
        
        Args:
            config_path: Path to configuration file
            lex: Whether to use lexical invariance (True) or normal mode (False)
            device: Device to use (auto-detected if None)
        """
        self.config = load_model_config(config_path)
        self.lex = lex
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        print(f"Lexical invariance: {self.lex}")
        
        # Initialize model
        model_config = self.config['model'].copy()
        model_config['lex'] = self.lex
        self.model = UrnTransformerDecoder(**model_config).to(self.device)
        
        # Print model info
        total_params, trainable_params, frozen_params = count_parameters(self.model)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable, {frozen_params:,} frozen")
        
        # Initialize optimizer
        training_config = self.config['training']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Loss function for causal language modeling
        self.criterion = nn.CrossEntropyLoss()
        
        # Training tracking
        self.train_losses = []
        self.eval_losses = []
        self.step = 0
        
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
        input_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        targets = sequences[:, 1:].contiguous()        # (batch_size, seq_len-1)
        
        # Reshape for loss calculation
        input_logits = input_logits.view(-1, self.model.vocab_size)  # (batch_size * (seq_len-1), vocab_size)
        targets = targets.view(-1)                                   # (batch_size * (seq_len-1))
        
        # Calculate loss
        loss = self.criterion(input_logits, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        max_grad_norm = self.config['training']['max_grad_norm']
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        self.optimizer.step()
        
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
            num_epochs = self.config['training']['num_epochs']
        
        eval_frequency = self.config['evaluation']['eval_frequency']
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Evaluation frequency: every {eval_frequency} steps")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Training loop
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Evaluation
                if self.step % eval_frequency == 0:
                    eval_loss = self.evaluate(test_loader)
                    self.eval_losses.append(eval_loss)
                    
                    print(f"\nStep {self.step}: Train Loss = {loss:.4f}, Eval Loss = {eval_loss:.4f}")
            
            # Record epoch loss
            avg_epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_epoch_loss)
            
            print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
    
    def save_model(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'lex': self.lex,
            'step': self.step,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.train_losses = checkpoint['train_losses']
        self.eval_losses = checkpoint['eval_losses']
        
        print(f"Model loaded from {path}")
    
    def plot_losses(self):
        """Plot training and evaluation losses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training loss per epoch
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Evaluation loss
        if self.eval_losses:
            eval_steps = np.arange(len(self.eval_losses)) * self.config['evaluation']['eval_frequency']
            ax2.plot(eval_steps, self.eval_losses)
            ax2.set_title('Evaluation Loss')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def run_experiment(config_path="configs/dummy.config", lex=False, train_steps=None, test_steps=256, num_epochs=10):
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
    print("="*60)
    print(f"LEXURN EXPERIMENT - Lexical Invariance: {lex}")
    print("="*60)
    
    # Create data loaders
    train_loader, test_loader, test_dataset, config = create_dataloaders(
        config_path=config_path,
        train_steps=train_steps,
        test_steps=test_steps
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
    print("\n" + "="*60)
    print("TRAINING NORMAL MODEL")
    print("="*60)
    normal_trainer = run_experiment(
        lex=False,
        train_steps=train_steps,
        test_steps=test_steps,
        num_epochs=num_epochs
    )
    
    # Lexical invariance model
    print("\n" + "="*60)
    print("TRAINING LEXICAL INVARIANCE MODEL")
    print("="*60)
    lex_trainer = run_experiment(
        lex=True,
        train_steps=train_steps,
        test_steps=test_steps,
        num_epochs=num_epochs
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Normal model final loss: {normal_trainer.train_losses[-1]:.4f}")
    print(f"Lexical model final loss: {lex_trainer.train_losses[-1]:.4f}")