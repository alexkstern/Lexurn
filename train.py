import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
from typing import Dict, Optional
from tqdm.auto import tqdm
from evaluation_functions import symmetrized_kl_div


class LexurnTrainer:
    """Minimal trainer for causal language modeling with KL divergence logging."""
    
    def __init__(self, model: nn.Module, device: str = "cpu", learning_rate: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step with causal language modeling loss."""
        sequences = batch.to(self.device)
        
        # Standard causal LM: predict next token
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        
        # Flatten for cross-entropy
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate_symmetrized_kl_divergence(self, dataloader: DataLoader, urns: torch.Tensor) -> float:
        """Evaluate symmetrized KL divergence between model predictions and ground truth distributions."""
        self.model.eval()
        
        total_sym_kl = 0.0
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Evaluating KL", position=0, leave=True, ncols=100, ascii=True):
            if isinstance(batch, tuple):
                sequences = batch[0].to(self.device)
            else:
                sequences = batch.to(self.device)
                
            batch_size = sequences.size(0)
            
            # Get model predictions for last position
            inputs = sequences[:, :-1]
            logits = self.model(inputs)
            last_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Convert to probabilities
            probs = F.softmax(last_logits, dim=-1)
            
            # Get corresponding ground truth distributions
            for i in range(batch_size):
                if hasattr(dataloader.dataset, 'task_ids'):
                    task_id = dataloader.dataset.task_ids[num_samples + i]
                    true_dist = urns[task_id]
                    
                    # Calculate symmetrized KL divergence
                    sym_kl_div = symmetrized_kl_div(probs[i].unsqueeze(0), true_dist.unsqueeze(0)).item()
                    total_sym_kl += sym_kl_div
            
            num_samples += batch_size
        
        self.model.train()
        return total_sym_kl / num_samples
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training", position=0, leave=True, ncols=100, ascii=True):
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    urns: torch.Tensor,
    num_epochs: int = 10,
    device: str = "cpu",
    learning_rate: float = 1e-4,
    eval_interval: int = 5
) -> None:
    """Complete training loop with symmetrized KL divergence logging."""
    
    trainer = LexurnTrainer(model, device=device, learning_rate=learning_rate)
    
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0, leave=True, ncols=100, ascii=True):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Log training loss
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss
        })
        
        # Periodic KL divergence evaluation
        if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
            
            # Evaluate symmetrized KL divergence on ID (validation) data
            id_sym_kl = trainer.evaluate_symmetrized_kl_divergence(val_loader, urns)
            
            # Evaluate symmetrized KL divergence on OOD (test) data  
            ood_sym_kl = trainer.evaluate_symmetrized_kl_divergence(test_loader, urns)
            
            wandb.log({
                "epoch": epoch,
                "id_symmetrized_kl_divergence": id_sym_kl,
                "ood_symmetrized_kl_divergence": ood_sym_kl
            })
            
            print(f"  ID Sym KL: {id_sym_kl:.4f}, OOD Sym KL: {ood_sym_kl:.4f}")