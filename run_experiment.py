import os, configparser
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
from tqdm import tqdm

from build_dataset import generate_dataset
from train import LexurnTrainer
from model import UrnTransformerDecoder
from evaluation_functions import calculate_in_context_distribution, symmetrized_kl_div
from utils import resolve_wandb_key, save_checkpoint


def _safe_device(requested: str) -> str:
    return requested if (requested == "cpu" or torch.cuda.is_available()) else "cpu"


def load_config(path: str = "experiment.config") -> Dict:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    device = _safe_device(cfg.get("System", "device"))
    return {
        # Dataset
        "n_train_samples": cfg.getint("Dataset", "n_train_samples"),
        "context_len":     cfg.getint("Dataset", "context_len"),
        "n_colors":        cfg.getint("Dataset", "n_colors"),
        "n_tasks":         cfg.getint("Dataset", "n_tasks"),
        
        # Model
        "d_model":  cfg.getint("Model", "d_model"),
        "n_layers": cfg.getint("Model", "n_layers"),
        "n_heads":  cfg.getint("Model", "n_heads"),
        
        # Training
        "n_epochs":      cfg.getint("Training", "n_epochs"),
        "batch_size":    cfg.getint("Training", "batch_size"),
        "learning_rate": cfg.getfloat("Training", "learning_rate"),
        
        # Evaluation
        "n_test_samples":     cfg.getint("Evaluation", "n_test_samples"),
        "n_urns_test":        cfg.getint("Evaluation", "n_urns_test"),
        "eval_epoch_frac":    cfg.getfloat("Evaluation", "eval_epoch_frac"),
        "use_early_stopping": cfg.getboolean("Evaluation", "use_early_stopping"),
        "patience":           cfg.getint("Evaluation", "patience"),
        "min_delta":          cfg.getfloat("Evaluation", "min_delta"),
        
        # System
        "seed":   cfg.getint("System", "seed"),
        "device": device,
    }


class UrnDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, task_ids):
        self.sequences = sequences
        self.task_ids = task_ids.squeeze()
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]


def low_entropy_dataset(n_colors: int, context_len: int):
    """Create low entropy dataset: one sequence per color (all same token)."""
    sequences = []
    for color in range(n_colors):
        seq = torch.full((context_len,), color, dtype=torch.long)
        sequences.append(seq)
    return torch.stack(sequences)


@torch.no_grad()
def evaluate_model_kl(trainer: LexurnTrainer, dataloader: DataLoader, vocab_size: int):
    """Evaluate model by comparing against Bayesian ICL solution."""
    trainer.model.eval()
    
    total_sym_kl = 0.0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating Model KL", leave=False):
        sequences = batch.to(trainer.device)
        batch_size = sequences.size(0)
        
        # Get model predictions for last position  
        inputs = sequences[:, :-1]
        logits = trainer.model(inputs)
        last_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        
        # Convert to probabilities
        model_probs = torch.softmax(last_logits, dim=-1)
        
        # Calculate Bayesian ICL solution for each sequence
        for i in range(batch_size):
            sequence = sequences[i]
            
            # Get Bayesian in-context distribution
            icl_dist = calculate_in_context_distribution(sequence, vocab_size=vocab_size)
            icl_dist = icl_dist.to(trainer.device)
            
            # Calculate symmetrized KL divergence
            sym_kl = symmetrized_kl_div(
                model_probs[i].unsqueeze(0), 
                icl_dist.unsqueeze(0)
            ).item()
            
            total_sym_kl += sym_kl
            num_samples += 1
    
    trainer.model.train()
    return total_sym_kl / num_samples


@torch.no_grad()
def evaluate_val_loss(trainer: LexurnTrainer, dataloader: DataLoader):
    """Evaluate validation loss."""
    trainer.model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating Val Loss", leave=False):
        sequences = batch.to(trainer.device)
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        
        logits = trainer.model(inputs)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        total_loss += loss.item()
        num_batches += 1
    
    trainer.model.train()
    return total_loss / num_batches


def run_lexurn_experiment(*,
    config_path: str = "experiment.config",
    use_wandb: bool = True,
    wandb_project: str = "lexurn",
    wandb_api_key: str | None = None,
    train_normal: bool = True,
    train_lexical: bool = True):

    cfg = load_config(config_path)
    device = cfg["device"]

    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["seed"])

    # Weights & Biases setup
    if use_wandb and wandb_api_key:
        wandb.login(key=wandb_api_key)

    lex_modes = []
    if train_normal:
        lex_modes.append(False)
    if train_lexical:
        lex_modes.append(True)
    if not lex_modes:
        raise ValueError("At least one of train_normal/train_lexical must be True.")

    # Generate datasets
    train_sequences, train_urns, train_task_ids = generate_dataset(
        context_len=cfg["context_len"], n_tasks=cfg["n_tasks"], n_colors=cfg["n_colors"],
        n_steps=cfg["n_train_samples"], alpha=1.0, seed=cfg["seed"]
    )

    id_sequences, _, id_task_ids = generate_dataset(
        context_len=cfg["context_len"], n_tasks=cfg["n_tasks"], n_colors=cfg["n_colors"],
        n_steps=cfg["n_test_samples"], alpha=1.0, seed=cfg["seed"] + 1000
    )

    ood_sequences, _, ood_task_ids = generate_dataset(
        context_len=cfg["context_len"], n_tasks=cfg["n_urns_test"], n_colors=cfg["n_colors"],
        n_steps=cfg["n_test_samples"], alpha=1.0, seed=cfg["seed"] + 2000
    )

    low_sequences = low_entropy_dataset(cfg["n_colors"], cfg["context_len"])
    low_task_ids = torch.zeros(low_sequences.size(0))

    # Create dataloaders
    train_loader = DataLoader(UrnDataset(train_sequences, train_task_ids), 
                             batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    id_loader = DataLoader(UrnDataset(id_sequences, id_task_ids), 
                          batch_size=cfg["batch_size"], shuffle=False)
    ood_loader = DataLoader(UrnDataset(ood_sequences, ood_task_ids), 
                           batch_size=cfg["batch_size"], shuffle=False)
    low_loader = DataLoader(UrnDataset(low_sequences, low_task_ids), 
                           batch_size=cfg["batch_size"], shuffle=False)

    # Train models
    for lex_mode in lex_modes:
        name = "Lexical" if lex_mode else "Normal"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        if use_wandb:
            run = wandb.init(
                project=wandb_project,
                name=f"{name.lower()}_{timestamp}",
                config={**cfg, "model_type": name.lower(), "lexical_invariance": lex_mode}
            )

        model = UrnTransformerDecoder(
            vocab_size=cfg["n_colors"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], context_len=cfg["context_len"], lex=lex_mode
        ).to(device)

        trainer = LexurnTrainer(model, device=device, learning_rate=cfg["learning_rate"])

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        Path("checkpoints").mkdir(exist_ok=True)
        checkpoint_path = f"checkpoints/{name.lower()}_{timestamp}.pt"
        
        # Calculate eval steps based on epoch fraction
        eval_steps = int(int(cfg["n_train_samples"]) / int(cfg["batch_size"]) * cfg["eval_epoch_frac"])
        print(f"Eval steps: {eval_steps}")
        
        step = 0
        try:
            for epoch in tqdm(range(cfg["n_epochs"]), desc=f"Training {name}"):
                epoch_train_loss = 0.0
                num_batches = 0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                    train_loss = trainer.train_step(batch)
                    epoch_train_loss += train_loss
                    num_batches += 1
                    step += 1
                    
                    if use_wandb:
                        wandb.log({"step": step, "train_loss": train_loss})
                    
                    # Step-based evaluation
                    if step % eval_steps == 0:
                        val_loss = evaluate_val_loss(trainer, id_loader)
                        id_sym_kl = evaluate_model_kl(trainer, id_loader, cfg["n_colors"])
                        ood_sym_kl = evaluate_model_kl(trainer, ood_loader, cfg["n_colors"])
                        low_sym_kl = evaluate_model_kl(trainer, low_loader, cfg["n_colors"])
                        
                        if use_wandb:
                            wandb.log({
                                "step": step, "epoch": epoch, "val_loss": val_loss,
                                "id_symmetrized_kl_divergence": id_sym_kl,
                                "ood_symmetrized_kl_divergence": ood_sym_kl,
                                "lowent_symmetrized_kl_divergence": low_sym_kl
                            })

                        # Early stopping based on validation loss
                        if val_loss < best_val_loss - cfg["min_delta"]:
                            best_val_loss = val_loss
                            patience_counter = 0
                            
                            # Save best model immediately when validation improves
                            save_checkpoint(
                                model=model,
                                config=cfg,
                                model_type=name.lower(),
                                checkpoint_path=checkpoint_path,
                                best_val_loss=best_val_loss,
                                epoch=epoch,
                                step=step,
                                train_loss=train_loss,
                                id_sym_kl=id_sym_kl,
                                ood_sym_kl=ood_sym_kl,
                                low_sym_kl=low_sym_kl,
                                timestamp=timestamp
                            )
                        else:
                            patience_counter += 1
                            if cfg["use_early_stopping"] and patience_counter >= cfg["patience"]:
                                raise StopIteration  # Break out of both loops

        except KeyboardInterrupt:
            if use_wandb:
                wandb.summary["interrupted"] = True
        
        except StopIteration:
            # Early stopping triggered
            pass

        finally:
            if use_wandb:
                wandb.finish()
            del model, trainer
            torch.cuda.empty_cache()


if __name__ == "__main__":

    api=None
    run_lexurn_experiment(
        config_path="experiment.config",
        use_wandb=True,
        wandb_project="lexurn",
        wandb_api_key=resolve_wandb_key(api_key=api),
        train_normal=True,
        train_lexical=True
    )