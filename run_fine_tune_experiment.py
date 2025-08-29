import os, configparser
from pathlib import Path
from typing import Dict


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
from tqdm.auto import tqdm

from build_dataset import generate_dataset
from train import LexurnTrainer
from model import UrnTransformerDecoder
from evaluation_functions import calculate_in_context_distribution, symmetrized_kl_div
from utils import resolve_wandb_key, save_checkpoint, generate_model_name
from wandb_utils import upload_prediction_tables, collect_predictions


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
def evaluate_model_kl_fast(
    trainer: LexurnTrainer,
    dataloader: DataLoader,
    vocab_size: int,
    collectors: dict | None = None,   # NEW
    split_name: str | None = None,    # NEW
    k_rows: int = 32                  # max rows to log per call
) -> float:
    """
    Vectorised KL evaluator.  If `collectors` is provided it will store
    up to `k_rows` example rows for the given `split_name`.
    """
    model, device = trainer.model, trainer.device
    model.eval()

    total_sym_kl = 0.0
    total_samples = 0

    rows_logged = 0  # to cap table size

    for sequences in tqdm(dataloader, desc=f"{split_name or 'eval'} KL", ncols=100, ascii=True):
        sequences = sequences.to(device)            # (B, L)
        B, Lm1 = sequences.size(0), sequences.size(1) - 1

        # 1) Model probs
        logits = model(sequences[:, :-1])           # (B, L-1, V)
        model_probs = torch.softmax(logits[:, -1], dim=-1)  # (B, V)

        # 2) ICL posterior
        ctx = sequences[:, :-1]
        counts = F.one_hot(ctx, vocab_size).sum(1).float()
        icl_probs = (counts + 1) / (ctx.size(1) + vocab_size)  # (B, V)

        # 3) Sym KL
        sym_kl = symmetrized_kl_div(model_probs, icl_probs)   # (B,)

        total_sym_kl += sym_kl.sum().item()
        total_samples += B

        # 4) Collect rows for W&B table
        if collectors is not None:
            if k_rows is None or rows_logged < k_rows:
                take = sequences.size(0) if k_rows is None else min(k_rows - rows_logged,
                                                                     sequences.size(0))
                collect_predictions(
                    collectors, split_name,
                    model_probs[:take], icl_probs[:take], sym_kl[:take]
                )
                rows_logged += take

    model.train()
    return total_sym_kl / total_samples


@torch.no_grad()
def evaluate_model_kl(trainer: LexurnTrainer, dataloader: DataLoader, vocab_size: int):
    """Evaluate model by comparing against Bayesian ICL solution."""
    trainer.model.eval()
    
    total_sym_kl = 0.0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating Model KL", position=0, leave=True, ncols=100, ascii=True):
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
    
    for batch in tqdm(dataloader, desc="Evaluating Val Loss", position=0, leave=True, ncols=100, ascii=True):
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


def transfer_model_weights(source_model, target_model, source_lex: bool, target_lex: bool):
    """Transfer compatible weights from source to target model."""
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    
    # Always transfer these components
    transferable_keys = [
        # Positional embeddings
        'pos_embedding.weight',
        # All transformer layers
        'layers.',
        # Final layer norm
        'ln_final.weight',
        'ln_final.bias'
    ]
    
    transferred = 0
    for key in source_state.keys():
        if any(transfer_key in key for transfer_key in transferable_keys):
            if key in target_state and source_state[key].shape == target_state[key].shape:
                target_state[key] = source_state[key]
                transferred += 1
    
    # Handle token embeddings and output projection based on mode
    if not source_lex and not target_lex:
        # Normal -> Normal: transfer token embeddings and output projection
        if 'token_embedding.weight' in source_state and 'token_embedding.weight' in target_state:
            target_state['token_embedding.weight'] = source_state['token_embedding.weight']
            transferred += 1
        if 'output_proj.weight' in source_state and 'output_proj.weight' in target_state:
            target_state['output_proj.weight'] = source_state['output_proj.weight']
            target_state['output_proj.bias'] = source_state['output_proj.bias']
            transferred += 2
    elif source_lex and not target_lex:
        # Lexical -> Normal: initialize token embeddings and output projection randomly
        print("Initializing token embeddings and output projection for normal mode...")
        # These will be initialized randomly by PyTorch
    elif not source_lex and target_lex:
        # Normal -> Lexical: don't transfer token embeddings (lexical mode doesn't use them)
        print("Normal -> Lexical: not transferring token embeddings (lexical mode uses random per-sequence embeddings)")
    elif source_lex and target_lex:
        # Lexical -> Lexical: no token embeddings to transfer
        print("Lexical -> Lexical: no token embeddings to transfer")
    
    target_model.load_state_dict(target_state)
    print(f"Transferred {transferred} parameter tensors from source model")
    return target_model


def run_fine_tune_experiment(*,
    checkpoint_path: str,
    config_path: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = "lexurn-finetune",
    wandb_api_key: str | None = None,
    target_lex: bool = False,  # Usually fine-tuning lexical->normal
    train_backbone: bool = True,
    n_epochs: int | None = None,
    learning_rate: float | None = None,
    eval_epoch_frac: float | None = None):
    """
    Fine-tune a pre-trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config_path: Optional path to new config (uses checkpoint config if None)
        target_lex: Target lexical mode (False for lexical->normal fine-tuning)
        n_epochs: Override epochs for fine-tuning
        learning_rate: Override learning rate for fine-tuning
        eval_epoch_frac: Override evaluation frequency (fraction of epoch)
    """
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint or override
    if config_path:
        cfg = load_config(config_path)
        print(f"Using config from: {config_path}")
    else:
        cfg = checkpoint['config']
        print("Using config from checkpoint")
    
    # Override fine-tuning specific parameters
    if n_epochs is not None:
        cfg["n_epochs"] = n_epochs
    if learning_rate is not None:
        cfg["learning_rate"] = learning_rate
    if eval_epoch_frac is not None:
        cfg["eval_epoch_frac"] = eval_epoch_frac
    
    device = _safe_device(cfg["device"])
    
    # Determine source model type from checkpoint
    model_type = checkpoint.get('model_type', 'unknown')
    source_lex = (model_type == 'lexical')
    
    print(f"Source model: {model_type} (lexical={source_lex})")
    print(f"Target model: {'lexical' if target_lex else 'normal'} (lexical={target_lex})")
    
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["seed"])
    
    # Create source model and load weights
    source_model = UrnTransformerDecoder(
        vocab_size=cfg["n_colors"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        context_len=cfg["context_len"],
        lex=source_lex
    )
    source_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create target model
    target_model = UrnTransformerDecoder(
        vocab_size=cfg["n_colors"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        context_len=cfg["context_len"],
        lex=target_lex
    ).to(device)
    
    # Transfer weights
    target_model = transfer_model_weights(source_model, target_model, source_lex, target_lex)

    if not train_backbone:
        for name, p in target_model.named_parameters():
            if name.startswith("token_embedding") or name.startswith("output_proj"):
                p.requires_grad = True     # keep head trainable
            else:
                p.requires_grad = False    # freeze backbone
        print("Backbone frozen; only token_embedding & output_proj will be updated.")
    
    # Clean up source model
    del source_model
    torch.cuda.empty_cache()
    
    # Generate model name for fine-tuning
    source_name = os.path.basename(checkpoint_path).replace('.pt', '')
    target_mode = "lexical" if target_lex else "normal"
    model_name = f"finetune_{source_name}_to_{target_mode}"
    
    return run_training_loop(
        model=target_model,
        cfg=cfg,
        device=device,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_api_key=wandb_api_key,
        model_name=model_name,
        source_checkpoint=checkpoint_path
    )


def run_training_loop(
    model,
    cfg,
    device,
    use_wandb: bool = True,
    wandb_project: str = "lexurn",
    wandb_api_key: str | None = None,
    model_name: str = "model",
    source_checkpoint: str | None = None):
    """Shared training loop for both original training and fine-tuning."""
    
    # Weights & Biases setup
    if use_wandb and wandb_api_key:
        wandb.login(key=wandb_api_key)
    
    if use_wandb:
        run_config = {**cfg, "model_type": "lexical" if model.lex else "normal", "lexical_invariance": model.lex}
        if source_checkpoint:
            run_config["source_checkpoint"] = source_checkpoint
        
        run = wandb.init(
            project=wandb_project,
            name=model_name,
            config=run_config
        )
    
    # Generate datasets
    train_sequences, _, train_task_ids = generate_dataset(
        context_len=cfg["context_len"], n_tasks=cfg["n_tasks"], n_colors=cfg["n_colors"],
        n_steps=cfg["n_train_samples"], alpha=1.0, 
        urn_seed=cfg["seed"], sampling_seed=cfg["seed"]
    )
    
    # ID evaluation: same urns as training, different sequences
    id_sequences, _, id_task_ids = generate_dataset(
        context_len=cfg["context_len"], n_tasks=cfg["n_tasks"], n_colors=cfg["n_colors"],
        n_steps=cfg["n_test_samples"], alpha=1.0, 
        urn_seed=cfg["seed"], sampling_seed=cfg["seed"] + 1000
    )
    
    # OOD evaluation: different urns from training
    ood_sequences, _, ood_task_ids = generate_dataset(
        context_len=cfg["context_len"], n_tasks=cfg["n_urns_test"], n_colors=cfg["n_colors"],
        n_steps=cfg["n_test_samples"], alpha=1.0, 
        urn_seed=cfg["seed"] + 2000, sampling_seed=cfg["seed"] + 2000
    )
    
    low_sequences = low_entropy_dataset(cfg["n_colors"], cfg["context_len"])
    low_task_ids = torch.zeros(low_sequences.size(0))
    
    # Create dataloaders
    train_loader = DataLoader(UrnDataset(train_sequences, train_task_ids), 
                             batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    batch_size = cfg["batch_size"] if cfg["batch_size"] > 1 else 128
    
    id_loader = DataLoader(UrnDataset(id_sequences, id_task_ids), 
                        batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(UrnDataset(ood_sequences, ood_task_ids), 
                            batch_size=batch_size, shuffle=False)
    low_loader = DataLoader(UrnDataset(low_sequences, low_task_ids), 
                            batch_size=batch_size, shuffle=False)
    
    # Create trainer
    trainer = LexurnTrainer(model, device=device, learning_rate=cfg["learning_rate"])
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    Path("checkpoints").mkdir(exist_ok=True)
    checkpoint_path = f"checkpoints/{model_name}.pt"
    
    # Calculate eval steps based on epoch fraction
    eval_steps = int(int(cfg["n_train_samples"]) / int(cfg["batch_size"]) * cfg["eval_epoch_frac"])
    print(f"Eval steps: {eval_steps}")
    
    step = 0
    try:
        for epoch in tqdm(range(cfg["n_epochs"]), desc=f"Training {model_name}", position=0, leave=True, ncols=100, ascii=True):
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", position=0, leave=True, ncols=100, ascii=True):
                train_loss = trainer.train_step(batch)
                epoch_train_loss += train_loss
                num_batches += 1
                step += 1
                
                if use_wandb:
                    wandb.log({"train_loss": train_loss}, step=step)
                
                collectors = {}  # one dict reused across evals
                
                if step % eval_steps == 0:
                    # ---------- ID ----------
                    id_sym_kl = evaluate_model_kl_fast(
                        trainer, id_loader, cfg["n_colors"],
                        collectors, "id", k_rows=None
                    )
                    # ---------- OOD ----------
                    ood_sym_kl = evaluate_model_kl_fast(
                        trainer, ood_loader, cfg["n_colors"],
                        collectors, "ood", k_rows=None
                    )
                    # ---------- LOW ----------
                    low_sym_kl = evaluate_model_kl_fast(
                        trainer, low_loader, cfg["n_colors"],
                        collectors, "low", k_rows=None
                    )
                    
                    # vanilla val-loss
                    val_loss = evaluate_val_loss(trainer, id_loader)
                    
                    # W&B scalars
                    if use_wandb:
                        wandb.log({
                            "epoch": epoch, "val_loss": val_loss,
                            "id_symmetrized_kl_divergence": id_sym_kl,
                            "ood_symmetrized_kl_divergence": ood_sym_kl,
                            "lowent_symmetrized_kl_divergence": low_sym_kl
                        }, step=step)
                    
                    # early stopping & checkpoint
                    if val_loss < best_val_loss - cfg["min_delta"]:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        save_checkpoint(
                            model=model,
                            config=cfg,
                            model_type="lexical" if model.lex else "normal",
                            checkpoint_path=checkpoint_path,
                            best_val_loss=best_val_loss,
                            epoch=epoch,
                            step=step,
                            train_loss=train_loss,
                            id_sym_kl=id_sym_kl,
                            ood_sym_kl=ood_sym_kl,
                            low_sym_kl=low_sym_kl,
                            model_name=model_name
                        )
                        
                        # upload tables
                        if use_wandb:
                            upload_prediction_tables(collectors, step)
                            collectors.clear()  # free memory
                    
                    else:
                        patience_counter += 1
                        if cfg["use_early_stopping"] and patience_counter >= cfg["patience"]:
                            raise StopIteration
    
    except KeyboardInterrupt:
        if use_wandb:
            wandb.summary["interrupted"] = True
    
    except StopIteration:
        # Early stopping triggered
        pass
    
    finally:
        if use_wandb:
            wandb.finish()
        torch.cuda.empty_cache()
    
    return checkpoint_path


if __name__ == "__main__":
    # Example usage for fine-tuning
    api=""
    # Example 1: Fine-tune a lexical model to normal mode
    checkpoint_path = "results/lexinv_experiment_n_urns_1_20250716_160312.pt"  # Replace with your actual checkpoint
    
    # Option 1: Fine-tune with same config as checkpoint
    run_fine_tune_experiment(
        checkpoint_path=checkpoint_path,
        target_lex=False,  # Fine-tune to normal mode
        use_wandb=True,
        wandb_project="lexurn-finetune",
        wandb_api_key=resolve_wandb_key(api_key=api),
        n_epochs=1,  # Reduced from 4
        learning_rate=5e-5,  # Reduced from 1e-4
        eval_epoch_frac=0.1,  # Reduced from 0.25
        train_backbone=False 
    )
    
    #learning_rate=5e-6,  # 0.000005 (half of current: 5e-5)
    #learning_rate=1e-6,  # 0.000001 (10x smaller)
    #learning_rate=5e-7,  # 0.0000005 (20x smaller)
    #learning_rate=1e-7,  # 0.0000001 (100x smaller)

    # Option 2: Fine-tune with new config
    # run_fine_tune_experiment(
    #     checkpoint_path=checkpoint_path,
    #     config_path="experiment.config",  # Use different config
    #     target_lex=False,
    #     use_wandb=True,
    #     wandb_project="lexurn-finetune",
    #     wandb_api_key=resolve_wandb_key(api_key=None)
    # )
