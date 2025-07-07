# ────────────────────────────────────────────────────────────────────────────────
#  Minimal table logger – uploads at most k rows per split
# ────────────────────────────────────────────────────────────────────────────────
import torch, wandb


def setup_wandb_config(config_dict: dict, lex_mode: bool, train_urns: torch.Tensor = None) -> None:
    """Initialize W&B config with experiment parameters."""
    cfg = wandb.config
    
    # Core experiment config
    cfg.update({
        # Dataset parameters
        "n_train_samples": config_dict["n_train_samples"],
        "context_len": config_dict["context_len"],
        "n_colors": config_dict["n_colors"],
        "n_tasks": config_dict["n_tasks"],
        
        # Model parameters
        "d_model": config_dict["d_model"],
        "n_layers": config_dict["n_layers"],
        "n_heads": config_dict["n_heads"],
        
        # Training parameters
        "n_epochs": config_dict["n_epochs"],
        "batch_size": config_dict["batch_size"],
        "learning_rate": config_dict["learning_rate"],
        
        # Evaluation parameters
        "n_test_samples": config_dict["n_test_samples"],
        "n_urns_test": config_dict["n_urns_test"],
        "eval_epoch_frac": config_dict["eval_epoch_frac"],
        "use_early_stopping": config_dict["use_early_stopping"],
        "patience": config_dict["patience"],
        "min_delta": config_dict["min_delta"],
        
        # System parameters
        "seed": config_dict["seed"],
        "device": config_dict["device"],
        
        # Lexical invariance flag
        "lexical": lex_mode,
    })
    
    # Add training urns if provided
    if train_urns is not None:
        cfg.update({
            "train_urns": train_urns.tolist(),
        })

def collect_predictions(collectors, split_name, model_probs, icl_probs, sym_kl_divs):
    """Collect predictions during batched evaluation for KL divergence analysis."""
    
    if split_name not in collectors:
        collectors[split_name] = []
    
    # Add all predictions from this batch
    for i in range(model_probs.size(0)):
        sample_data = {
            "model_prediction": model_probs[i].cpu().tolist(),
            "bayesian_icl_solution": icl_probs[i].cpu().tolist(),
            "symmetrized_kl_divergence": sym_kl_divs[i].item()
        }
        
        collectors[split_name].append(sample_data)

def _to_str(x, precision=4):
    """
    Convert a list of floats to a compact string like
    '[0.1234, 0.5032, 0.2500, 0.1234]'.
    """
    # x is already a Python list
    fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join(fmt.format(v) for v in x) + "]"

def upload_prediction_tables(collectors, step: int):
    """Upload all collected predictions as W&B tables."""
    
    for split_name, samples in collectors.items():
        columns = ["model_prediction", "bayesian_icl_solution", "symmetrized_kl_divergence"]
        table = wandb.Table(columns=columns)
        
        rows_added = 0
        for sample in samples:
            table.add_data(
                _to_str(sample["model_prediction"]),
                _to_str(sample["bayesian_icl_solution"]),
                sample["symmetrized_kl_divergence"]
            )
            rows_added += 1
        
        wandb.log({f"{split_name}_kl_analysis": table}, step=step)

