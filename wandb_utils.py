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

def collect_predictions(collectors, split_name, sequences, predictions, targets, vocab, urn_indices=None, urns=None):
    """Collect predictions during batched evaluation."""
    decode = lambda t: " ".join(vocab.decode_sequence(t))
    
    if split_name not in collectors:
        collectors[split_name] = []
    
    
    # Add all predictions from this batch
    for i in range(sequences.size(0)):
        correct = torch.equal(predictions[i], targets[i])
        sample_data = {
            "input": decode(sequences[i].cpu()),
            "predicted": decode(predictions[i].cpu()),
            "ground_truth": decode(targets[i].cpu()),
            "correct": correct
        }
        
        # Add urn information if provided
        if urn_indices is not None and urns is not None:
            urn_idx = urn_indices[i].item()
            sample_data["urn_idx"] = urn_idx
            sample_data["urn_vector"] = str(urns[urn_idx].cpu().tolist())
        
        collectors[split_name].append(sample_data)
        

def upload_prediction_tables(collectors, step: int):
    """Upload all collected predictions as W&B tables."""
    
    for split_name, samples in collectors.items():

        has_urn = split_name in ("ID", "OOD")
        
        if has_urn:
            columns = ["input_sequence", "predicted_ranking", "ground_truth", "correct", "urn_idx", "urn_vector"]
        else:
            columns = ["input_sequence", "predicted_ranking", "ground_truth", "correct"]
            
        table = wandb.Table(columns=columns)#, log_mode="INCREMENTAL")
        
        rows_added = 0
        for sample in samples:
            if has_urn:
                table.add_data(
                    sample["input"],
                    sample["predicted"],
                    sample["ground_truth"],
                    sample["correct"],
                    sample["urn_idx"],
                    sample["urn_vector"]
                )
            else:
                table.add_data(
                    sample["input"],
                    sample["predicted"],
                    sample["ground_truth"],
                    sample["correct"]
                )
            rows_added += 1
        
        #import pandas as pd

        #print(table.get_dataframe())
        wandb.log({f"{split_name}_predictions": table}, step=step)

