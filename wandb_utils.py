"""
Minimal Weights & Biases utilities for basic loss tracking only.
"""

import wandb


def log_loss(wandb_run, train_loss, eval_loss=None, step=None):
    """
    Log only basic training and evaluation losses.
    
    Args:
        wandb_run: Active wandb run
        train_loss: Training loss value
        eval_loss: Optional evaluation loss value
        step: Current training step
    """
    if wandb_run is None:
        return
        
    log_data = {"train_loss": train_loss}
    
    if eval_loss is not None:
        log_data["eval_loss"] = eval_loss
    
    if step is not None:
        wandb_run.log(log_data, step=step)
    else:
        wandb_run.log(log_data)