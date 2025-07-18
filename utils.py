
import os, importlib, getpass
from typing import Optional
import configparser
import torch
from pathlib import Path
from datetime import datetime

# ---------- helpers ----------
def in_colab() -> bool:
    """True if running in Colab (not just in a notebook, but also via !python)."""
    try:
        importlib.import_module("google.colab")
        return True
    except ModuleNotFoundError:
        return False

def resolve_wandb_key(
    api_key: Optional[str] = None,
    prompt_if_missing: bool = True,
) -> Optional[str]:
    """
    Return the first non-empty WANDB key we can find.
    Priority: CLI arg → $ENV → Colab → manual prompt (optional).
    """
    # 1) explicit api_key
    if api_key:
        return api_key

    # 2) environment variable
    env_key = os.getenv("WANDB_API_KEY")
    if env_key:
        return env_key

    # 3) Colab secrets
    if in_colab():
        try:
            from google.colab import userdata
            colab_key = userdata.get("WANDB_API_KEY")
            if colab_key:
                return colab_key
        except Exception:
            pass  # userdata may not exist

    # 4) interactive prompt
    if prompt_if_missing and os.isatty(0):          # ensure we’re in an interactive TTY
        try:
            entered = getpass.getpass(
                "Enter your WANDB_API_KEY (leave blank to skip): "
            ).strip()
            if entered:
                return entered
        except (EOFError, KeyboardInterrupt):
            pass  # user hit Ctrl-D/C

    # nothing found
    return None


def generate_model_name(config_path: str, lex_mode: bool, n_tasks: int, fine_tune=None) -> str:
    """Generate model name with format: normal/lexinv_configname_n_urns_X_datetime"""
    # Extract config file name without extension
    config_name = Path(config_path).stem
    
    # Model type prefix
    model_type = "lexinv" if lex_mode else "normal"
    
    # Generate timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Format: normal/lexinv_configname_n_urns_X_datetime
    name = f"{model_type}_{config_name}_n_urns_{n_tasks}_{timestamp}"

    if fine_tune:
        name += "_finetune"

    return name


def save_checkpoint(model, config, model_type, checkpoint_path, **extra_info):
    """
    Save model checkpoint with configuration and additional information.
    
    Args:
        model: PyTorch model to save
        config: Configuration dictionary
        model_type: String indicating model type (e.g., 'normal', 'lexical')
        checkpoint_path: Path to save checkpoint
        **extra_info: Additional information to save (e.g., best_val_loss, epoch, etc.)
    """
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': model_type,
        **extra_info
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_data
