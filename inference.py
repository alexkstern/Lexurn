import torch
import argparse
from pathlib import Path
from utils import load_config

def get_device():
    """Get the appropriate device (auto-detect like in utils.py)"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(checkpoint_path,device="cpu"):
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)
    return checkpoint

def print_config_checkpoint(checkpoint):
    """Load a .pt checkpoint file and print its configuration"""    
    for key in checkpoint.keys():
        print(key)
        if key != "model_state_dict" and key != "optimizer_state_dict" and key != "train_losses" and key != "eval_losses":
            print(checkpoint[key])
    
    print("\n" + "=" * 50)
    
    # Print config if it exists
    if 'config' in checkpoint:
        print("Config found:")
        config = checkpoint['config']
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("No 'config' key found in checkpoint")
    
    print("\n" + "=" * 50)
    
    return checkpoint


if __name__ == "__main__":
    checkpoint_path="checkpoints/seq_80_4M/lexical_single_task_lexical_20250617_012740.pt"
    device= get_device()

    checkpoint= load_checkpoint(checkpoint_path,device)
    print_config_checkpoint(checkpoint)

    checkpoint_path="checkpoints/seq_10_4M_2/lexical_single_task_lexical_20250617_113729.pt"
    device= get_device()

    checkpoint= load_checkpoint(checkpoint_path,device)
    print_config_checkpoint(checkpoint)