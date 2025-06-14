#!/usr/bin/env python3
"""
Quick test of the new task generalization functionality.
"""

import torch
from model import UrnTransformerDecoder
from generate_urns import generate_urns
from utils import load_model_config
import torch.nn.functional as F

def test_generalization_quick():
    """Quick test of the new task generalization function."""
    
    # Load config
    config = load_model_config("configs/single_task.config")
    
    # Create mock trained models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    normal_model = UrnTransformerDecoder(
        vocab_size=config['model']['vocab_size'],
        context_len=config['model']['context_len'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        lex=False
    ).to(device)
    
    lex_model = UrnTransformerDecoder(
        vocab_size=config['model']['vocab_size'],
        context_len=config['model']['context_len'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        lex=True
    ).to(device)
    
    print("Models created successfully")
    
    # Test the generalization function
    n_test_tasks = 3  # Small number for quick test
    context_len = config['model']['context_len']
    vocab_size = config['model']['vocab_size']
    eval_context_len = context_len - 1
    
    print(f"Testing {n_test_tasks} new tasks")
    print(f"Context length: {context_len}, Vocab size: {vocab_size}")
    
    # Generate new urns
    new_urns = generate_urns(n_tasks=n_test_tasks, n_colors=vocab_size, alpha=1.0, device=device)
    print(f"Generated {n_test_tasks} new urns")
    
    normal_model.eval()
    lex_model.eval()
    
    normal_kl_divergences = []
    lex_kl_divergences = []
    
    with torch.no_grad():
        for task_idx in range(n_test_tasks):
            true_urn = new_urns[task_idx]
            
            # Sample context from this urn
            context_dist = torch.distributions.Categorical(true_urn)
            context_sequence = context_dist.sample((eval_context_len,))
            
            # Prepare model input
            context_input = context_sequence.unsqueeze(0).to(device)
            
            # Get predictions
            normal_logits = normal_model(context_input)
            lex_logits = lex_model(context_input)
            
            normal_pred = F.softmax(normal_logits[0, -1, :], dim=0)
            lex_pred = F.softmax(lex_logits[0, -1, :], dim=0)
            
            # Compute KL divergence
            def kl_div(p, q, eps=1e-10):
                p = p + eps
                q = q + eps
                p = p / p.sum()
                q = q / q.sum()
                return torch.sum(p * torch.log(p / q)).item()
            
            normal_kl = kl_div(true_urn, normal_pred)
            lex_kl = kl_div(true_urn, lex_pred)
            
            normal_kl_divergences.append(normal_kl)
            lex_kl_divergences.append(lex_kl)
            
            print(f"\nTask {task_idx}:")
            print(f"  True urn:     {true_urn.cpu().numpy().round(3)}")
            print(f"  Context:      {context_sequence.tolist()}")
            print(f"  Normal pred:  {normal_pred.cpu().numpy().round(3)} (KL: {normal_kl:.4f})")
            print(f"  Lexical pred: {lex_pred.cpu().numpy().round(3)} (KL: {lex_kl:.4f})")
    
    print(f"\nResults:")
    print(f"Normal model KL mean: {sum(normal_kl_divergences)/len(normal_kl_divergences):.4f}")
    print(f"Lexical model KL mean: {sum(lex_kl_divergences)/len(lex_kl_divergences):.4f}")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_generalization_quick()