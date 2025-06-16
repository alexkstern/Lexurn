#!/usr/bin/env python3
"""
Generalization test script for Lexurn project.

Tests model generalization ability on fresh unseen tasks by:
1. Loading trained models from checkpoints (both normal and lexical)
2. Generating new random urns (tasks) never seen during training
3. Creating test sequences from these urns
4. Comparing model predictions vs true in-context distributions using KL divergence
5. Comparing performance between normal and lexical invariance models

Usage:
    python test_generalization.py
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from model import UrnTransformerDecoder
from utils import load_model_config
from generate_urns import generate_urns


def calculate_in_context_distribution(sequence, vocab_size=4):
    """
    Calculate empirical distribution from context tokens (first context_len-1 tokens).
    
    Args:
        sequence: tensor of shape (context_len,) containing token ids
        vocab_size: number of possible tokens
    
    Returns:
        distribution: tensor of shape (vocab_size,) with probabilities
    """
    context = sequence[:-1]  # Exclude last token (prediction target)
    
    # Count occurrences of each token
    counts = torch.zeros(vocab_size)
    for token in context:
        counts[token] += 1
    
    # Convert to probabilities (add small epsilon to avoid zero probabilities)
    eps = 1e-8
    total = counts.sum()
    if total == 0:
        # If no context, uniform distribution
        distribution = torch.ones(vocab_size) / vocab_size
    else:
        distribution = (counts + eps) / (total + eps * vocab_size)
    
    return distribution


def generate_test_sequences(urns, context_len=8, num_samples_per_urn=32, device="cpu"):
    """
    Generate test sequences from urns.
    
    Args:
        urns: tensor of shape (num_urns, vocab_size) with probability distributions
        context_len: length of sequences to generate
        num_samples_per_urn: number of sequences to generate per urn
        device: device to use
    
    Returns:
        sequences: tensor of shape (num_urns * num_samples_per_urn, context_len)
        urn_indices: tensor mapping each sequence to its urn index
    """
    num_urns, vocab_size = urns.shape
    total_sequences = num_urns * num_samples_per_urn
    
    sequences = torch.zeros(total_sequences, context_len, dtype=torch.long, device=device)
    urn_indices = torch.zeros(total_sequences, dtype=torch.long, device=device)
    
    for urn_idx, urn in enumerate(urns):
        start_idx = urn_idx * num_samples_per_urn
        end_idx = start_idx + num_samples_per_urn
        
        # Sample sequences from this urn's distribution
        dist = torch.distributions.Categorical(urn)
        sequences[start_idx:end_idx] = dist.sample((num_samples_per_urn, context_len))
        urn_indices[start_idx:end_idx] = urn_idx
    
    return sequences, urn_indices


def test_generalization(checkpoint_path, config_path, num_tasks=10, num_samples=256, device="auto"):
    """
    Test model generalization on fresh unseen tasks.
    
    Args:
        checkpoint_path: path to saved model checkpoint
        config_path: path to config file
        num_tasks: number of new tasks (urns) to generate
        num_samples: total number of test samples to evaluate
        device: device to use for computation
    
    Returns:
        dict with results including mean KL divergence
    """
    print(f"Testing generalization with {num_tasks} tasks and {num_samples} samples")
    
    # Load config
    configs = load_model_config(config_path)
    model_params = configs["model"]
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model type from checkpoint filename or saved state
    is_lexical = 'lexical' in checkpoint_path.lower()
    model_params["lex"] = is_lexical
    
    # Create model
    model = UrnTransformerDecoder(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {'lexical' if is_lexical else 'normal'} model from {checkpoint_path}")
    
    # Generate fresh unseen urns
    urns = generate_urns(
        n_tasks=num_tasks, 
        n_colors=model_params["vocab_size"], 
        alpha=1.0, 
        device=device,
        seed=12345  # Different seed from training
    )
    print(f"Generated {num_tasks} fresh urns")
    
    # Generate test sequences
    samples_per_urn = num_samples // num_tasks
    sequences, urn_indices = generate_test_sequences(
        urns, 
        context_len=model_params["context_len"],
        num_samples_per_urn=samples_per_urn,
        device=device
    )
    
    print(f"Generated {len(sequences)} test sequences")
    
    # Calculate KL divergences
    kl_divergences = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_urn_indices = urn_indices[i:i+batch_size]
            
            # Get model predictions for last position
            logits = model(batch_sequences)  # Shape: (batch, seq_len, vocab_size)
            last_logits = logits[:, -1, :]  # Last position predictions
            model_probs = F.softmax(last_logits, dim=-1)
            
            # Calculate in-context distributions for each sequence
            for j in range(len(batch_sequences)):
                sequence = batch_sequences[j]
                urn_idx = batch_urn_indices[j]
                
                # Get model prediction
                model_dist = model_probs[j]
                
                # Get in-context distribution
                context_dist = calculate_in_context_distribution(
                    sequence, vocab_size=model_params["vocab_size"]
                ).to(device)
                
                # Get actual urn distribution
                actual_urn_dist = urns[urn_idx]
                
                # Calculate KL divergences
                kl_div_context = F.kl_div(
                    torch.log(model_dist + 1e-8),
                    context_dist,
                    reduction='sum'
                ).item()
                
                kl_div_urn = F.kl_div(
                    torch.log(model_dist + 1e-8),
                    actual_urn_dist,
                    reduction='sum'
                ).item()
                
                kl_divergences.append({
                    'kl_divergence_context': kl_div_context,
                    'kl_divergence_urn': kl_div_urn,
                    'urn_idx': urn_idx.item(),
                    'model_dist': model_dist.cpu().numpy(),
                    'context_dist': context_dist.cpu().numpy(),
                    'actual_urn_dist': actual_urn_dist.cpu().numpy(),
                    'sequence': sequence.cpu().numpy()
                })
    
    # Calculate statistics for both KL divergences
    kl_context_values = [item['kl_divergence_context'] for item in kl_divergences]
    kl_urn_values = [item['kl_divergence_urn'] for item in kl_divergences]
    
    mean_kl_context = np.mean(kl_context_values)
    std_kl_context = np.std(kl_context_values)
    median_kl_context = np.median(kl_context_values)
    
    mean_kl_urn = np.mean(kl_urn_values)
    std_kl_urn = np.std(kl_urn_values)
    median_kl_urn = np.median(kl_urn_values)
    
    results = {
        'checkpoint_path': checkpoint_path,
        'config_path': config_path,
        'is_lexical': is_lexical,
        'num_tasks': num_tasks,
        'num_samples': len(kl_divergences),
        'mean_kl_divergence_context': mean_kl_context,
        'std_kl_divergence_context': std_kl_context,
        'median_kl_divergence_context': median_kl_context,
        'mean_kl_divergence_urn': mean_kl_urn,
        'std_kl_divergence_urn': std_kl_urn,
        'median_kl_divergence_urn': median_kl_urn,
        'all_results': kl_divergences
    }
    
    # Print summary
    print("\n" + "="*60)
    print("GENERALIZATION TEST RESULTS")
    print("="*60)
    print(f"Model Type: {'Lexical Invariance' if is_lexical else 'Normal'}")
    print(f"Tasks Tested: {num_tasks}")
    print(f"Samples Evaluated: {len(kl_divergences)}")
    print(f"\nKL Divergence vs In-Context Distribution:")
    print(f"  Mean: {mean_kl_context:.4f} ± {std_kl_context:.4f}")
    print(f"  Median: {median_kl_context:.4f}")
    print(f"\nKL Divergence vs Actual Urn Distribution:")
    print(f"  Mean: {mean_kl_urn:.4f} ± {std_kl_urn:.4f}")
    print(f"  Median: {median_kl_urn:.4f}")
    
    # Show some example comparisons
    print("\nExample Predictions vs Context & Urn Distributions:")
    print("-" * 70)
    for i in range(min(5, len(kl_divergences))):
        item = kl_divergences[i]
        print(f"Sample {i+1} (KL_context={item['kl_divergence_context']:.4f}, KL_urn={item['kl_divergence_urn']:.4f}):")
        print(f"  Model:    {item['model_dist']}")
        print(f"  Context:  {item['context_dist']}")
        print(f"  Urn:      {item['actual_urn_dist']}")
        print(f"  Sequence: {item['sequence']}")
        print()
    
    return results


def test_both_models_fair_comparison(normal_checkpoint, lexical_checkpoint, config_path, num_tasks=10, num_samples=256, device="cuda"):
    """
    Test both models on the same test tasks for fair comparison.
    
    Args:
        normal_checkpoint: Path to normal model checkpoint
        lexical_checkpoint: Path to lexical model checkpoint
        config_path: Path to config file
        num_tasks: Number of test tasks to generate
        num_samples: Total number of test samples
        device: Device to use
    
    Returns:
        Dictionary with results for both models
    """
    print("Testing generalization on both models with identical test tasks...")
    print("="*80)
    
    # Load config
    configs = load_model_config(config_path)
    model_params = configs["model"]
    
    # Generate test tasks once for fair comparison
    urns = generate_urns(
        n_tasks=num_tasks, 
        n_colors=model_params["vocab_size"], 
        alpha=1.0, 
        device=device,
        seed=12345  # Fixed seed for reproducibility
    )
    print(f"Generated {num_tasks} test urns for fair comparison")
    
    # Generate test sequences once
    samples_per_urn = num_samples // num_tasks
    sequences, urn_indices = generate_test_sequences(
        urns, 
        context_len=model_params["context_len"],
        num_samples_per_urn=samples_per_urn,
        device=device
    )
    print(f"Generated {len(sequences)} test sequences")
    
    # Test normal model
    print("\n1. TESTING NORMAL MODEL")
    print("-" * 40)
    normal_results = test_model_on_sequences(
        checkpoint_path=normal_checkpoint,
        config_path=config_path,
        sequences=sequences,
        urn_indices=urn_indices,
        urns=urns,
        device=device
    )
    
    # Test lexical model
    print("\n2. TESTING LEXICAL MODEL")
    print("-" * 40)
    lexical_results = test_model_on_sequences(
        checkpoint_path=lexical_checkpoint,
        config_path=config_path,
        sequences=sequences,
        urn_indices=urn_indices,
        urns=urns,
        device=device
    )
    
    return normal_results, lexical_results, urns, sequences


def test_model_on_sequences(checkpoint_path, config_path, sequences, urn_indices, urns, device="cuda"):
    """
    Test a single model on pre-generated sequences.
    """
    # Load config
    configs = load_model_config(config_path)
    model_params = configs["model"]
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model type from checkpoint filename
    is_lexical = 'lexical' in checkpoint_path.lower()
    model_params["lex"] = is_lexical
    
    # Create model
    model = UrnTransformerDecoder(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {'lexical' if is_lexical else 'normal'} model from {checkpoint_path}")
    
    # Calculate KL divergences
    kl_divergences = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_urn_indices = urn_indices[i:i+batch_size]
            
            # Get model predictions for last position
            logits = model(batch_sequences)  # Shape: (batch, seq_len, vocab_size)
            last_logits = logits[:, -1, :]  # Last position predictions
            model_probs = F.softmax(last_logits, dim=-1)
            
            # Calculate in-context distributions for each sequence
            for j in range(len(batch_sequences)):
                sequence = batch_sequences[j]
                urn_idx = batch_urn_indices[j]
                
                # Get model prediction
                model_dist = model_probs[j]
                
                # Get in-context distribution
                context_dist = calculate_in_context_distribution(
                    sequence, vocab_size=model_params["vocab_size"]
                ).to(device)
                
                # Get actual urn distribution
                actual_urn_dist = urns[urn_idx]
                
                # Calculate KL divergences
                kl_div_context = F.kl_div(
                    torch.log(model_dist + 1e-8),
                    context_dist,
                    reduction='sum'
                ).item()
                
                kl_div_urn = F.kl_div(
                    torch.log(model_dist + 1e-8),
                    actual_urn_dist,
                    reduction='sum'
                ).item()
                
                kl_divergences.append({
                    'kl_divergence_context': kl_div_context,
                    'kl_divergence_urn': kl_div_urn,
                    'urn_idx': urn_idx.item(),
                    'model_dist': model_dist.cpu().numpy(),
                    'context_dist': context_dist.cpu().numpy(),
                    'actual_urn_dist': actual_urn_dist.cpu().numpy(),
                    'sequence': sequence.cpu().numpy()
                })
    
    # Calculate statistics
    kl_context_values = [item['kl_divergence_context'] for item in kl_divergences]
    kl_urn_values = [item['kl_divergence_urn'] for item in kl_divergences]
    
    mean_kl_context = np.mean(kl_context_values)
    std_kl_context = np.std(kl_context_values)
    mean_kl_urn = np.mean(kl_urn_values)
    std_kl_urn = np.std(kl_urn_values)
    
    results = {
        'checkpoint_path': checkpoint_path,
        'is_lexical': is_lexical,
        'num_samples': len(kl_divergences),
        'mean_kl_divergence_context': mean_kl_context,
        'std_kl_divergence_context': std_kl_context,
        'mean_kl_divergence_urn': mean_kl_urn,
        'std_kl_divergence_urn': std_kl_urn,
        'all_results': kl_divergences
    }
    
    # Print summary
    print(f"Model Type: {'Lexical Invariance' if is_lexical else 'Normal'}")
    print(f"Samples Evaluated: {len(kl_divergences)}")
    print(f"KL Divergence vs In-Context: {mean_kl_context:.4f} ± {std_kl_context:.4f}")
    print(f"KL Divergence vs Actual Urn: {mean_kl_urn:.4f} ± {std_kl_urn:.4f}")
    
    return results


if __name__ == "__main__":
    # Test parameters
    config_path = "configs/single_task.config"
    normal_checkpoint = "checkpoints/normal_single_task_experiment_20250615_110926.pt"
    lexical_checkpoint = "checkpoints/lexical_single_task_experiment_20250615_110926.pt"
    num_tasks = 10
    num_samples = 256
    device = "cuda"
    
    # Test both models with identical test tasks
    normal_results, lexical_results, test_urns, test_sequences = test_both_models_fair_comparison(
        normal_checkpoint=normal_checkpoint,
        lexical_checkpoint=lexical_checkpoint,
        config_path=config_path,
        num_tasks=num_tasks,
        num_samples=num_samples,
        device=device
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("KL Divergence vs In-Context Distribution:")
    print(f"  Normal Model:  {normal_results['mean_kl_divergence_context']:.4f} ± {normal_results['std_kl_divergence_context']:.4f}")
    print(f"  Lexical Model: {lexical_results['mean_kl_divergence_context']:.4f} ± {lexical_results['std_kl_divergence_context']:.4f}")
    
    kl_diff_context = lexical_results['mean_kl_divergence_context'] - normal_results['mean_kl_divergence_context']
    print(f"  Difference (Lexical - Normal): {kl_diff_context:.4f}")
    
    print("\nKL Divergence vs Actual Urn Distribution:")
    print(f"  Normal Model:  {normal_results['mean_kl_divergence_urn']:.4f} ± {normal_results['std_kl_divergence_urn']:.4f}")
    print(f"  Lexical Model: {lexical_results['mean_kl_divergence_urn']:.4f} ± {lexical_results['std_kl_divergence_urn']:.4f}")
    
    kl_diff_urn = lexical_results['mean_kl_divergence_urn'] - normal_results['mean_kl_divergence_urn']
    print(f"  Difference (Lexical - Normal): {kl_diff_urn:.4f}")
    
    print("\nInterpretation:")
    if kl_diff_context < 0:
        print("✓ Lexical model has LOWER KL vs in-context (better at frequency counting)")
    else:
        print("✗ Normal model has LOWER KL vs in-context")
        
    if kl_diff_urn < 0:
        print("✓ Lexical model has LOWER KL vs actual urn (better true generalization)")
    else:
        print("✗ Normal model has LOWER KL vs actual urn")
    
    # Save results with test data for reproducibility
    results_summary = {
        'normal': normal_results,
        'lexical': lexical_results,
        'test_urns': test_urns,
        'test_sequences': test_sequences,
        'comparison': {
            'kl_difference_context': kl_diff_context,
            'kl_difference_urn': kl_diff_urn,
            'lexical_better_context': kl_diff_context < 0,
            'lexical_better_urn': kl_diff_urn < 0
        }
    }
    
    torch.save(results_summary, "results/generalization_test_comparison.pt", _use_new_zipfile_serialization=False)
    print(f"\nResults saved to: results/generalization_test_comparison.pt")