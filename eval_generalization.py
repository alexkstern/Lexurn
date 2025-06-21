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
from tqdm import tqdm
from model import UrnTransformerDecoder
from utils import load_model_config, kl_div
from generate_urns import generate_urns

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Ensure reproducible random number generation
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_relative_distance(model_dist, bayesian_posterior_dist, memorizing_dist):
    """
    Calculate relative distance metric from the paper.
    
    The metric compares how similar the model is to two reference predictors:
    - G: generalizing predictor (Bayesian posterior with uniform prior)
    - M: memorizing predictor (training urn distribution)
    
    Formula: r = (d(h, G) - d(h, M)) / d(G, M)
    Then: d_rel = (r + 1) / 2
    
    Where:
    - h = model prediction distribution
    - G = Bayesian posterior with uniform prior (generalizing predictor)
    - M = memorizing predictor (training urn distribution)
    - d(·, ·) = KL divergence
    
    Interpretation:
    - d_rel ≈ 0: Model closer to generalizer (uses context-aware Bayesian prediction)
    - d_rel ≈ 1: Model closer to memorizer (uses training urn distribution)
    - d_rel ≈ 0.5: Model equidistant between both predictors
    
    Args:
        model_dist: Model's predicted distribution
        bayesian_posterior_dist: Bayesian posterior from context tokens (G)
        memorizing_dist: Training urn distribution (M)
    
    Returns:
        relative_distance: Bounded metric in [0, 1]
    """
    # d(h, G) = KL(h || G) - how far is model from generalizer
    kl_model_generalizer = kl_div(model_dist, bayesian_posterior_dist)
    
    # d(h, M) = KL(h || M) - how far is model from memorizer
    kl_model_memorizer = kl_div(model_dist, memorizing_dist)
    
    # d(G, M) - distance between generalizer and memorizer
    kl_generalizer_memorizer = kl_div(bayesian_posterior_dist, memorizing_dist)
    
    # Avoid division by zero
    if abs(kl_generalizer_memorizer) < 1e-8:
        # If G and M are identical, return 0.5 (equidistant)
        return 0.5
    
    # Calculate r = (d(h, G) - d(h, M)) / d(G, M)
    r = (kl_model_generalizer - kl_model_memorizer) / kl_generalizer_memorizer
    
    # Rescale to [0, 1]: d_rel = (r + 1) / 2
    relative_distance = (r + 1) / 2
    
    return relative_distance


def calculate_in_context_distribution(sequence, vocab_size=4):
    """
    Calculate Bayesian posterior distribution with uniform Dirichlet prior from context tokens.
    
    With uniform Dirichlet prior (alpha=1 for all tokens), the posterior is:
    posterior(token) = (count(token) + 1) / (total_count + vocab_size)
    
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
    
    # Bayesian posterior with uniform Dirichlet prior (alpha=1)
    # posterior = (counts + alpha) / (total + alpha * vocab_size)
    # With alpha=1: posterior = (counts + 1) / (total + vocab_size)
    total = counts.sum()
    distribution = (counts + 1) / (total + vocab_size)
    
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
    
    print(f"Generating {total_sequences} test sequences from {num_urns} urns...")
    for urn_idx, urn in enumerate(tqdm(urns, desc="Sampling from urns")):
        start_idx = urn_idx * num_samples_per_urn
        end_idx = start_idx + num_samples_per_urn
        
        # Sample sequences from this urn's distribution
        dist = torch.distributions.Categorical(urn)
        sequences[start_idx:end_idx] = dist.sample((num_samples_per_urn, context_len))
        urn_indices[start_idx:end_idx] = urn_idx
    
    return sequences, urn_indices


def load_model(checkpoint_path, config_path, device="cuda"):
    """Load model from checkpoint with proper configuration."""
    configs = load_model_config(config_path)
    model_params = configs["model"]
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    is_lexical = 'lexical' in checkpoint_path.lower()
    model_params["lex"] = is_lexical
    
    model = UrnTransformerDecoder(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {'lexical' if is_lexical else 'normal'} model from {checkpoint_path}")
    return model, model_params, is_lexical


def calculate_kl_divergences(model, sequences, urn_indices, urns, model_params, memorizing_dist=None, device="cuda"):
    """Calculate KL divergences for model predictions vs context and urn distributions."""
    kl_divergences = []
    
    with torch.no_grad():
        batch_size = 32
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Evaluating model", total=num_batches):
            batch_sequences = sequences[i:i+batch_size]
            batch_urn_indices = urn_indices[i:i+batch_size]
            
            logits = model(batch_sequences)
            last_logits = logits[:, -1, :]
            model_probs = F.softmax(last_logits, dim=-1)
            
            for j in range(len(batch_sequences)):
                sequence = batch_sequences[j]
                urn_idx = batch_urn_indices[j]
                model_dist = model_probs[j]
                
                context_dist = calculate_in_context_distribution(
                    sequence, vocab_size=model_params["vocab_size"]
                ).to(device)
                
                actual_urn_dist = urns[urn_idx]
                
                kl_div_context = kl_div(model_dist, context_dist)
                
                kl_div_urn = kl_div(model_dist, actual_urn_dist)
                
                # Calculate relative distance if memorizing_dist is provided
                relative_dist = None
                if memorizing_dist is not None:
                    relative_dist = calculate_relative_distance(
                        model_dist, context_dist, memorizing_dist
                    )
                
                kl_divergences.append({
                    'kl_divergence_context': kl_div_context,
                    'kl_divergence_urn': kl_div_urn,
                    'relative_distance': relative_dist,
                    'urn_idx': urn_idx.item(),
                    'model_dist': model_dist.cpu().numpy(),
                    'context_dist': context_dist.cpu().numpy(),
                    'actual_urn_dist': actual_urn_dist.cpu().numpy(),
                    'sequence': sequence.cpu().numpy()
                })
    
    return kl_divergences


def calculate_statistics(kl_divergences):
    """Calculate statistics from KL divergence results."""
    kl_context_values = [item['kl_divergence_context'] for item in kl_divergences]
    kl_urn_values = [item['kl_divergence_urn'] for item in kl_divergences]
    
    # Filter out None values for relative distance
    relative_dist_values = [item['relative_distance'] for item in kl_divergences if item['relative_distance'] is not None]
    
    stats = {
        'mean_kl_divergence_context': np.mean(kl_context_values),
        'std_kl_divergence_context': np.std(kl_context_values),
        'median_kl_divergence_context': np.median(kl_context_values),
        'mean_kl_divergence_urn': np.mean(kl_urn_values),
        'std_kl_divergence_urn': np.std(kl_urn_values),
        'median_kl_divergence_urn': np.median(kl_urn_values),
    }
    
    # Only add relative distance stats if we have values
    if relative_dist_values:
        stats.update({
            'mean_relative_distance': np.mean(relative_dist_values),
            'std_relative_distance': np.std(relative_dist_values),
            'median_relative_distance': np.median(relative_dist_values)
        })
    else:
        stats.update({
            'mean_relative_distance': None,
            'std_relative_distance': None,
            'median_relative_distance': None
        })
    
    return stats


def print_results_summary(results, is_lexical, num_tasks, num_samples, kl_divergences):
    """Print formatted results summary."""
    print("\n" + "="*60)
    print("GENERALIZATION TEST RESULTS")
    print("="*60)
    print(f"Model Type: {'Lexical Invariance' if is_lexical else 'Normal'}")
    print(f"Tasks Tested: {num_tasks}")
    print(f"Samples Evaluated: {num_samples}")
    print(f"\nKL Divergence vs In-Context Distribution:")
    print(f"  Mean: {results['mean_kl_divergence_context']:.4f} ± {results['std_kl_divergence_context']:.4f}")
    print(f"  Median: {results['median_kl_divergence_context']:.4f}")
    print(f"\nKL Divergence vs Actual Urn Distribution:")
    print(f"  Mean: {results['mean_kl_divergence_urn']:.4f} ± {results['std_kl_divergence_urn']:.4f}")
    print(f"  Median: {results['median_kl_divergence_urn']:.4f}")
    
    print(f"\nRelative Distance (0 = generalizing, 1 = memorizing, 0.5 = equidistant):")
    print(f"  Mean: {results['mean_relative_distance']:.4f} ± {results['std_relative_distance']:.4f}")
    print(f"  Median: {results['median_relative_distance']:.4f}")
    
    # Show some example comparisons (randomly selected)
    print("\nExample Predictions vs Context & Urn Distributions:")
    print("-" * 70)
    num_examples = min(10, len(kl_divergences))
    indices = np.random.choice(len(kl_divergences), size=num_examples, replace=False)
    for i, idx in enumerate(indices):
        item = kl_divergences[idx]
        print(f"Sample {i+1} (KL_context={item['kl_divergence_context']:.4f}, KL_urn={item['kl_divergence_urn']:.4f}, Rel_dist={item['relative_distance']:.4f}):")
        print(f"  Model:    {item['model_dist']}")
        print(f"  Context:  {item['context_dist']}")
        print(f"  Urn:      {item['actual_urn_dist']}")
        print(f"  Sequence: {item['sequence']}")
        print()


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
    print(f"Testing generalization with {num_tasks} tasks and {num_samples} samples per task ({num_tasks * num_samples} total samples)")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, model_params, is_lexical = load_model(checkpoint_path, config_path, device)
    
    # Generate fresh unseen urns
    print(f"Generating {num_tasks} fresh test urns...")
    with tqdm(total=1, desc="Creating test urns") as pbar:
        urns = generate_urns(
            n_tasks=num_tasks, 
            n_colors=model_params["vocab_size"], 
            alpha=1.0, 
            device=device,
            seed=12345  # Different seed from training
        )
        pbar.update(1)
    print(f"Generated {num_tasks} fresh urns")
    
    # Generate test sequences
    samples_per_urn = num_samples
    sequences, urn_indices = generate_test_sequences(
        urns, 
        context_len=model_params["context_len"],
        num_samples_per_urn=samples_per_urn,
        device=device
    )
    
    print(f"Generated {len(sequences)} test sequences")
    
    # Calculate KL divergences
    kl_divergences = calculate_kl_divergences(model, sequences, urn_indices, urns, model_params, device)
    
    # Calculate statistics
    stats = calculate_statistics(kl_divergences)
    
    results = {
        'checkpoint_path': checkpoint_path,
        'config_path': config_path,
        'is_lexical': is_lexical,
        'num_tasks': num_tasks,
        'num_samples': len(kl_divergences),
        'all_results': kl_divergences,
        **stats
    }
    
    # Print summary
    print_results_summary(results, is_lexical, num_tasks, len(kl_divergences), kl_divergences)
    
    return results


def print_detailed_logits_comparison(normal_results, lexical_results, sequences, urn_indices, num_examples=10):
    """Print detailed logits comparison for randomly selected examples."""
    print("\n" + "="*80)
    print("DETAILED LOGITS COMPARISON - RANDOMLY SELECTED EXAMPLES")
    print("="*80)
    
    num_examples = min(num_examples, len(sequences))
    indices = np.random.choice(len(sequences), size=num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        sequence = sequences[idx]
        urn_idx = urn_indices[idx]
        
        normal_item = normal_results['all_results'][idx]
        lexical_item = lexical_results['all_results'][idx]
        
        print(f"\nExample {i+1}:")
        print(f"Sequence: {sequence.cpu().numpy()}")
        print(f"Urn Index: {urn_idx.item()}")
        
        print(f"Normal Model:    {normal_item['model_dist']}")
        print(f"Lexical Model:   {lexical_item['model_dist']}")
        print(f"Context Dist:    {normal_item['context_dist']}")
        print(f"Actual Urn:      {normal_item['actual_urn_dist']}")
        
        print(f"KL(Normal|Context):  {normal_item['kl_divergence_context']:.4f}")
        print(f"KL(Lexical|Context): {lexical_item['kl_divergence_context']:.4f}")
        print(f"KL(Normal|Urn):      {normal_item['kl_divergence_urn']:.4f}")
        print(f"KL(Lexical|Urn):     {lexical_item['kl_divergence_urn']:.4f}")
        print(f"Rel Dist Normal:     {normal_item['relative_distance']:.4f}")
        print(f"Rel Dist Lexical:    {lexical_item['relative_distance']:.4f}")


def test_both_models_fair_comparison(normal_checkpoint, lexical_checkpoint, config_path, memorizing_dist=None, num_tasks=10, num_samples=256, device="cuda"):
    """
    Test both models on the same test tasks for fair comparison.
    
    Args:
        normal_checkpoint: Path to normal model checkpoint
        lexical_checkpoint: Path to lexical model checkpoint
        config_path: Path to config file
        memorizing_dist: Training urn distribution (M) for relative distance calculation
        num_tasks: Number of test tasks to generate
        num_samples: Total number of test samples
        device: Device to use
    
    Returns:
        Dictionary with results for both models
    """
    print(f"Testing generalization on both models with {num_tasks} tasks and {num_samples} samples per task ({num_tasks * num_samples} total samples)...")
    print("="*80)
    
    # Load config
    configs = load_model_config(config_path)
    model_params = configs["model"]
    
    # Generate larger pool of test tasks and randomly select subset
    # This ensures we don't always test the same tasks when num_tasks < total_pool
    total_task_pool = num_tasks # Generate at least 50 tasks or 2x requested
    print(f"Generating {total_task_pool} test urns...")
    with tqdm(total=1, desc="Creating test urns") as pbar:
        all_urns = generate_urns(
            n_tasks=total_task_pool, 
            n_colors=model_params["vocab_size"], 
            alpha=1.0, 
            device=device,
            seed=12345  # Fixed seed for reproducibility
        )
        pbar.update(1)
    
    # Randomly select num_tasks from the pool
    torch.manual_seed(54321)  # Different seed for task selection
    selected_indices = torch.randperm(total_task_pool)[:num_tasks]
    urns = all_urns[selected_indices]
    
    print(f"Generated {total_task_pool} test urns, randomly selecting {num_samples} per urn")
    
    # Generate test sequences once
    samples_per_urn = num_samples
    sequences, urn_indices = generate_test_sequences(
        urns, 
        context_len=model_params["context_len"],
        num_samples_per_urn=samples_per_urn,
        device=device
    )
    print(f"Generated {len(sequences)} test sequences")
    
    # Check if memorizing distribution is provided for relative distance calculation
    if memorizing_dist is not None:
        print(f"Using provided memorizing distribution M: {memorizing_dist.cpu().numpy()}")
    else:
        print("WARNING: No memorizing distribution (M) provided. Relative distance will not be calculated.")
    
    # Test normal model
    print("\n1. TESTING NORMAL MODEL")
    print("-" * 40)
    normal_results = test_model_on_sequences(
        checkpoint_path=normal_checkpoint,
        config_path=config_path,
        sequences=sequences,
        urn_indices=urn_indices,
        urns=urns,
        memorizing_dist=memorizing_dist,
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
        memorizing_dist=memorizing_dist,
        device=device
    )
    
    return normal_results, lexical_results, urns, sequences, urn_indices


def test_model_on_sequences(checkpoint_path, config_path, sequences, urn_indices, urns, memorizing_dist=None, device="cuda"):
    """Test a single model on pre-generated sequences."""
    model, model_params, is_lexical = load_model(checkpoint_path, config_path, device)
    
    # Calculate KL divergences
    kl_divergences = calculate_kl_divergences(model, sequences, urn_indices, urns, model_params, memorizing_dist, device)
    
    # Calculate statistics
    stats = calculate_statistics(kl_divergences)
    
    results = {
        'checkpoint_path': checkpoint_path,
        'is_lexical': is_lexical,
        'num_samples': len(kl_divergences),
        'all_results': kl_divergences,
        **stats
    }
    
    # Print summary
    print(f"Model Type: {'Lexical Invariance' if is_lexical else 'Normal'}")
    print(f"Samples Evaluated: {len(kl_divergences)}")
    print(f"KL Divergence vs In-Context: {results['mean_kl_divergence_context']:.4f} ± {results['std_kl_divergence_context']:.4f}")
    print(f"KL Divergence vs Actual Urn: {results['mean_kl_divergence_urn']:.4f} ± {results['std_kl_divergence_urn']:.4f}")
    if memorizing_dist is not None:
        print(f"Relative Distance: {results['mean_relative_distance']:.4f} ± {results['std_relative_distance']:.4f}")
    else:
        print("Relative Distance: Not calculated (no memorizing distribution provided)")
    
    return results


if __name__ == "__main__":
    # Set seed for reproducible evaluation
    set_seed(42)

    num_tasks = 30
    num_samples = 1000
    device = "cuda"
    
    # Test parameters
    #seq_30 4.2M param model
    #config_path = "configs/seq_30_4M.config"
    #normal_checkpoint = "checkpoints/seq_30_4M/normal_single_task_experiment_20250615_110926.pt"
    #lexical_checkpoint = "checkpoints/seq_30_4M/lexical_single_task_experiment_20250615_110926.pt"
    # Define memorizing predictor M 
    #memorizing_dist = torch.tensor([0.4105, 0.3253, 0.0443, 0.2200], device=device)


    #seq_80 4.2M param model
    #config_path = "configs/seq_80_4M_single_task_normal.config"
    #normal_checkpoint = "checkpoints/seq_80_4M/normal_single_task_normal_20250617_012209.pt"
    #lexical_checkpoint = "checkpoints/seq_80_4M/lexical_single_task_lexical_20250617_012740.pt"
    #Define memorizing predictor M
    #memorizing_dist = torch.tensor([0.4105, 0.3253, 0.0443, 0.2200], device=device)
    
    #seq_10 4.2M param model
    #config_path = "configs/seq_10_4M_single_task_normal.config"
    #normal_checkpoint = "checkpoints/seq_10_4M/normal_single_task_normal_20250617_093952.pt"
    #lexical_checkpoint = "checkpoints/seq_10_4M/lexical_single_task_lexical_20250617_094308.pt"
    #Define memorizing predictor M
    #memorizing_dist = torch.tensor([0.4105, 0.3253, 0.0443, 0.2200], device=device)

    #seq_10 4.2M param model 2
    #config_path = "configs/seq_10_4M_single_task_normal.config"
    #normal_checkpoint = "checkpoints/seq_10_4M_2/normal_single_task_normal_20250617_113536.pt"
    #lexical_checkpoint = "checkpoints/seq_10_4M_2/lexical_single_task_lexical_20250617_113729.pt"
    #Define memorizing predictor M
    #memorizing_dist = torch.tensor([0.4105, 0.3253, 0.0443, 0.2200], device=device)


    #config_path = "configs/seq_180_4M_single_task_normal.config"
    #normal_checkpoint = "checkpoints/seq_180_4M/normal_single_task_normal_20250618_232741.pt"
    #lexical_checkpoint = "checkpoints/seq_180_4M/lexical_single_task_lexical_20250618_233133.pt"

    #config_path = "configs/seq_180_4M_single_task_normal.config"
    #normal_checkpoint = "checkpoints/seq_180_4M/normal_single_task_normal_20250618_232741.pt"
    #lexical_checkpoint = "checkpoints/seq_180_4M/lexical_single_task_lexical_20250618_233133.pt"

    #Define memorizing predictor M
    #memorizing_dist = torch.tensor([0.4105, 0.3253, 0.0443, 0.2200], device=device)


    config_path = "configs/seq_30_4M_single_task_10_epochs_lexical.config"
    normal_checkpoint = "checkpoints/seq_30_4M_epochs_10_seed_43/normal_single_task_normal_20250620_220519.pt"
    lexical_checkpoint = "checkpoints/seq_30_4M_epochs_10_seed_43/lexical_single_task_lexical_20250620_220652.pt"

    #Define memorizing predictor M
    #urn_1=torch.tensor([0.4105, 0.3253, 0.0443, 0.2200], device=device)
    ##urn_2= torch.tensor([0.273, 0.222, 0.5,   0.005],device=device)
    #memorizing_dist = torch.mean(torch.stack([urn_1,urn_2]),dim=0)

    memorizing_dist=torch.tensor([0.082, 0.074, 0.674, 0.171], device=device)


    # Test both models with identical test tasks
    normal_results, lexical_results, test_urns, test_sequences, test_urn_indices = test_both_models_fair_comparison(
        normal_checkpoint=normal_checkpoint,
        lexical_checkpoint=lexical_checkpoint,
        config_path=config_path,
        memorizing_dist=memorizing_dist,
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
    
    print("\nRelative Distance:")
    if normal_results['mean_relative_distance'] is not None and lexical_results['mean_relative_distance'] is not None:
        print(f"  Normal Model:  {normal_results['mean_relative_distance']:.4f} ± {normal_results['std_relative_distance']:.4f}")
        print(f"  Lexical Model: {lexical_results['mean_relative_distance']:.4f} ± {lexical_results['std_relative_distance']:.4f}")
        
        rel_dist_diff = lexical_results['mean_relative_distance'] - normal_results['mean_relative_distance']
        print(f"  Difference (Lexical - Normal): {rel_dist_diff:.4f}")
    else:
        print("  Not calculated (memorizing distribution M not provided)")
        rel_dist_diff = None
    
    # Print detailed comparison for first 10 examples
    print_detailed_logits_comparison(normal_results, lexical_results, test_sequences, 
                                   test_urn_indices, num_examples=10)
    
    # Save results with test data for reproducibility
    results_summary = {
        'normal': normal_results,
        'lexical': lexical_results,
        'test_urns': test_urns,
        'test_sequences': test_sequences,
        'comparison': {
            'kl_difference_context': kl_diff_context,
            'kl_difference_urn': kl_diff_urn,
            'relative_distance_difference': rel_dist_diff,
            'lexical_better_context': kl_diff_context < 0,
            'lexical_better_urn': kl_diff_urn < 0,
            'lexical_more_generalizing': rel_dist_diff < 0 if rel_dist_diff is not None else None
        }
    }
    
    torch.save(results_summary, "results/generalization_test_comparison.pt", _use_new_zipfile_serialization=False)
    print(f"\nResults saved to: results/generalization_test_comparison.pt")