import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from utils import kl_div


def analyze_entropy_kl_relationship(kl_divs: np.ndarray, 
                                   sequences: torch.Tensor, 
                                   model_name: str = "Model") -> Dict:
    """
    Analyze the relationship between sequence entropy and KL divergence.
    
    Hypothesis: High entropy sequences → low KL, low entropy sequences → higher KL
    
    Args:
        kl_divs: Array of KL divergences for each sequence
        sequences: Test sequences used for evaluation
        model_name: Name for reporting
    
    Returns:
        Dictionary with entropy-KL relationship analysis
    """
    # Calculate entropy for each full sequence
    entropies = []
    
    for seq in sequences:
        # Count each token (0=red, 1=green, 2=blue, 3=yellow)
        counts = torch.bincount(seq.long(), minlength=4).float()
        
        # Calculate empirical entropy
        probs = counts / counts.sum()
        entropy = torch.distributions.Categorical(probs).entropy()
        entropies.append(entropy.item())
    
    entropies = np.array(entropies)
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(entropies, kl_divs)
    spearman_corr, spearman_p = spearmanr(entropies, kl_divs)
    
    # Bin sequences by entropy quartiles
    entropy_quartiles = np.percentile(entropies, [25, 50, 75])
    
    bins = {
        'low_entropy': entropies <= entropy_quartiles[0],
        'mid_low_entropy': (entropies > entropy_quartiles[0]) & (entropies <= entropy_quartiles[1]),
        'mid_high_entropy': (entropies > entropy_quartiles[1]) & (entropies <= entropy_quartiles[2]),
        'high_entropy': entropies > entropy_quartiles[2]
    }
    
    bin_stats = {}
    for bin_name, mask in bins.items():
        if np.sum(mask) > 0:
            bin_stats[bin_name] = {
                'count': np.sum(mask),
                'mean_entropy': np.mean(entropies[mask]),
                'mean_kl': np.mean(kl_divs[mask]),
                'std_kl': np.std(kl_divs[mask])
            }
    
    results = {
        'entropies': entropies,
        'kl_divs': kl_divs,
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'entropy_quartiles': entropy_quartiles,
        'bin_stats': bin_stats,
        'overall_stats': {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'mean_kl': np.mean(kl_divs),
            'std_kl': np.std(kl_divs)
        }
    }
    
    # Print analysis
    print(f"\n{model_name} Entropy-KL Relationship Analysis:")
    print(f"  Overall: entropy {results['overall_stats']['mean_entropy']:.3f}±{results['overall_stats']['std_entropy']:.3f}, KL {results['overall_stats']['mean_kl']:.4f}±{results['overall_stats']['std_kl']:.4f}")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
    
    print(f"\n  Entropy bins (quartiles):")
    for bin_name, stats in bin_stats.items():
        print(f"    {bin_name}: {stats['count']} seqs, entropy={stats['mean_entropy']:.3f}, KL={stats['mean_kl']:.4f}±{stats['std_kl']:.4f}")
    
    # Test hypothesis: high entropy → low KL
    if 'high_entropy' in bin_stats and 'low_entropy' in bin_stats:
        high_entropy_kl = bin_stats['high_entropy']['mean_kl']
        low_entropy_kl = bin_stats['low_entropy']['mean_kl']
        kl_difference = low_entropy_kl - high_entropy_kl
        print(f"\n  Hypothesis test: Low entropy KL - High entropy KL = {kl_difference:.4f}")
        if kl_difference > 0:
            print(f"    ✓ SUPPORTS hypothesis: low entropy → higher KL, high entropy → lower KL")
        else:
            print(f"    ✗ CONTRADICTS hypothesis: high entropy → higher KL")
    
    return results


# Legacy function for backwards compatibility
def analyze_outliers(kl_divs: np.ndarray, 
                    sequences: torch.Tensor, 
                    urn_indices: torch.Tensor,
                    test_urns: torch.Tensor,
                    bayesian_dists: torch.Tensor,
                    model_name: str = "Model") -> Dict:
    """Legacy wrapper - now focuses on entropy-KL relationship."""
    return analyze_entropy_kl_relationship(kl_divs, sequences, model_name)


def compare_outlier_groups(good_analysis: Dict, bad_analysis: Dict) -> None:
    """Legacy function - now does nothing to reduce clutter."""
    pass


def show_detailed_outliers(kl_divs: np.ndarray,
                          sequences: torch.Tensor,
                          bayesian_dists: torch.Tensor,
                          model_predictions: torch.Tensor,
                          model_name: str = "Model",
                          n_best: int = 5,
                          n_worst: int = 5) -> None:
    """
    Show detailed analysis of best and worst performing sequences.
    
    Args:
        kl_divs: KL divergences for each sequence
        sequences: Test sequences 
        bayesian_dists: Bayesian posterior distributions
        model_predictions: Model output distributions
        model_name: Name for reporting
        n_best: Number of best examples to show
        n_worst: Number of worst examples to show
    """
    # Calculate entropy for each full sequence
    entropies = []
    for seq in sequences:
        counts = torch.bincount(seq.long(), minlength=4).float()
        probs = counts / counts.sum()
        entropy = torch.distributions.Categorical(probs).entropy().item()
        entropies.append(entropy)
    
    entropies = np.array(entropies)
    
    # Get best and worst performers
    sorted_indices = np.argsort(kl_divs)
    best_indices = sorted_indices[:n_best]
    worst_indices = sorted_indices[-n_worst:]
    
    print(f"\n{model_name} Detailed Outlier Analysis:")
    print(f"=" * 60)
    
    # Show best performers (low KL)
    print(f"\nBEST PERFORMERS (lowest KL divergence):")
    for i, idx in enumerate(best_indices):
        seq = sequences[idx]
        context = seq[:-1]
        target = seq[-1]
        bayesian = bayesian_dists[idx]
        model_pred = model_predictions[idx]
        kl_val = kl_divs[idx]
        entropy = entropies[idx]
        
        print(f"\n  {i+1}. Sequence #{idx}: KL = {kl_val:.4f}, Entropy = {entropy:.3f}")
        print(f"     Context: {context.tolist()} → Target: {target.item()}")
        print(f"     Bayesian: [{', '.join([f'{x:.3f}' for x in bayesian])}]")
        print(f"     Model:    [{', '.join([f'{x:.3f}' for x in model_pred])}]")
    
    # Show worst performers (high KL)
    print(f"\nWORST PERFORMERS (highest KL divergence):")
    for i, idx in enumerate(worst_indices):
        seq = sequences[idx]
        context = seq[:-1]
        target = seq[-1]
        bayesian = bayesian_dists[idx]
        model_pred = model_predictions[idx]
        kl_val = kl_divs[idx]
        entropy = entropies[idx]
        
        print(f"\n  {i+1}. Sequence #{idx}: KL = {kl_val:.4f}, Entropy = {entropy:.3f}")
        print(f"     Context: {context.tolist()} → Target: {target.item()}")
        print(f"     Bayesian: [{', '.join([f'{x:.3f}' for x in bayesian])}]")
        print(f"     Model:    [{', '.join([f'{x:.3f}' for x in model_pred])}]")


def show_example_sequences(indices: np.ndarray,
                          sequences: torch.Tensor,
                          bayesian_dists: torch.Tensor,
                          kl_divs: np.ndarray,
                          group_name: str,
                          n_examples: int = 3) -> None:
    """Legacy function - now does nothing to reduce clutter."""
    pass