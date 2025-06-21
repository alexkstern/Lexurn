import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from inference import get_device, load_checkpoint
from eval_generalization import load_model, calculate_in_context_distribution, generate_test_sequences
from generate_urns import generate_urns
from utils import kl_div, load_model_config
from outlier_analysis import analyze_outliers, compare_outlier_groups, show_example_sequences, show_detailed_outliers

def compare_models_at_context_length(paths_list_models, list_config_paths=None, list_context_lengths=[10,15], n_tasks=10, num_samples_per_urn=25, device="cpu", analyze_outliers_flag=False):
    """
    Compare multiple models at specified context lengths.
    
    Args:
        paths_list_models: List of checkpoint paths for models to compare
        list_config_paths: List of config paths (must match model paths if provided)
        list_context_lengths: List of context lengths to evaluate at
        n_tasks: Number of test urns to generate
        num_samples_per_urn: Number of sequences to generate per urn
        device: Device to run on
        analyze_outliers_flag: Whether to perform detailed outlier analysis
    """
    # Assertions to ensure configs match model paths if provided
    if list_config_paths is not None:
        assert len(list_config_paths) == len(paths_list_models), \
            f"Number of config paths ({len(list_config_paths)}) must match number of model paths ({len(paths_list_models)})"
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load models
    models = []
    model_infos = []
    
    print(f"Loading {len(paths_list_models)} models...")
    for i, model_path in enumerate(paths_list_models):
        if list_config_paths is not None:
            # Use provided config paths
            config_path = list_config_paths[i]
            model, model_params, is_lexical = load_model(model_path, config_path, device)
        else:
            # Load checkpoint and extract config from it
            checkpoint = load_checkpoint(model_path, device)
            if 'config' not in checkpoint:
                raise ValueError(f"No config found in checkpoint {model_path}. Please provide config paths.")
            
            # Create model from checkpoint config
            config = checkpoint['config']
            model_params = config.get('model', config)  # Handle different config structures
            is_lexical = 'lexical' in model_path.lower() or model_params.get('lex', False)
            model_params['lex'] = is_lexical
            
            from model import UrnTransformerDecoder
            model = UrnTransformerDecoder(**model_params).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        
        models.append(model)
        model_infos.append({
            'path': model_path,
            'params': model_params,
            'is_lexical': is_lexical,
            'name': f"Model{i+1} ({'lexical' if is_lexical else 'normal'})"
        })
        print(f"  Loaded {model_infos[-1]['name']} from {model_path}")
    
    # Generate test data once for all context lengths
    print(f"\nGenerating {n_tasks} test urns...")
    test_urns = generate_urns(
        n_tasks=n_tasks,
        n_colors=4,
        alpha=1.0,
        device=device
    )
    test_urns = torch.tensor(test_urns, dtype=torch.float32, device=device)
    
    # Evaluate at each context length
    for context_len in list_context_lengths:
        print(f"\n{'='*60}")
        print(f"EVALUATING AT CONTEXT LENGTH {context_len}")
        print(f"{'='*60}")
        
        # Generate test sequences for this context length
        sequences, urn_indices = generate_test_sequences(
            test_urns, 
            context_len=context_len, 
            num_samples_per_urn=num_samples_per_urn,
            device=device
        )
        
        # Calculate Bayesian posterior distributions for each sequence
        print("Calculating Bayesian reference distributions...")
        bayesian_dists = []
        for seq in sequences:
            dist = calculate_in_context_distribution(seq, vocab_size=4)
            bayesian_dists.append(dist)
        bayesian_dists = torch.stack(bayesian_dists).to(device)
        
        # Evaluate each model
        results = {}
        for model, model_info in zip(models, model_infos):
            print(f"\nEvaluating {model_info['name']}...")
            kl_divs = []
            model_predictions = []
            
            with torch.no_grad():
                for i, seq in enumerate(tqdm(sequences, desc=f"Processing {model_info['name']}")):
                    # Get model predictions for the last token
                    seq_input = seq.unsqueeze(0)  # Add batch dimension
                    logits = model(seq_input)  # Shape: (1, seq_len, vocab_size)
                    
                    # Get logits for the last position (prediction target)
                    last_logits = logits[0, -1, :]  # Shape: (vocab_size,)
                    model_dist = F.softmax(last_logits, dim=-1)
                    model_predictions.append(model_dist)
                    
                    # Calculate KL divergence vs Bayesian posterior
                    kl_div_val = kl_div(model_dist, bayesian_dists[i])
                    kl_divs.append(kl_div_val)
            
            # Calculate statistics
            kl_divs = np.array(kl_divs)
            model_predictions = torch.stack(model_predictions)
            
            results[model_info['name']] = {
                'kl_divs': kl_divs,
                'mean_kl': np.mean(kl_divs),
                'std_kl': np.std(kl_divs),
                'median_kl': np.median(kl_divs),
                'model_info': model_info,
                'model_predictions': model_predictions
            }
            
            print(f"  Results for {model_info['name']}:")
            print(f"    Mean KL:   {results[model_info['name']]['mean_kl']:.4f}")
            print(f"    Std KL:    {results[model_info['name']]['std_kl']:.4f}")
            print(f"    Median KL: {results[model_info['name']]['median_kl']:.4f}")
            
            # Perform outlier analysis only if flag is set
            if analyze_outliers_flag:
                outlier_results = analyze_outliers(
                    kl_divs, sequences, urn_indices, test_urns, bayesian_dists, 
                    model_info['name']
                )
                results[model_info['name']]['outlier_analysis'] = outlier_results
                
                # Show detailed outlier examples
                show_detailed_outliers(
                    kl_divs, sequences, bayesian_dists, model_predictions,
                    model_info['name'], n_best=3, n_worst=3
                )
        
        # Print comparison summary
        print(f"\n{'='*40}")
        print("COMPARISON SUMMARY")
        print(f"{'='*40}")
        
        # Sort by mean KL (lower is better)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_kl'])
        
        print("Ranking (lower KL divergence = better generalization):")
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            print(f"  {rank}. {model_name}: {result['mean_kl']:.4f} ± {result['std_kl']:.4f}")
        
        # Print two random examples of predictions (only if not doing detailed analysis)
        if not analyze_outliers_flag:
            print("\nExample predictions (first 2 sequences):")
            with torch.no_grad():
                for i in range(min(2, len(sequences))):
                    seq = sequences[i]
                    context = seq[:-1]  # Remove last token (target)
                    print(f"\nContext {i+1}: {context.tolist()}")
                    print(f"  Bayesian reference: [{', '.join([f'{x:.3f}' for x in bayesian_dists[i]])}]")
                    
                    for model, model_info in zip(models, model_infos):
                        logits = model(seq.unsqueeze(0))
                        dist = F.softmax(logits[0, -1, :], dim=-1)
                        print(f"  {model_info['name']:20}: [{', '.join([f'{x:.3f}' for x in dist])}]")





if __name__ == "__main__":
    # Model configurations and checkpoints
    seq_30_config = "configs/seq_30_4M.config"
    seq_30_checkpoint = "checkpoints/seq_30_4M/lexical_single_task_experiment_20250615_110926.pt"
    
    seq_80_config = "configs/seq_80_4M_single_task_lexical.config"
    seq_80_checkpoint = "checkpoints/seq_80_4M/lexical_single_task_lexical_20250617_012740.pt"

    seq_30_4M_single_task_10_epochs_config= "configs/seq_30_4M_single_task_10_epochs_lexical.config"
    seq_30_4M_single_task_10_epochs_checkpoint= "checkpoints/seq_30_4M_epochs_10_seed_43/lexical_single_task_lexical_20250620_220652.pt"
    
    # Context lengths to evaluate
    context_lengths = [5,10, 15,20,25,30]  # Start with smaller test
    #context_lengths = [5,10, 15,20,25,30,35,40,45,50,55,60,65,70,75,80]  # Start with smaller test
    #context_lengths = [20,40,60,80]  # Start with smaller test

    device = get_device()
    
    print("Comparing seq_30 vs seq_80 lexical models...")
    print("=" * 50)
    
    # List of models to compare
    #model_paths = [seq_30_checkpoint, seq_80_checkpoint]
    #config_paths = [seq_30_config, seq_80_config]
    #test

    model_paths = [seq_30_4M_single_task_10_epochs_checkpoint, seq_80_checkpoint]
    config_paths = [seq_30_4M_single_task_10_epochs_config, seq_80_config]
    
    compare_models_at_context_length(
        paths_list_models=model_paths,
        list_config_paths=config_paths, 
        list_context_lengths=context_lengths,
        n_tasks=10,  # Number of test urns
        num_samples_per_urn=100,  # Samples per urn
        device=device,
        analyze_outliers_flag=True  # Set to True to see detailed entropy-KL analysis
    )