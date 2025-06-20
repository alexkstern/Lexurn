#!/usr/bin/env python3
"""
Complete Lexurn experiment using only config files.
Run single-task or multi-task experiments to test lexical invariance.
"""

import torch
import numpy as np
import os
from datetime import datetime

from train import LexurnTrainer
from dataloader import create_dataloaders
from utils import load_model_config, kl_div
from generate_urns import generate_urns
import torch.nn.functional as F
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_new_task_generalization(
    normal_model, lex_model, config, device="cpu", n_test_tasks=10
):
    """
    Test both models on completely new unseen tasks.

    This tests true generalization by comparing model predictions
    against ground truth probability distributions on fresh tasks.

    Args:
        normal_model: Normal trained model
        lex_model: Lexical invariance model
        config: Experiment configuration
        device: Computation device
        n_test_tasks: Number of new tasks to test on

    Returns:
        results: Dictionary with KL divergences for both models
    """
    print("\n=== NEW TASK GENERALIZATION TEST ===")
    print(f"Testing on {n_test_tasks} completely new tasks")

    # Extract config parameters
    context_len = config["model"]["context_len"]
    vocab_size = config["model"]["vocab_size"]
    eval_context_len = context_len - 1  # Predict last token

    # Generate fresh new urns (never seen in training)
    new_urns = generate_urns(
        n_tasks=n_test_tasks, n_colors=vocab_size, alpha=1.0, device=device
    )
    print(f"Generated {n_test_tasks} new urns with {vocab_size} colors")

    normal_model.eval()
    lex_model.eval()

    normal_kl_divergences = []
    lex_kl_divergences = []

    with torch.no_grad():
        for task_idx in range(n_test_tasks):
            true_urn = new_urns[task_idx]  # Ground truth distribution

            # Sample a context sequence from this urn
            context_dist = torch.distributions.Categorical(true_urn)
            context_sequence = context_dist.sample(
                (eval_context_len,)
            )  # (context_len-1,)

            # Prepare input for models
            context_input = context_sequence.unsqueeze(0).to(
                device
            )  # (1, context_len-1)

            # Get model predictions
            normal_logits = normal_model(
                context_input
            )  # (1, context_len-1, vocab_size)
            lex_logits = lex_model(context_input)  # (1, context_len-1, vocab_size)

            # Get prediction for next token (last position)
            normal_pred = F.softmax(normal_logits[0, -1, :], dim=0)  # (vocab_size,)
            lex_pred = F.softmax(lex_logits[0, -1, :], dim=0)  # (vocab_size,)

            # Compute KL divergence against ground truth
            normal_kl = kl_div(true_urn, normal_pred)
            lex_kl = kl_div(true_urn, lex_pred)

            normal_kl_divergences.append(normal_kl)
            lex_kl_divergences.append(lex_kl)

            # Print first few examples for inspection
            if task_idx < 3:
                print(f"\nTask {task_idx}:")
                print(f"  True urn:     {true_urn.cpu().numpy().round(3)}")
                print(f"  Context:      {context_sequence.tolist()}")
                print(
                    f"  Normal pred:  {normal_pred.cpu().numpy().round(3)} (KL: {normal_kl:.4f})"
                )
                print(
                    f"  Lexical pred: {lex_pred.cpu().numpy().round(3)} (KL: {lex_kl:.4f})"
                )

    # Aggregate results
    normal_kl_mean = np.mean(normal_kl_divergences)
    normal_kl_std = np.std(normal_kl_divergences)
    lex_kl_mean = np.mean(lex_kl_divergences)
    lex_kl_std = np.std(lex_kl_divergences)

    print("\n=== NEW TASK GENERALIZATION RESULTS ===")
    print(
        f"Normal Model KL vs Ground Truth:  {normal_kl_mean:.4f} ± {normal_kl_std:.4f}"
    )
    print(f"Lexical Model KL vs Ground Truth: {lex_kl_mean:.4f} ± {lex_kl_std:.4f}")

    improvement = normal_kl_mean - lex_kl_mean
    print(f"Improvement (Normal - Lexical):   {improvement:.4f}")

    results = {
        "normal_kl_mean": normal_kl_mean,
        "normal_kl_std": normal_kl_std,
        "lex_kl_mean": lex_kl_mean,
        "lex_kl_std": lex_kl_std,
        "improvement": improvement,
        "normal_kl_values": normal_kl_divergences,
        "lex_kl_values": lex_kl_divergences,
        "n_test_tasks": n_test_tasks,
    }

    return results


def run_single_model_experiment(config_path, shared_dataloaders=None):
    """
    Run experiment for a single model type (normal or lexical).

    Args:
        config_path: Path to configuration file
        shared_dataloaders: Optional tuple of (train_loader, test_loader, test_dataset) to share between experiments

    Returns:
        results: Dictionary with training and evaluation results
    """

    # Load all settings from config
    config = load_model_config(config_path)

    # Extract settings
    train_steps = config["dataset"]["n_steps"]
    test_steps = config["evaluation"]["test_steps"]
    num_epochs = config["training"]["num_epochs"]
    n_tasks = config["dataset"]["n_tasks"]
    save_results = config["experiment"]["save_results"]
    experiment_name = config["experiment"]["config_name"]
    model_type = config["experiment"]["model_type"]  # "normal" or "lexical"
    
    is_lexical = model_type == "lexical"

    print("=" * 80)
    print(f"LEXURN EXPERIMENT: {model_type.upper()} MODEL")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Model type: {model_type}")
    print(f"Training on: {n_tasks} task(s) with {train_steps} samples")
    print(f"Testing on: {test_steps} samples")
    print(f"Training epochs: {num_epochs}")

    if n_tasks == 1:
        print(
            "→ SINGLE-TASK EXPERIMENT: Testing memorization vs generalization on one urn"
        )
    else:
        print(f"→ MULTI-TASK EXPERIMENT: Testing across {n_tasks} different urns")
    print()

    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create or use shared data loaders
    if shared_dataloaders is not None:
        print("Using shared datasets...")
        train_loader, test_loader, test_dataset = shared_dataloaders
    else:
        print("Creating datasets...")
        train_loader, test_loader, test_dataset, _ = create_dataloaders(
            config_path=config_path, train_steps=train_steps, test_steps=test_steps
        )

    print("Dataset info:")
    print(f"  Tasks (urns): {test_dataset.n_tasks}")
    print(f"  Context length: {test_dataset.context_len}")
    print(f"  Vocabulary size: {test_dataset.vocab_size}")

    if test_dataset.n_tasks <= 5:
        print("  Urn distributions:")
        for i, urn in enumerate(test_dataset.urns):
            print(f"    Urn {i}: {urn.numpy().round(3)}")
    print()

    # ========== TRAIN MODEL ==========
    print(f"PHASE 1: Training {model_type.title()} Model")
    print("-" * 40)

    # Extract training urns from dataset for saving in checkpoint
    training_urns = test_dataset.urns  # Use same urns as test dataset since they're from same generation
    
    trainer = LexurnTrainer(config_path=config_path, lex=is_lexical, training_urns=training_urns)
    trainer.train(train_loader, test_loader, num_epochs=num_epochs)

    if save_results:
        os.makedirs("checkpoints", exist_ok=True)
        model_path = f"checkpoints/{model_type}_{experiment_name}_{timestamp}.pt"
        trainer.save_model(model_path)

    # Skip evaluation - can be done post-hoc with test_generalization.py

    # Save detailed results
    results = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "model_type": model_type,
        "config": config,
        "n_tasks": n_tasks,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "num_epochs": num_epochs,
        "trainer": {
            "train_losses": trainer.train_losses,
            "eval_losses": trainer.eval_losses,
            "final_loss": trainer.train_losses[-1] if trainer.train_losses else None,
            "early_stopped": trainer.early_stopped,
            "best_eval_loss": trainer.best_eval_loss if trainer.early_stopping else None,
            "patience_counter": trainer.patience_counter if trainer.early_stopping else None,
        },
    }

    if save_results:
        os.makedirs("results", exist_ok=True)
        results_path = f"results/{model_type}_{experiment_name}_{timestamp}.pt"
        torch.save(results, results_path)
        print(f"Results saved to: {results_path}")

    # Finish wandb run
    trainer.finish_wandb()

    return results


def run_both_models_experiment(config_normal,config_lexinv):
    """
    Run both normal and lexical models sequentially on the SAME dataset for fair comparison.
    """
    print("Running both normal and lexical models on SAME dataset...")
    
    # Create shared dataset ONCE for fair comparison
    print("\n" + "=" * 100)
    print("CREATING SHARED DATASET FOR FAIR COMPARISON")
    print("=" * 100)
    
    # Use normal config to create dataset (both configs should have same dataset params anyway)
    config = load_model_config(config_normal)
    train_steps = config["dataset"]["n_steps"]
    test_steps = config["evaluation"]["test_steps"]
    
    print(f"Creating shared dataset with {train_steps} train steps, {test_steps} test steps")
    train_loader, test_loader, test_dataset, _ = create_dataloaders(
        config_path=config_normal, 
        train_steps=train_steps, 
        test_steps=test_steps
    )
    
    print(f"Shared dataset created - {test_dataset.n_tasks} task(s), {len(train_loader)} train batches")
    shared_dataloaders = (train_loader, test_loader, test_dataset)
    
    # Run normal model
    print("\n" + "=" * 100)
    print("STARTING NORMAL MODEL EXPERIMENT (using shared dataset)")
    print("=" * 100)
    normal_results = run_single_model_experiment(config_normal, shared_dataloaders)
    
    # Run lexical model with SAME dataset
    print("\n" + "=" * 100)
    print("STARTING LEXICAL MODEL EXPERIMENT (using shared dataset)")
    print("=" * 100)
    lexical_results = run_single_model_experiment(config_lexinv, shared_dataloaders)
    
    print("\n" + "=" * 100)
    print("BOTH EXPERIMENTS COMPLETED - FAIR COMPARISON ENSURED")
    print("=" * 100)
    print(f"Normal model results saved to: results/normal_{normal_results['experiment_name']}_{normal_results['timestamp']}.pt")
    print(f"Lexical model results saved to: results/lexical_{lexical_results['experiment_name']}_{lexical_results['timestamp']}.pt")
    print("Both models trained on IDENTICAL dataset for fair comparison")
    
    return normal_results, lexical_results


if __name__ == "__main__":
    # Run normal model
    #run_single_model_experiment("configs/single_task_normal.config")
    
    # Run lexical model
    #run_single_model_experiment("configs/single_task_lexical.config")
    #path_normal="configs/single_task_normal.config"
    #path_lexinv="configs/single_task_lexical.config"
    path_normal="configs/seq_30_4M_single_task_3_epochs_normal.config"
    path_lexinv="configs/seq_30_4M_single_task_3_epochs_lexical.config"
    
    
    run_both_models_experiment(path_normal,path_lexinv)

