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
from evaluation import compare_models
from dataloader import create_dataloaders
from utils import load_model_config
from generate_urns import generate_urns
import torch.nn.functional as F


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
            def kl_div(p, q, eps=1e-10):
                """KL(p || q) where p is true, q is predicted"""
                p = p + eps
                q = q + eps
                p = p / p.sum()
                q = q / q.sum()
                return torch.sum(p * torch.log(p / q)).item()

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

    if improvement > 0.1:
        print("→ LEXICAL INVARIANCE SIGNIFICANTLY IMPROVES GENERALIZATION")
    elif improvement < -0.1:
        print("→ LEXICAL INVARIANCE HURTS GENERALIZATION")
    else:
        print("→ LEXICAL INVARIANCE HAS MIXED EFFECT")

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


def run_lexurn_experiment(config_path):
    """
    Run complete Lexurn experiment from config file.

    Args:
        config_path: Path to configuration file

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

    print("=" * 80)
    print("LEXURN EXPERIMENT: LEXICAL INVARIANCE IN TRANSFORMER LEARNING")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
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

    # Create data loaders (same test set for fair comparison)
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

    # ========== TRAIN NORMAL MODEL ==========
    print("PHASE 1: Training Normal Model")
    print("-" * 40)

    normal_trainer = LexurnTrainer(config_path=config_path, lex=False)
    normal_trainer.train(train_loader, test_loader, num_epochs=num_epochs)

    if save_results:
        os.makedirs("checkpoints", exist_ok=True)
        normal_model_path = f"checkpoints/normal_{experiment_name}_{timestamp}.pt"
        normal_trainer.save_model(normal_model_path)

    # ========== TRAIN LEXICAL INVARIANCE MODEL ==========
    print("\nPHASE 2: Training Lexical Invariance Model")
    print("-" * 40)

    lex_trainer = LexurnTrainer(config_path=config_path, lex=True)
    lex_trainer.train(train_loader, test_loader, num_epochs=num_epochs)

    if save_results:
        lex_model_path = f"checkpoints/lexical_{experiment_name}_{timestamp}.pt"
        lex_trainer.save_model(lex_model_path)

    # ========== EVALUATION ==========
    print("\nPHASE 3: Model Evaluation")
    print("-" * 40)

    # Compare models on test set
    normal_results, lex_results = compare_models(
        normal_trainer.model,
        lex_trainer.model,
        test_dataset,
        device=normal_trainer.device,
    )

    rel_div_diff = lex_results["rel_div_mean"] - normal_results["rel_div_mean"]

    # ========== NEW TASK GENERALIZATION TEST ==========
    print("\nPHASE 4: New Task Generalization Test")
    print("-" * 40)

    generalization_results = test_new_task_generalization(
        normal_trainer.model,
        lex_trainer.model,
        config,
        device=normal_trainer.device,
        n_test_tasks=10,
    )

    # Save detailed results
    results = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "config": config,
        "n_tasks": n_tasks,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "num_epochs": num_epochs,
        "normal_trainer": {
            "train_losses": normal_trainer.train_losses,
            "eval_losses": normal_trainer.eval_losses,
            "final_loss": normal_trainer.train_losses[-1],
        },
        "lex_trainer": {
            "train_losses": lex_trainer.train_losses,
            "eval_losses": lex_trainer.eval_losses,
            "final_loss": lex_trainer.train_losses[-1],
        },
        "normal_evaluation": normal_results,
        "lex_evaluation": lex_results,
        "rel_div_difference": rel_div_diff,
        "generalization_test": generalization_results,
    }

    if save_results:
        os.makedirs("results", exist_ok=True)
        results_path = f"results/{experiment_name}_{timestamp}.pt"
        torch.save(results, results_path)
        print(f"Results saved to: {results_path}")

    return results


def main():
    """Run experiment from specified config file."""
    config_path = "configs/single_task.config"
    # config_path = "configs/multi_task.config"
    run_lexurn_experiment(config_path)


if __name__ == "__main__":
    main()
