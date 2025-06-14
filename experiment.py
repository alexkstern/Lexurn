#!/usr/bin/env python3
"""
Complete Lexurn experiment: Train both normal and lexical invariance models,
then evaluate their generalization vs memorization behavior.
"""

import torch
import os
from datetime import datetime

from train import LexurnTrainer
from evaluation import compare_models
from dataset import create_dataloaders


def run_lexurn_experiment(
    config_path="configs/dummy.config",
    train_steps=2000,
    test_steps=256,
    num_epochs=10,
    save_results=True,
):
    """
    Run complete Lexurn experiment comparing normal vs lexical invariance models.

    Args:
        config_path: Path to configuration file
        train_steps: Number of training samples
        test_steps: Number of test samples for evaluation
        num_epochs: Training epochs
        save_results: Whether to save model checkpoints

    Returns:
        results: Dictionary with training and evaluation results
    """

    print("=" * 80)
    print("LEXURN EXPERIMENT: LEXICAL INVARIANCE IN TRANSFORMER LEARNING")
    print("=" * 80)
    print(f"Training samples: {train_steps}")
    print(f"Test samples: {test_steps}")
    print(f"Training epochs: {num_epochs}")
    print()

    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create data loaders (same test set for fair comparison)
    print("Creating datasets...")
    train_loader, test_loader, test_dataset, config = create_dataloaders(
        config_path=config_path, train_steps=train_steps, test_steps=test_steps
    )

    print("Dataset info:")
    print(f"  Tasks (urns): {test_dataset.n_tasks}")
    print(f"  Context length: {test_dataset.context_len}")
    print(f"  Vocabulary size: {test_dataset.vocab_size}")
    if test_dataset.n_tasks <= 5:
        print("  Urn distributions:")
        for i, urn in enumerate(test_dataset.urns):
            print(f"    Urn {i}: {urn.numpy()}")
    print()

    # ========== TRAIN NORMAL MODEL ==========
    print("PHASE 1: Training Normal Model")
    print("-" * 40)

    normal_trainer = LexurnTrainer(config_path=config_path, lex=False)
    normal_trainer.train(train_loader, test_loader, num_epochs=num_epochs)

    if save_results:
        os.makedirs("checkpoints", exist_ok=True)
        normal_model_path = f"checkpoints/normal_{timestamp}.pt"
        normal_trainer.save_model(normal_model_path)

    # ========== TRAIN LEXICAL INVARIANCE MODEL ==========
    print("\nPHASE 2: Training Lexical Invariance Model")
    print("-" * 40)

    lex_trainer = LexurnTrainer(config_path=config_path, lex=True)
    lex_trainer.train(train_loader, test_loader, num_epochs=num_epochs)

    if save_results:
        lex_model_path = f"checkpoints/lexical_{timestamp}.pt"
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

    # ========== RESULTS SUMMARY ==========
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Training performance
    print("Training Performance:")
    print(f"  Normal model final loss: {normal_trainer.train_losses[-1]:.4f}")
    print(f"  Lexical model final loss: {lex_trainer.train_losses[-1]:.4f}")

    print("\nGeneralization Analysis:")
    print(
        f"  Normal model relative divergence: {normal_results['rel_div_mean']:.4f} ± {normal_results['rel_div_std']:.4f}"
    )
    print(
        f"  Lexical model relative divergence: {lex_results['rel_div_mean']:.4f} ± {lex_results['rel_div_std']:.4f}"
    )

    # Hypothesis test
    rel_div_diff = lex_results["rel_div_mean"] - normal_results["rel_div_mean"]
    print("\nHypothesis Test:")
    print(f"  Relative divergence difference: {rel_div_diff:.4f}")

    if rel_div_diff < -0.1:
        result_interpretation = (
            "✅ HYPOTHESIS CONFIRMED: Lexical invariance improves generalization"
        )
        success = True
    elif rel_div_diff > 0.1:
        result_interpretation = (
            "❌ HYPOTHESIS REJECTED: Lexical invariance worsens generalization"
        )
        success = False
    else:
        result_interpretation = (
            "⚠️  HYPOTHESIS UNCLEAR: Mixed or no effect from lexical invariance"
        )
        success = None

    print(f"  {result_interpretation}")

    # Interpretation
    print("\nInterpretation:")
    if normal_results["rel_div_mean"] > 0.6:
        print("  Normal model: MEMORIZING (high relative divergence)")
    elif normal_results["rel_div_mean"] < 0.4:
        print("  Normal model: GENERALIZING (low relative divergence)")
    else:
        print("  Normal model: MIXED behavior")

    if lex_results["rel_div_mean"] > 0.6:
        print("  Lexical model: MEMORIZING (high relative divergence)")
    elif lex_results["rel_div_mean"] < 0.4:
        print("  Lexical model: GENERALIZING (low relative divergence)")
    else:
        print("  Lexical model: MIXED behavior")

    # Save detailed results
    results = {
        "timestamp": timestamp,
        "config": config,
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
        "hypothesis_result": result_interpretation,
        "success": success,
    }

    if save_results:
        results_path = f"results_{timestamp}.pt"
        torch.save(results, results_path)
        print(f"\nResults saved to: {results_path}")

    print("=" * 80)

    return results


def quick_test():
    """Run a quick test with minimal data for debugging."""
    print("Running quick test...")

    results = run_lexurn_experiment(
        train_steps=500, test_steps=128, num_epochs=3, save_results=False
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Lexurn experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--train-steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--test-steps", type=int, default=256, help="Test steps")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "--config", type=str, default="configs/dummy.config", help="Config file"
    )

    args = parser.parse_args()

    if args.quick:
        results = quick_test()
    else:
        results = run_lexurn_experiment(
            config_path=args.config,
            train_steps=args.train_steps,
            test_steps=args.test_steps,
            num_epochs=args.epochs,
        )

    print("Experiment complete!")
