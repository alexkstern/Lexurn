# Lexurn

Research project studying **lexical invariance in Transformer learning**. Tests whether randomizing token embeddings forces models to generalize rather than memorize.

## Repository Structure

### Core Model & Training
- `model.py` - 4-layer Transformer decoder with lexical invariance mechanism
- `train.py` - Training loop with early stopping and model checkpointing  
- `run_experiment.py` - End-to-end experiment pipeline for both normal and lexical models

### Data Generation & Loading
- `generate_urns.py` - Creates probability distributions using Dirichlet sampling
- `build_dataset.py` - Generates synthetic "balls and urns" training sequences
- `dataloader.py` - PyTorch dataset for causal language modeling
- `configs/dummy.config` - Model and training configuration

### Evaluation & Analysis
- `eval_generalization.py` - Tests model generalization on unseen tasks
- `evaluation.py` - Utility functions for computing KL divergences
- `inference.py` - Model inference utilities
- `compare_context_lengths.py` - Analysis of context length effects
- `outlier_analysis.py` - Statistical analysis of model predictions

### Utilities & Integration
- `utils.py` - Configuration loading and helper functions
- `wandb_utils.py` - Weights & Biases experiment tracking
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (WANDB_API_KEY)

## How It Works

**Core Architecture:**
- 4-layer Transformer decoder (128d, 4 heads, vocab_size=4) 
- **Lexical invariance mechanism**: randomizes token embeddings per sequence when `lex=True`
- Trains on synthetic "balls and urns" sequences with causal language modeling

##  Results 

**Lexical invariance dramatically improves generalization:**
- Normal model: 0.54 ± 0.28 KL divergence vs in-context distribution
- Lexical model: 0.14 ± 0.11 KL divergence (**73% improvement**)
- Tested on out-of-distribution tasks

## Next Steps

- [ ] **Wandb integration** for experiment tracking and visualization
- [ ] Multi-task experiments across different urn configurations
- [ ] Ablation studies on model architecture and hyperparameters
- [ ] Scale experiments to larger vocabularies and sequence lengths
