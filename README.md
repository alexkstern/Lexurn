# Lexurn: Lexical Invariance in Transformer Learning

This repository contains the implementation for **Experiment 1** of my thesis research on lexical invariance in transformer models through the lens of in-context learning (ICL). The work investigates whether lexically invariant training encourages generalization in a controlled synthetic environment.

## Thesis Context

This repository implements a controlled synthetic toy experiment inspired by Wurgaft et al. (2025) to test whether lexical invariant training encourages generalization in the Balls & Urns setting. Rather than relying on natural language corpora, this setup allows us to isolate the effect of lexical invariance under minimal confounding factors, simplifying analysis to a basic data distribution.

### Experimental Design

The experiment proceeds as follows:

1. **Dataset Construction**: We generate training and evaluation sequences from categorical urns sampled under a Dirichlet prior, controlling the number of training urns and separating in-distribution from out-of-distribution splits.

2. **Model Architectures**: We compare a standard decoder-only Transformer with fixed embeddings to a lexically invariant variant that resamples token embeddings independently for each sequence.

3. **Training and Evaluation**: Both models are trained on sequences from urns and then evaluated for alignment to the Bayesian posterior on both in-distribution and out-of-distribution test sets.

## Repository Structure

```
Lexurn/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── experiment.config           # Main experiment configuration
├── CLAUDE.md                   # Project instructions and context
├── .env                        # Environment variables
├── configs/                    # Alternative configurations
│   ├── small.config           # Small model configuration
│   ├── medium.config          # Medium model configuration  
│   └── large.config           # Large model configuration
├── checkpoints/               # Saved model checkpoints
├── results/                   # Experiment results
├── wandb/                     # Weights & Biases logs
└── Core Python Modules:
    ├── build_dataset.py       # Dataset generation
    ├── dataloader.py          # PyTorch data loading
    ├── model.py               # Transformer architectures
    ├── train.py               # Training loop and evaluation
    ├── evaluation_functions.py # Evaluation metrics
    ├── run_experiment.py      # Main experiment runner
    ├── wandb_utils.py         # W&B logging utilities
    └── utils.py               # General utilities
```

### Core Modules

#### Dataset Generation
- **`build_dataset.py`**: Vectorized GPU implementation for generating sequences from Dirichlet-distributed urns
  - `generate_dataset()`: Main function for creating training/test sequences
  - `generate_urns()`: Creates Dirichlet-distributed categorical distributions
  - Supports single-sequence training mode for controlled experiments

- **`dataloader.py`**: PyTorch dataset wrapper for urn sequences with causal language modeling support

#### Model Architecture  
- **`model.py`**: Contains `UrnTransformerDecoder` - the main model class
  - **Normal Mode**: Standard transformer with learned token embeddings
  - **Lexical Mode**: Resamples random embeddings per sequence using `torch.vmap`
  - Configurable depth, width, and attention heads

#### Training & Evaluation
- **`train.py`**: `LexurnTrainer` class for causal language modeling
  - Standard next-token prediction training
  - Integrated KL divergence evaluation against Bayesian ICL solution

- **`evaluation_functions.py`**: Comprehensive evaluation metrics:
  - `symmetrized_kl_div()`: Primary evaluation metric
  - `calculate_in_context_distribution()`: Bayesian posterior with uniform Dirichlet prior
  - `bayesian_memorizing_predictor()`: Bayesian inference over training tasks
  - Relative distance metrics (`r`, `d_rel`) for comparing predictors

#### Experiment Orchestration
- **`run_experiment.py`**: Main experiment runner with:
  - Multi-configuration support (small/medium/large models)
  - W&B integration for experiment tracking
  - Early stopping and checkpointing
  - Three evaluation splits: ID, OOD, and low-entropy

- **`wandb_utils.py`**: W&B table logging for prediction analysis
- **`utils.py`**: W&B API key resolution, model naming, and checkpoint management

## Google Colab Usage

This repository is designed to be imported and used as modules in Google Colab to leverage GPU acceleration:

```python
# Clone and setup
!git clone https://github.com/your-username/Lexurn.git
%cd Lexurn
!pip install -r requirements.txt

# Import and run experiments
from run_experiment import run_lexurn_experiment
from utils import resolve_wandb_key

# Run with default configuration
run_lexurn_experiment(
    config_path="experiment.config",
    use_wandb=True,
    wandb_project="lexurn-thesis",
    wandb_api_key=resolve_wandb_key(),
    train_normal=True,
    train_lexical=True
)
```

## Configuration Files

### Main Configuration (`experiment.config`)
Current settings for the primary experiment:
- **Dataset**: 1M training samples, 128 context length, 8 colors, 1 training urn
- **Model**: 256 d_model, 4 layers, 8 attention heads  
- **Training**: 5 epochs, batch size 64, 1e-4 learning rate
- **Evaluation**: 500 test samples, evaluation every 0.25 epochs

### Alternative Configurations
- **`configs/small.config`**: Smaller model (128 d_model, 4 colors) for quick experiments
- **`configs/medium.config`**: Medium-scale experiments  
- **`configs/large.config`**: Large-scale experiments

## Key Research Questions

1. **Lexical Invariance**: Can transformers learn to be independent of specific token identities while maintaining performance?

2. **Bayesian Alignment**: How well do transformer predictions approximate the optimal Bayesian ICL solution?

3. **Generalization**: What are the differences between normal and lexically invariant models in terms of in-distribution vs out-of-distribution performance?

## Evaluation Metrics

### Primary Metrics
- **Symmetrized KL Divergence**: Measures alignment between model predictions and Bayesian ICL posterior
- **Relative Distance (r)**: Quantifies position between generalizing and memorizing predictors  
- **Validation Loss**: Standard cross-entropy for early stopping

### Data Splits
- **ID (In-Distribution)**: Same task distribution as training, different sequences
- **OOD (Out-of-Distribution)**: Different task distribution from training  
- **Low-entropy**: Deterministic sequences (one token type per sequence)

## Running Experiments

### Basic Usage
```bash
python run_experiment.py
```

### Custom Configurations
```python
# Multi-scale experiment across different model sizes
for config in ["small.config", "medium.config", "large.config"]:
    for n_tasks in [1, 4, 16, 256]:
        run_lexurn_experiment(
            config_path=f"configs/{config}",
            n_tasks=n_tasks,
            train_normal=True,
            train_lexical=True
        )
```

## Dependencies

From `requirements.txt`:
- `torch>=2.0.0`
- `tqdm`
- `configparser` 
- `wandb`

## Citation

This work is inspired by:

```bibtex
@article{wurgaft2025iclstrategies,
  title   = {In-Context Learning Strategies Emerge Rationally},
  author  = {Wurgaft, Nir and Lubana, Ekdeep Singh and Park, Core Francisco and Tanaka, Hidenori and Reddy, Siddharth and Goodman, Noah D.},
  journal = {arXiv preprint arXiv:2506.17859},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.17859}
}
```

## Hardware Requirements

- CUDA-compatible GPU recommended for efficient training
- Vectorized operations optimized for GPU acceleration
- Memory requirements scale with model size and batch size

## Monitoring

The repository integrates with Weights & Biases for comprehensive tracking:
- Real-time loss and KL divergence metrics
- Prediction analysis tables comparing model vs Bayesian solutions
- Model checkpointing with best validation performance
- Hyperparameter and configuration tracking

## License

This code is part of thesis research. Please contact the author for usage permissions and collaboration opportunities.