# Lexurn

Research project studying **lexical invariance in Transformer learning**. Tests whether randomizing token embeddings forces models to generalize rather than memorize.

## How It Works

**Core Architecture:**
- 4-layer Transformer decoder (256d, 4 heads, vocab_size=4) ~ 4.2 M params
- **Lexical invariance mechanism**: randomizes token embeddings per sequence when `lex=True`
- Trains on synthetic "balls and urns" sequences with causal language modeling

**Key Components:**
- `generate_urns.py` + `build_dataset.py`: Generate probability distributions and training sequences
- `dataloader.py`: PyTorch dataset for causal LM (context_len-1 tokens → predict last token)
- `model.py`: Transformer with optional lexical invariance
- `run_experiment.py`: End-to-end training pipeline
- `test_generalization.py`: Fair comparison evaluation on unseen urns

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
