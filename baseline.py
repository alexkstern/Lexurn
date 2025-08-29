#!/usr/bin/env python
#compare simple linear regression to transformer doing lexurn ICL task
"""
autoreg_embedding_baseline.py
A *token‑embedding‑only* causal language model – structurally closer to your
Transformer code but pared down to the absolute minimum:

  • **E** ∈ ℝ^{V×d} – learnable token embeddings.
  • **W = E**       – tied output projection (weight sharing à la GPT‑2).
  • Hidden state at position *t* is the **mean of the prefix embeddings**
    0..t‑1 – computed with a causal mask / running cumulative sum, so the model
    never peeks at future tokens.

Thus the forward pass produces logits for *every* time‑step, we train with the
standard causal‑LM cross‑entropy over the whole sequence, and evaluation uses
the exact Bayesian in‑context learner for KL comparisons (ID & OOD).  All
printouts go straight to stdout – no W&B.

Hyper‑parameters are kept **identical** to the numbers in your original script
(n_train = 1 000 000, n_val = 1 000, etc.).  The code is self‑contained except
for `build_dataset.py` and `evaluation_functions.py`, which are already in your
repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Local helpers --------------------------------------------------------------
from build_dataset import generate_dataset
from evaluation_functions import calculate_in_context_distribution, symmetrized_kl_div


# Cache management functions -------------------------------------------------
def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clear_all_cache():
    """Clear all caches including GPU and system caches."""
    clear_gpu_cache()
    # Clear any other PyTorch caches
    torch.backends.cuda.cufft_plan_cache.clear()
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True


class PrefixMeanCausalModel(nn.Module):
    """Causal *bag‑of‑embeddings* with tied output projection."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L)
        # Embed tokens
        emb = self.token_embedding(x)                    # (B, L, d)

        # Causal aggregation – prefix mean for every position
        # Using running cumsum (O(L)) rather than explicit mask (O(L²))
        cumsum = emb.cumsum(dim=1)                       # (B, L, d)
        lengths = torch.arange(1, x.size(1) + 1,
                               device=x.device).view(1, -1, 1)
        h = cumsum / lengths                             # (B, L, d)

        # Project to vocab logits
        logits = self.output_proj(h)                     # (B, L, V)
        return logits


# ---------------- training / evaluation helpers ----------------------------
@torch.no_grad()
def eval_symkl(model, loader, vocab_size: int, device: str) -> float:
    """Symmetrised KL (model ⇄ Bayesian ICL) averaged over dataset."""
    model.eval()
    total, count = 0.0, 0
    for seq in loader:
        seq = seq.to(device)
        logits = model(seq[:, :-1])                      # (B, L‑1, V)
        probs  = torch.softmax(logits[:, -1], dim=-1)    # prediction for token L
        icl = torch.stack([
            calculate_in_context_distribution(s.cpu(), vocab_size)
            for s in seq.cpu()
        ]).to(device)
        total += symmetrized_kl_div(probs, icl).sum().item()
        count += seq.size(0)
    return total / count


def main() -> None:
    # ---------------- hyper‑parameters -----------------
    ctx_len   = 128
    V         = 4              # colours / vocab size
    n_tasks   = 1              # single urn (ID) – OOD handled separately
    n_train   = 1_000_000
    n_val     = 1_000
    d_model   = 64
    bs        = 128
    epochs    = 20
    lr        = 1e-4
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ---------------- datasets -------------------------
    train_seq, _, _ = generate_dataset(
        context_len=ctx_len, n_tasks=n_tasks, n_colors=V, n_steps=n_train, 
        device="cpu", urn_seed=0, sampling_seed=0
    )
    val_id_seq, _, _ = generate_dataset(
        context_len=ctx_len, n_tasks=n_tasks, n_colors=V, n_steps=n_val, 
        device="cpu", urn_seed=0, sampling_seed=1
    )  # same urns, different sequences
    val_ood_seq, _, _ = generate_dataset(
        context_len=ctx_len, n_tasks=250, n_colors=V, n_steps=n_val, 
        device="cpu", urn_seed=1234, sampling_seed=1234
    )  # different urns

    train_loader   = DataLoader(train_seq,   bs, shuffle=True,  drop_last=True)
    val_id_loader  = DataLoader(val_id_seq,  bs, shuffle=False)
    val_ood_loader = DataLoader(val_ood_seq, bs, shuffle=False)

    # ---------------- model & optimiser ---------------
    model = PrefixMeanCausalModel(V, d_model).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Clear cache before training
    clear_all_cache()

    # ---------------- training loop -------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for seq in tqdm(train_loader, desc=f"train‑ep{epoch}"):
            seq = seq.to(device)
            logits = model(seq[:, :-1])                  # (B, L‑1, V)
            loss = F.cross_entropy(
                logits.reshape(-1, V), seq[:, 1:].reshape(-1)
            )
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * seq.size(0)

        train_loss = running / n_train
        id_kl  = eval_symkl(model, val_id_loader,  V, device)
        ood_kl = eval_symkl(model, val_ood_loader, V, device)
        print(f"epoch {epoch:2d}/{epochs} | CE = {train_loss:.4f} | KL_ID = {id_kl:.4f} | KL_OOD = {ood_kl:.4f}")
        
        # Clear cache after each epoch to prevent memory buildup
        clear_gpu_cache()


if __name__ == "__main__":
    main()
