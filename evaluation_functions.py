import torch

# Function that calculates KL divergence
def kl_div(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    calc KL divergence between two prob distributions p and q
    p and q may contain (1,...,n) tensors
    i.e. each row is a probability distribution

    return shape tensor (One KL value per row/ per prob dist)
    """
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p.shape = {p.shape}, q.shape = {q.shape}")

    # avoid division by zero
    p = torch.clamp(p.clone(), min=eps)
    q = torch.clamp(q.clone(), min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    return torch.sum(p * torch.log(p / q), dim=-1)


# Function that calculates symmetrized KL divergence
def symmetrized_kl_div(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Idea of symmetrized KL is to take the average between the
    KL of p and q and KL of q and p, as KL div isn't symmetric
    """
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p.shape = {p.shape}, q.shape = {q.shape}")

    kl_1 = kl_div(p, q, eps=eps)
    kl_2 = kl_div(q, p, eps=eps)

    symmetrized = (kl_1 + kl_2) * 0.5
    return symmetrized


# Function that calculates relative distance
def r(h: torch.Tensor, g: torch.Tensor, m: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the signed, normalised distance `r` between the model’s predictions `h` 
    and two reference predictors: the generalising predictor `g` and the memorising predictor `m`.

    This score quantifies how much the model resembles the generalising vs. memorising behaviour,
    relative to the gap between those two predictors.

    Definition:
        r = (D(h, G) - D(h, M)) / D(G, M)

    Where:
        - h: model prediction distribution (batch of probability vectors)
        - g: generalising predictor distribution (Bayesian posterior given uniform Dirichlet prior)
        - m: memorising predictor distribution (empirical distribution over training urns)
        - D(⋅, ⋅): symmetrised KL divergence between distributions

    Interpretation:
        - r = -1   → the model exactly matches the generalising predictor (G)
        - r = +1   → the model exactly matches the memorising predictor (M)
        - r  > 0   → the model is closer to M than to G
        - r  < 0   → the model is closer to G than to M

    Notes:
        - To ensure numerical stability, a small ε is used to clamp the denominator D(G, M) ≥ ε.
        - If G and M are nearly identical, r may become unstable or ill-defined, so clamping is critical.

    Returns:
        A tensor of shape (batch,) where each element is the r value for a single distribution in the batch.
    """
    div_h_g = symmetrized_kl_div(h, g, eps=eps)
    div_h_m = symmetrized_kl_div(h, m, eps=eps)

    num = div_h_g - div_h_m
    div_g_m = symmetrized_kl_div(g, m, eps=eps)
    den = torch.clamp(div_g_m, min=eps)

    r = num / den
    return r


# Function that calculates relative distance in [0, 1]
def d_rel(r):
    """
    Compute the relative distance d_rel from the signed distance r.

    This rescales the signed distance r ∈ [−1, 1] into the unit interval [0, 1]:

        d_rel = (r + 1) / 2

    Interpretation:
        - d_rel = 0   → model exactly matches the generalising predictor
        - d_rel = 1   → model exactly matches the memorising predictor
        - d_rel = 0.5 → model lies halfway between the two

    The output is clamped to ensure it stays within [0, 1].

    Args:
        r (Tensor): Signed relative distance tensor

    Returns:
        Tensor: Rescaled relative distance in [0, 1]
    """
    drel = (r + 1) / 2
    drel = torch.clamp(drel, min=0.0, max=1.0)
    return drel


# Function that calculates entropy (with option of log base 2 entropy)
def entropy(p: torch.Tensor, *, bits: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Shannon entropy of a batch of probability distributions.

    Args
    ----
    p   : (batch, k) Tensor
          Each row is a probability vector.  Values need not be strictly
          normalised; they will be renormalised internally.
    bits: bool (default=False)
          • False  → return entropy in nats  (log base e)
          • True   → return entropy in bits  (log base 2)
    eps : float
          Numerical floor to avoid log(0).

    Returns
    -------
    Tensor of shape (batch,) – one entropy value per row.
    """
    p = torch.clamp(p.clone(), min=eps)
    p = p / p.sum(dim=-1, keepdim=True)

    log_p = torch.log2(p) if bits else torch.log(p)
    H = -torch.sum(p * log_p, dim=-1)
    return H
