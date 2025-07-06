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


"""
# Original slow version - kept for reference
def bayesian_memorizing_predictor_slow(sequence, training_urns, eps=1e-8):
    
    Compute Bayesian memorizing predictor for next token given a sequence.
    
    This implements the steps you described:
    1. For each training task (urn), compute likelihood of the sequence
    2. Normalize likelihoods to get posterior probabilities over tasks  
    3. Compute weighted average of each task's next-token prediction
    
    Args:
        sequence: Input sequence tensor of shape (seq_len,) with token indices
        training_urns: Training task distributions of shape (n_tasks, vocab_size)
        eps: Small epsilon for numerical stability
        
    Returns:
        torch.Tensor: Predicted next-token distribution of shape (vocab_size,)
    
    n_tasks, vocab_size = training_urns.shape
    device = training_urns.device
    
    # If only one training urn, simply return it
    if n_tasks == 1:
        return training_urns[0]
    
    # Compute likelihood of sequence under each task
    likelihoods = torch.ones(n_tasks, device=device)
    
    for task_idx in range(n_tasks):
        task_dist = training_urns[task_idx]  # (vocab_size,)
        
        # Compute likelihood of sequence under this task
        likelihood = 1.0
        for token in sequence:
            token_prob = torch.clamp(task_dist[token], min=eps)
            likelihood *= token_prob
        
        likelihoods[task_idx] = likelihood
    
    # Normalize to get posterior probabilities (Bayes rule with uniform prior)
    likelihood_sum = torch.clamp(likelihoods.sum(), min=eps)
    posterior_weights = likelihoods / likelihood_sum
    
    # Compute weighted average prediction for next token
    # Example: if posterior weights are [0.5, 0.3, 0.2] and task distributions are:
    # Task 0: [0.1, 0.2, 0.3, 0.4], Task 1: [0.4, 0.3, 0.2, 0.1], Task 2: [0.25, 0.25, 0.25, 0.25]
    # Result: 0.5*[0.1,0.2,0.3,0.4] + 0.3*[0.4,0.3,0.2,0.1] + 0.2*[0.25,0.25,0.25,0.25]
    next_token_pred = torch.zeros(vocab_size, device=device)
    for task_idx in range(n_tasks):
        next_token_pred += posterior_weights[task_idx] * training_urns[task_idx]
    
    return next_token_pred
"""

def bayesian_memorizing_predictor(sequence, training_urns, eps=1e-8):
    """
    Compute Bayesian memorizing predictor for next token given a sequence.
    
    Optimized version using vectorized operations and log-space computation.
    
    Args:
        sequence: Input sequence tensor of shape (seq_len,) with token indices
        training_urns: Training task distributions of shape (n_tasks, vocab_size)
        eps: Small epsilon for numerical stability
        
    Returns:
        torch.Tensor: Predicted next-token distribution of shape (vocab_size,)
    """
    n_tasks, vocab_size = training_urns.shape
    device = training_urns.device
    
    # If only one training urn, simply return it
    if n_tasks == 1:
        return training_urns[0]
    
    # Clamp training urns to avoid log(0)
    training_urns_clamped = torch.clamp(training_urns, min=eps)
    
    # Compute log-likelihood of sequence under each task (vectorized)
    # Shape: (n_tasks, seq_len)
    log_token_probs = torch.log(training_urns_clamped)[:, sequence]
    
    # Sum log probabilities for each task to get log-likelihood
    # Shape: (n_tasks,)
    log_likelihoods = log_token_probs.sum(dim=1)
    
    # Convert back to likelihood space using log-sum-exp trick for numerical stability
    max_log_likelihood = log_likelihoods.max()
    normalized_log_likelihoods = log_likelihoods - max_log_likelihood
    likelihoods = torch.exp(normalized_log_likelihoods)
    
    # Normalize to get posterior probabilities (Bayes rule with uniform prior)
    likelihood_sum = torch.clamp(likelihoods.sum(), min=eps)
    posterior_weights = likelihoods / likelihood_sum
    
    # Compute weighted average prediction for next token (vectorized)
    # Shape: (vocab_size,)
    next_token_pred = torch.matmul(posterior_weights, training_urns)
    
    return next_token_pred


def calculate_in_context_distribution(sequence, vocab_size=4):
    """
    Calculate Bayesian posterior distribution with uniform Dirichlet prior from context tokens.
    
    With uniform Dirichlet prior (alpha=1 for all tokens), the posterior is:
    posterior(token) = (count(token) + 1) / (total_count + vocab_size)
    
    Args:
        sequence: tensor of shape (context_len,) containing token ids
        vocab_size: number of possible tokens
    
    Returns:
        distribution: tensor of shape (vocab_size,) with probabilities
    """
    context = sequence[:-1]  # Exclude last token (prediction target)
    
    # Count occurrences of each token
    counts = torch.zeros(vocab_size)
    for token in context:
        counts[token] += 1
    
    # Bayesian posterior with uniform Dirichlet prior (alpha=1)
    # posterior = (counts + alpha) / (total + alpha * vocab_size)
    # With alpha=1: posterior = (counts + 1) / (total + vocab_size)
    total = counts.sum()
    distribution = (counts + 1) / (total + vocab_size)
    
    return distribution


if __name__ == "__main__":
    print("Testing evaluation_functions.py...")
    
    # Test kl_div
    print("\n1. Testing kl_div...")
    p = torch.tensor([[0.5, 0.3, 0.2], [0.8, 0.1, 0.1]])
    q = torch.tensor([[0.4, 0.4, 0.2], [0.7, 0.2, 0.1]]) 
    kl_result = kl_div(p, q)
    print(f"KL divergence result shape: {kl_result.shape}")
    print(f"KL divergence values: {kl_result}")
    assert kl_result.shape == (2,), "KL divergence should return one value per row"
    assert torch.all(kl_result >= 0), "KL divergence should be non-negative"
    print("✓ kl_div passed")
    
    # Test symmetrized_kl_div
    print("\n2. Testing symmetrized_kl_div...")
    sym_kl = symmetrized_kl_div(p, q)
    print(f"Symmetric KL divergence: {sym_kl}")
    assert sym_kl.shape == (2,), "Symmetric KL should return one value per row"
    assert torch.all(sym_kl >= 0), "Symmetric KL should be non-negative"
    # Check symmetry property
    sym_kl_reverse = symmetrized_kl_div(q, p)
    assert torch.allclose(sym_kl, sym_kl_reverse, atol=1e-6), "Symmetric KL should be symmetric"
    print("✓ symmetrized_kl_div passed")
    
    # Test r (relative distance)
    print("\n3. Testing r (relative distance)...")
    h = torch.tensor([[0.5, 0.3, 0.2]])  # model prediction
    g = torch.tensor([[0.6, 0.2, 0.2]])  # generalizing predictor
    m = torch.tensor([[0.3, 0.5, 0.2]])  # memorizing predictor
    r_result = r(h, g, m)
    print(f"Relative distance r: {r_result}")
    assert r_result.shape == (1,), "r should return one value per batch element"
    # r should be bounded roughly between -1 and 1 for reasonable inputs
    print("✓ r passed")
    
    # Test d_rel
    print("\n4. Testing d_rel...")
    d_rel_result = d_rel(r_result)
    print(f"d_rel result: {d_rel_result}")
    assert torch.all(d_rel_result >= 0) and torch.all(d_rel_result <= 1), "d_rel should be in [0,1]"
    # Test edge cases
    edge_r = torch.tensor([-1.0, 0.0, 1.0])
    edge_d_rel = d_rel(edge_r)
    expected = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(edge_d_rel, expected), "d_rel edge cases failed"
    print("✓ d_rel passed")
    
    # Test entropy
    print("\n5. Testing entropy...")
    uniform_dist = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    deterministic_dist = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    H_uniform = entropy(uniform_dist)
    H_deterministic = entropy(deterministic_dist)
    print(f"Uniform entropy: {H_uniform}")
    print(f"Deterministic entropy: {H_deterministic}")
    assert H_uniform > H_deterministic, "Uniform distribution should have higher entropy"
    # Test bits=True
    H_uniform_bits = entropy(uniform_dist, bits=True)
    print(f"Uniform entropy in bits: {H_uniform_bits}")
    assert torch.allclose(H_uniform_bits, torch.log2(torch.tensor(4.0)), atol=1e-5), "Uniform 4-way entropy should be 2 bits"
    print("✓ entropy passed")
    
    # Test bayesian_memorizing_predictor
    print("\n6. Testing bayesian_memorizing_predictor...")
    # Create simple test case
    training_urns = torch.tensor([
        [0.8, 0.1, 0.05, 0.05],  # Urn favoring token 0
        [0.1, 0.8, 0.05, 0.05],  # Urn favoring token 1  
        [0.05, 0.05, 0.8, 0.1]   # Urn favoring token 2
    ])
    sequence = torch.tensor([0, 0, 0])  # Sequence of mostly 0s
    pred = bayesian_memorizing_predictor(sequence, training_urns)
    print(f"Prediction for sequence [0,0,0]: {pred}")
    assert pred.shape == (4,), "Prediction should have vocab_size elements"
    assert torch.allclose(pred.sum(), torch.tensor(1.0), atol=1e-5), "Prediction should sum to 1"
    assert pred[0] > pred[1], "Should favor token 0 given sequence of 0s"
    
    # Test single urn case
    single_urn = training_urns[:1]  # Just first urn
    pred_single = bayesian_memorizing_predictor(sequence, single_urn)
    assert torch.allclose(pred_single, single_urn[0]), "Single urn should return that urn"
    print("✓ bayesian_memorizing_predictor passed")
    
    # Test calculate_in_context_distribution
    print("\n7. Testing calculate_in_context_distribution...")
    test_sequence = torch.tensor([0, 0, 1, 2, 3])  # Last token is target
    context_dist = calculate_in_context_distribution(test_sequence, vocab_size=4)
    print(f"Context distribution: {context_dist}")
    assert context_dist.shape == (4,), "Should return vocab_size distribution"
    assert torch.allclose(context_dist.sum(), torch.tensor(1.0), atol=1e-5), "Should sum to 1"
    # With sequence [0,0,1,2] (excluding target), counts are [2,1,1,0]
    # With alpha=1: posterior = ([2,1,1,0] + 1) / (4 + 4) = [3,2,2,1] / 8
    expected = torch.tensor([3/8, 2/8, 2/8, 1/8])
    assert torch.allclose(context_dist, expected), "Incorrect Bayesian posterior calculation"
    print("✓ calculate_in_context_distribution passed")
    
    print("\n✅ All tests passed!")
    
    # Original example code
    print("\nOriginal example:")
    from generate_urns import generate_urns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = 20
    dim = 4
    example = generate_urns(n_tasks=D, n_colors=dim, device=device)
    print(example)
