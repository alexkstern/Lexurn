import torch


# ----------  helpers ---------------------------------------------------------
def generate_urns(
    n_tasks: int = 10,
    n_colors: int = 4,
    alpha: float = 1.0,
    *,
    device: str | torch.device = "cuda",
    seed: int | None = None,
) -> torch.Tensor:                          # (n_tasks, n_colors)
    if seed is not None:
        torch.manual_seed(seed)
    alpha_vec = torch.full((n_colors,), float(alpha), device=device)
    return torch.distributions.Dirichlet(alpha_vec).sample((n_tasks,))


# ----------  main API --------------------------------------------------------
@torch.inference_mode()                     # no autograd bookkeeping
def generate_dataset(
    context_len: int = 8,
    n_tasks: int = 32,
    n_colors: int = 4,
    n_steps: int = 10_000,
    alpha: float = 1.0,
    seed: int | None = None,
    urn_seed: int | None = None,
    sampling_seed: int | None = None,
    eval_mode: bool = False,
    device: str | torch.device = "cuda",
    train_on_one_sequence: bool = False,   # ← no asterisk, default False
):
    """
    Vectorised GPU implementation.

    Parameters
    ----------
    seed : int | None
        Legacy parameter for backward compatibility. If provided and 
        urn_seed/sampling_seed are None, used for both urn generation and sampling.
    urn_seed : int | None
        Seed specifically for urn generation (Dirichlet sampling).
    sampling_seed : int | None
        Seed specifically for sequence sampling from urns.
    train_on_one_sequence : bool, default False
        If True, a *single* sequence is sampled once and then copied
        `n_steps` times.  All entries in `task_ids` are identical.

    Returns
    -------
    dataset  : (n_steps, context_len)   int64
    urns     : (n_tasks, n_colors)      float32
    task_ids : (n_steps,)               int64
    """
    # Handle seed logic for backward compatibility
    if urn_seed is None and sampling_seed is None and seed is not None:
        # Legacy behavior: use same seed for both
        urn_seed = seed
        sampling_seed = seed
    
    # 1. Generate the urns (Dirichlet over colours)
    urns = generate_urns(
        n_tasks=n_tasks,
        n_colors=n_colors,
        alpha=alpha,
        device=device,
        seed=urn_seed,
    )
    
    # 2. Set sampling seed for sequence generation
    if sampling_seed is not None:
        torch.manual_seed(sampling_seed)

    # ------------------------------------------------------------------ #
    #  ---------- single-sequence mode ----------                        #
    # ------------------------------------------------------------------ #
    if train_on_one_sequence:
        # (a) pick one urn at random
        single_task_id = torch.randint(0, n_tasks, (), device=device, dtype=torch.long).item()

        # (b) sample ONE sequence from that urn
        cat_probs = urns[single_task_id].unsqueeze(0)            # (1, n_colors)
        single_seq = torch.multinomial(
            cat_probs,
            num_samples=context_len,
            replacement=True,
        )                                                         # (1, context_len)

        # (c) replicate it n_steps times
        dataset  = single_seq.repeat(n_steps, 1)                  # (n_steps, context_len)
        task_ids = torch.full((n_steps,),
                              single_task_id,
                              device=device,
                              dtype=torch.long)
        return dataset, urns, task_ids

    # ------------------------------------------------------------------ #
    #  ---------- fully-random path (original) ----------                #
    # ------------------------------------------------------------------ #
    # 2. Pick an urn for every step in one go
    task_ids = torch.randint(0, n_tasks, (n_steps,), device=device)

    # 3. Pull the corresponding probability vectors (broadcasted indexing)
    cat_probs = urns[task_ids]                                    # (n_steps, n_colors)

    # 4. Sample context_len tokens per row in one vectorised call
    dataset = torch.multinomial(
        cat_probs,
        num_samples=context_len,
        replacement=True,
    )                                                             # (n_steps, context_len)

    return dataset, urns, task_ids


# ----------  quick smoke-test -----------------------------------------------
if __name__ == "__main__":
    # should print (100, 8) (4, 4) (100,) and assert pass
    dataset, urns, task_ids = generate_dataset(
        context_len=8,
        n_tasks=4,
        n_colors=4,
        n_steps=100,
        alpha=1.0,
        seed=42,
        device="cuda",
        train_on_one_sequence=True,
    )
    print(dataset.shape, urns.shape, task_ids.shape)
    assert torch.all(dataset == dataset[0])      # every row identical
    assert torch.unique(task_ids).numel() == 1   # single task id

    torch.manual_seed(42)
    

    for seed in range(0,5):
        dataset, urns, task_ids = generate_dataset(
            context_len=8,
            n_tasks=1,
            n_colors=8,
            n_steps=100,
            alpha=1.0,
            seed=seed,
            device="cuda",
            train_on_one_sequence=False,
        )
        print("For seed",seed," the training urn distribution was:")
        print(urns[0].tolist())
