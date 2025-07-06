import torch


def generate_urns(
    n_tasks: int = 10,
    n_colors: int = 4,
    alpha: float = 1.0,
    *,
    device: str | torch.device = "cuda",   # default to GPU
    seed: int | None = None,
) -> torch.Tensor:                         # (n_tasks, n_colors)
    if seed is not None:
        torch.manual_seed(seed)
    alpha_vec = torch.full((n_colors,), float(alpha), device=device)
    return torch.distributions.Dirichlet(alpha_vec).sample((n_tasks,))


@torch.inference_mode()                   # no autograd bookkeeping
def generate_dataset(
    context_len: int = 8,
    n_tasks: int = 32,
    n_colors: int = 4,
    n_steps: int = 10_000,
    alpha: float = 1.0,
    seed: int | None = None,
    device: str | torch.device = "cuda",
):
    """
    Vectorised GPU implementation.

    Returns
    -------
    dataset  : (n_steps, context_len)   int64
    urns     : (n_tasks, n_colors)      float32
    task_ids : (n_steps,)               int64
    """
    if seed is not None:
        torch.manual_seed(seed)

    # 1. Generate the urns (Dirichlet over colours)
    urns = generate_urns(
        n_tasks=n_tasks,
        n_colors=n_colors,
        alpha=alpha,
        device=device,
        seed=None,        # already handled
    )

    # 2. Pick an urn for every step in one go
    task_ids = torch.randint(0, n_tasks, (n_steps,), device=device)

    # 3. Pull the corresponding probability vectors (broadcasted indexing)
    cat_probs = urns[task_ids]            # (n_steps, n_colors)

    # 4. Sample context_len tokens per row in one vectorised call
    #    torch.multinomial works row-wise when input is 2-D.
    dataset = torch.multinomial(
        cat_probs,
        num_samples=context_len,
        replacement=True,                 # with replacement per token
    )                                      # (n_steps, context_len)

    return dataset, urns, task_ids


# quick smoke-test on GPU
if __name__ == "__main__":
    dataset, urns, task_ids = generate_dataset(
        context_len=8,
        n_tasks=4,
        n_colors=4,
        n_steps=100,
        alpha=1.0,
        seed=42,
        device="cuda",
    )
    print(dataset.shape, urns.shape, task_ids.shape)
