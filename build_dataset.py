# dataset_builder.py
import torch
from generate_urns import generate_urns
from utils import load_config


def generate_dataset(
    context_len=8,
    n_tasks=32,
    n_colors=4,
    n_steps=10000,
    alpha=1.0,
    seed=None,
    device="cpu",
):
    """
    Generates a full dataset of synthetic token sequences using D urns (tasks).

    Returns:
        - dataset: Tensor of shape (n_steps, context_len), integer token IDs
        - urns: Tensor of shape (n_tasks, n_colors), probability distributions
        - task_ids: Tensor of shape (n_steps,), the urn index used per sample
    """

    if seed is not None:
        torch.manual_seed(seed)

    # Step 1: Generate D urns
    urns = generate_urns(
        n_tasks=n_tasks, n_colors=n_colors, alpha=alpha, device=device, seed=seed
    )

    # Step 2: Sample a dataset
    dataset = torch.zeros((n_steps, context_len), dtype=torch.long)
    task_ids = torch.zeros((n_steps, 1), dtype=torch.long)

    for i in range(n_steps):
        # Sample an urn index using uniform distribution
        uniform_dist = torch.distributions.Uniform(0, n_tasks)
        index_urn = int(uniform_dist.sample().item())  # must make it an int
        cat_probabilities = urns[index_urn]

        # Samples from a categorical distribution defined by the selected urn
        samples = torch.distributions.Categorical(cat_probabilities).sample(
            torch.Size([context_len])
        )
        # If cat_probabilities = [0.1, 0.3, 0.4, 0.2]
        # Categorical will sample:
        # - 0 with probability 0.1
        # - 1 with probability 0.3
        # - 2 with probability 0.4
        # - 3 with probability 0.2
        dataset[i] = samples  # Replaces the i-th row with 8 samples
        task_ids[i] = index_urn

    return dataset, urns, task_ids


if __name__ == "__main__":
    path = "configs/dummy.config"
    config = load_config(path)
    dataset_dict = config["dataset"]

    context_length = int(config["dataset"]["context"])
    d_tasks = int(config["dataset"]["d_tasks"])
    n_colors = int(config["dataset"]["n_colors"])

    dataset, urns, task_ids = generate_dataset(
        n_tasks=d_tasks, context_len=context_length, n_colors=n_colors, n_steps=32
    )

    print(dataset)
    print(urns)
    print(task_ids)
