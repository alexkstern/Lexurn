import torch
from torch.utils.data import Dataset, DataLoader
from build_dataset import generate_dataset

class UrnDataset(Dataset):
    """
    PyTorch Dataset for urn sequences with causal language modeling.
    Returns raw sequences - training loop handles input/target shifting.
    """
    
    def __init__(self, context_len=8, n_tasks=1, n_colors=4, n_steps=1000, alpha=1.0, seed=42):
        """
        Args:
            context_len: Length of sequences
            n_tasks: Number of different urns/distributions  
            n_colors: Number of token types (vocab size)
            n_steps: Number of sequences to generate
            alpha: Dirichlet concentration parameter
            seed: Random seed
        """
        self.n_tasks = n_tasks
        self.n_steps = n_steps
        self.context_len = context_len
        self.vocab_size = n_colors
        self.seed = seed
        
        # Generate the dataset
        self.sequences, self.urns, self.task_ids = generate_dataset(
            context_len=self.context_len,
            n_tasks=self.n_tasks,
            n_colors=self.vocab_size,
            n_steps=self.n_steps,
            alpha=1.0,
            seed=self.seed
        )
        
        # Store for evaluation
        self.task_ids = self.task_ids.squeeze()  # Remove extra dimension
        
    def __len__(self):
        return self.n_steps
    
    def __getitem__(self, idx):
        """
        Returns raw sequence for causal language modeling.
        
        Returns:
            sequence: Full sequence of shape (context_len,)
        """
        return self.sequences[idx]
    
if __name__ == "__main__":
    # Test dataset creation
    dataset = UrnDataset(context_len=8, n_tasks=2, n_colors=4, n_steps=10, seed=42)
    print(f"Dataset length: {len(dataset)}")
    print(f"First sequence: {dataset[0]}")
    print(f"Urns shape: {dataset.urns.shape}")
    print(f"Task IDs shape: {dataset.task_ids.shape}")