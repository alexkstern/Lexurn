import torch
from torch.utils.data import Dataset, DataLoader
from build_dataset import generate_dataset
from utils import load_model_config

class UrnDataset(Dataset):
    """
    PyTorch Dataset for urn sequences with causal language modeling.
    Returns raw sequences - training loop handles input/target shifting.
    """
    
    def __init__(self, config_path="configs/dummy.config", n_steps=None):
        """
        Args:
            config_path: Path to config file
            n_steps: Override n_steps from config (useful for experiments)
        """
        # Load configuration
        self.config = load_model_config(config_path)
        
        # Dataset parameters from config
        dataset_config = self.config['dataset']
        model_config = self.config['model']
        
        self.n_tasks = dataset_config['n_tasks']
        self.n_steps = n_steps if n_steps is not None else dataset_config['n_steps']
        self.context_len = model_config['context_len']
        self.vocab_size = model_config['vocab_size']
        self.seed = dataset_config['seed']
        
        # Generate the dataset
        self.sequences, self.urns, self.task_ids = generate_dataset(
            context_len=self.context_len,
            n_tasks=self.n_tasks,
            n_colors=self.vocab_size,
            n_steps=self.n_steps,
            alpha=1.0,  # Default Dirichlet concentration
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
    
    def get_evaluation_context(self, idx):
        """
        Get evaluation context: first (context_len-1) tokens to predict the last token.
        
        Returns:
            context: First (context_len-1) tokens
            true_next: Last token to predict
            task_id: Which urn generated this sequence
        """
        sequence = self.sequences[idx]
        context = sequence[:-1]          # First (context_len-1) tokens
        true_next = sequence[-1]         # Last token
        task_id = self.task_ids[idx]     # Which urn was used
        
        return context, true_next, task_id

def create_dataloaders(config_path="configs/dummy.config", train_steps=None, test_steps=256):
    """
    Create train and test dataloaders from config.
    
    Args:
        config_path: Path to config file
        train_steps: Override training steps (None = use config)
        test_steps: Number of test samples (default 256 as per evaluation)
        
    Returns:
        train_loader, test_loader, test_dataset, config
    """
    config = load_model_config(config_path)
    batch_size = config['training']['batch_size']
    
    # Create training dataset
    train_dataset = UrnDataset(config_path=config_path, n_steps=train_steps)
    
    # Create test dataset (separate generation for evaluation)
    test_dataset = UrnDataset(config_path=config_path, n_steps=test_steps)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader, test_dataset, config

if __name__ == "__main__":
    # Test the dataset
    print("=== Testing UrnDataset ===")
    
    # Test with small dataset
    dataset = UrnDataset(n_steps=5)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Context length: {dataset.context_len}")
    print(f"Number of tasks: {dataset.n_tasks}")
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Urns shape: {dataset.urns.shape}")
    
    if dataset.n_tasks == 1:
        print(f"Single urn distribution: {dataset.urns[0]}")
    
    # Test samples
    print("\n=== Sample Data ===")
    for i in range(3):
        sequence = dataset[i]
        context, true_next, task_id = dataset.get_evaluation_context(i)
        
        print(f"Sample {i}:")
        print(f"  Full sequence: {sequence}")
        print(f"  Context: {context}")
        print(f"  True next: {true_next}")
        print(f"  Task ID: {task_id}")
    
    # Test dataloader
    print("\n=== Testing DataLoader ===")
    train_loader, test_loader, test_ds, config = create_dataloaders(train_steps=16, test_steps=8)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # Test one batch
    for batch_idx, sequences in enumerate(train_loader):
        print(f"Batch {batch_idx}: shape {sequences.shape}")
        print(f"First sequence: {sequences[0]}")
        break