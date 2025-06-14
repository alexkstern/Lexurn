import torch
import numpy as np
from collections import Counter
import torch.nn.functional as F
from scipy.special import logsumexp

class FrequencyPredictor:
    """
    G(x): Frequency-based predictor that counts tokens in context.
    
    This predictor uses simple frequency counting from the context
    to predict the next token distribution.
    """
    
    def __init__(self, vocab_size=4):
        self.vocab_size = vocab_size
    
    def predict(self, context):
        """
        Predict next token distribution based on frequency in context.
        
        Args:
            context: Token sequence of shape (context_len-1,) or (batch_size, context_len-1)
            
        Returns:
            probabilities: Distribution over vocab of shape (vocab_size,) or (batch_size, vocab_size)
        """
        if context.dim() == 1:
            # Single sequence
            return self._predict_single(context)
        else:
            # Batch of sequences
            batch_size = context.shape[0]
            probs = torch.zeros(batch_size, self.vocab_size)
            
            for i in range(batch_size):
                probs[i] = self._predict_single(context[i])
            
            return probs
    
    def _predict_single(self, context):
        """Predict for single context sequence."""
        # Count frequencies in context
        counts = torch.zeros(self.vocab_size)
        
        for token in context:
            counts[token.item()] += 1
        
        # Convert to probabilities (uniform if no tokens seen)
        if counts.sum() == 0:
            # No tokens in context - uniform distribution
            probs = torch.ones(self.vocab_size) / self.vocab_size
        else:
            # Normalize counts to probabilities
            probs = counts / counts.sum()
        
        return probs

class BayesianPredictor:
    """
    M(x): Bayesian inference predictor that uses training urn distributions.
    
    This predictor infers which urn most likely generated the context,
    then uses that urn's distribution to predict the next token.
    """
    
    def __init__(self, urns, vocab_size=4):
        """
        Args:
            urns: Training urn distributions of shape (n_tasks, vocab_size)
            vocab_size: Vocabulary size
        """
        self.urns = torch.tensor(urns, dtype=torch.float32)
        self.n_tasks = self.urns.shape[0]
        self.vocab_size = vocab_size
    
    def predict(self, context):
        """
        Predict next token using Bayesian inference over training urns.
        
        Args:
            context: Token sequence of shape (context_len-1,) or (batch_size, context_len-1)
            
        Returns:
            probabilities: Distribution over vocab of shape (vocab_size,) or (batch_size, vocab_size)
        """
        if context.dim() == 1:
            # Single sequence
            return self._predict_single(context)
        else:
            # Batch of sequences
            batch_size = context.shape[0]
            probs = torch.zeros(batch_size, self.vocab_size)
            
            for i in range(batch_size):
                probs[i] = self._predict_single(context[i])
            
            return probs
    
    def _predict_single(self, context):
        """Predict for single context using Bayesian inference."""
        context_len = len(context)
        
        if context_len == 0:
            # No context - uniform over urns
            return torch.mean(self.urns, dim=0)
        
        # Calculate likelihood of context under each urn
        log_likelihoods = torch.zeros(self.n_tasks)
        
        for task_idx in range(self.n_tasks):
            urn_probs = self.urns[task_idx]
            
            # Calculate log likelihood of context under this urn
            log_likelihood = 0.0
            for token in context:
                token_prob = urn_probs[token.item()]
                # Add small epsilon to avoid log(0)
                log_likelihood += torch.log(token_prob + 1e-10)
            
            log_likelihoods[task_idx] = log_likelihood
        
        # Convert to posterior probabilities (uniform prior)
        # Subtract max for numerical stability
        log_likelihoods = log_likelihoods - torch.max(log_likelihoods)
        posterior = torch.exp(log_likelihoods)
        posterior = posterior / torch.sum(posterior)
        
        # Weighted mixture of urn distributions
        prediction = torch.zeros(self.vocab_size)
        for task_idx in range(self.n_tasks):
            prediction += posterior[task_idx] * self.urns[task_idx]
        
        return prediction

class EvaluationMetrics:
    """
    Compute KL divergences and relative divergence metrics for model evaluation.
    """
    
    def __init__(self, vocab_size=4):
        self.vocab_size = vocab_size
    
    def kl_divergence(self, p, q, epsilon=1e-10):
        """
        Compute KL divergence KL(p || q) = sum(p * log(p/q))
        
        Args:
            p: True distribution
            q: Predicted distribution
            epsilon: Small value to avoid log(0)
            
        Returns:
            kl_div: KL divergence value
        """
        # Add epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon
        
        # Normalize to ensure valid probability distributions
        p = p / torch.sum(p)
        q = q / torch.sum(q)
        
        # Compute KL divergence
        kl_div = torch.sum(p * torch.log(p / q))
        
        return kl_div.item()
    
    def relative_divergence(self, model_pred, g_pred, m_pred, epsilon=1e-10):
        """
        Compute relative divergence = KL(model || G) / (KL(model || G) + KL(model || M))
        
        This measures whether the model is closer to the frequency predictor G
        (generalization, low relative divergence) or the Bayesian predictor M 
        (memorization, high relative divergence).
        
        Args:
            model_pred: Model prediction distribution
            g_pred: Frequency predictor distribution  
            m_pred: Bayesian predictor distribution
            epsilon: Small value for numerical stability
            
        Returns:
            rel_div: Relative divergence in [0, 1]
        """
        kl_g = self.kl_divergence(model_pred, g_pred, epsilon)
        kl_m = self.kl_divergence(model_pred, m_pred, epsilon)
        
        # Relative divergence
        total_div = kl_g + kl_m
        if total_div < epsilon:
            # If both divergences are near zero, return 0.5 (neutral)
            return 0.5
        
        rel_div = kl_g / total_div
        
        return rel_div
    
    def evaluate_model(self, model, test_dataset, device='cpu'):
        """
        Comprehensive evaluation of model against baseline predictors.
        
        Args:
            model: Trained Lexurn model
            test_dataset: UrnDataset for evaluation
            device: Device for computation
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        model.eval()
        
        # Initialize predictors
        g_predictor = FrequencyPredictor(vocab_size=self.vocab_size)
        m_predictor = BayesianPredictor(
            urns=test_dataset.urns.numpy(),
            vocab_size=self.vocab_size
        )
        
        # Collect predictions
        model_preds = []
        g_preds = []
        m_preds = []
        true_nexts = []
        contexts = []
        
        n_samples = len(test_dataset)
        
        with torch.no_grad():
            for idx in range(n_samples):
                # Get evaluation context
                context, true_next, task_id = test_dataset.get_evaluation_context(idx)
                
                # Model prediction
                context_input = context.unsqueeze(0).to(device)  # Add batch dim
                model_logits = model(context_input)  # (1, context_len-1, vocab_size)
                model_pred = F.softmax(model_logits[0, -1, :], dim=0)  # Last position
                
                # Baseline predictions
                g_pred = g_predictor.predict(context)
                m_pred = m_predictor.predict(context)
                
                # Store results
                model_preds.append(model_pred.cpu())
                g_preds.append(g_pred)
                m_preds.append(m_pred)
                true_nexts.append(true_next.item())
                contexts.append(context)
        
        # Convert to tensors
        model_preds = torch.stack(model_preds)
        g_preds = torch.stack(g_preds)
        m_preds = torch.stack(m_preds)
        
        # Compute metrics
        kl_g_values = []
        kl_m_values = []
        rel_div_values = []
        
        for i in range(n_samples):
            kl_g = self.kl_divergence(model_preds[i], g_preds[i])
            kl_m = self.kl_divergence(model_preds[i], m_preds[i])
            rel_div = self.relative_divergence(model_preds[i], g_preds[i], m_preds[i])
            
            kl_g_values.append(kl_g)
            kl_m_values.append(kl_m)
            rel_div_values.append(rel_div)
        
        # Aggregate results
        results = {
            'kl_g_mean': np.mean(kl_g_values),
            'kl_g_std': np.std(kl_g_values),
            'kl_m_mean': np.mean(kl_m_values),
            'kl_m_std': np.std(kl_m_values),
            'rel_div_mean': np.mean(rel_div_values),
            'rel_div_std': np.std(rel_div_values),
            'n_samples': n_samples,
            'kl_g_values': kl_g_values,
            'kl_m_values': kl_m_values,
            'rel_div_values': rel_div_values
        }
        
        return results
    
    def print_results(self, results, model_name="Model"):
        """Print evaluation results in a formatted way."""
        print(f"\n=== {model_name} Evaluation Results ===")
        print(f"Samples evaluated: {results['n_samples']}")
        print(f"KL(model || G): {results['kl_g_mean']:.4f} ± {results['kl_g_std']:.4f}")
        print(f"KL(model || M): {results['kl_m_mean']:.4f} ± {results['kl_m_std']:.4f}")
        print(f"Relative divergence: {results['rel_div_mean']:.4f} ± {results['rel_div_std']:.4f}")
        
        # Interpretation
        if results['rel_div_mean'] < 0.4:
            print("→ Model is GENERALIZING (closer to frequency predictor)")
        elif results['rel_div_mean'] > 0.6:
            print("→ Model is MEMORIZING (closer to Bayesian predictor)")
        else:
            print("→ Model behavior is MIXED")

def compare_models(normal_model, lex_model, test_dataset, device='cpu'):
    """
    Compare normal vs lexical invariance models on the same test set.
    
    Args:
        normal_model: Normal trained model
        lex_model: Lexical invariance trained model
        test_dataset: Test dataset
        device: Computation device
        
    Returns:
        normal_results, lex_results: Evaluation results for both models
    """
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    evaluator = EvaluationMetrics()
    
    # Evaluate both models
    print("Evaluating normal model...")
    normal_results = evaluator.evaluate_model(normal_model, test_dataset, device)
    
    print("Evaluating lexical invariance model...")
    lex_results = evaluator.evaluate_model(lex_model, test_dataset, device)
    
    # Print results
    evaluator.print_results(normal_results, "Normal Model")
    evaluator.print_results(lex_results, "Lexical Invariance Model")
    
    # Comparison summary
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Normal model relative divergence: {normal_results['rel_div_mean']:.4f}")
    print(f"Lexical model relative divergence: {lex_results['rel_div_mean']:.4f}")
    
    diff = lex_results['rel_div_mean'] - normal_results['rel_div_mean']
    if diff < -0.1:
        print("→ Lexical invariance IMPROVES generalization significantly")
    elif diff > 0.1:
        print("→ Lexical invariance WORSENS generalization")
    else:
        print("→ Lexical invariance has MIXED effect on generalization")
    
    return normal_results, lex_results

if __name__ == "__main__":
    # Test the evaluation components
    print("Testing evaluation components...")
    
    # Create mock data
    vocab_size = 4
    context_len = 7
    n_tasks = 3
    
    # Mock urns
    urns = np.random.dirichlet([1, 1, 1, 1], size=n_tasks)
    print(f"Mock urns:\n{urns}")
    
    # Mock context
    context = torch.randint(0, vocab_size, (context_len,))
    print(f"Mock context: {context}")
    
    # Test frequency predictor
    g_predictor = FrequencyPredictor(vocab_size)
    g_pred = g_predictor.predict(context)
    print(f"G(x) prediction: {g_pred}")
    
    # Test Bayesian predictor
    m_predictor = BayesianPredictor(urns, vocab_size)
    m_pred = m_predictor.predict(context)
    print(f"M(x) prediction: {m_pred}")
    
    # Test metrics
    evaluator = EvaluationMetrics(vocab_size)
    
    # Mock model prediction
    model_pred = torch.tensor([0.1, 0.4, 0.3, 0.2])
    
    kl_g = evaluator.kl_divergence(model_pred, g_pred)
    kl_m = evaluator.kl_divergence(model_pred, m_pred)
    rel_div = evaluator.relative_divergence(model_pred, g_pred, m_pred)
    
    print(f"KL(model || G): {kl_g:.4f}")
    print(f"KL(model || M): {kl_m:.4f}")
    print(f"Relative divergence: {rel_div:.4f}")
    
    print("\nEvaluation components working correctly!")