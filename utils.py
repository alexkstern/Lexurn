import configparser
import torch


def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    config_dict = {}
    for section in config.sections():
        config_dict[section] = dict(config[section])

    return config_dict


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return total_params, trainable_params, frozen_params


def load_model_config(config_path):
    """
    Load configuration and return model initialization parameters
    """
    config = load_config(config_path)

    # Model architecture parameters
    model_config = config["model"]
    model_params = {
        "vocab_size": int(model_config["vocab_size"]),
        "d_model": int(model_config["d_model"]),
        "n_layers": int(model_config["n_layers"]),
        "n_heads": int(model_config["n_heads"]),
        "context_len": int(model_config["context_len"]),
        "lex": False,  # Will be overridden by trainer
    }

    # Training parameters
    training_config = config["training"]
    training_params = {
        "learning_rate": float(training_config["learning_rate"]),
        "batch_size": int(training_config["batch_size"]),
        "num_epochs": int(training_config["num_epochs"]),
        "optimizer": training_config["optimizer"],
        "weight_decay": float(training_config["weight_decay"]),
        "warmup_steps": int(training_config["warmup_steps"]),
        "max_grad_norm": float(training_config["max_grad_norm"]),
        "wandb": training_config.get("wandb", "False").lower() == "true",
        "early_stopping": training_config.get("early_stopping", "False").lower() == "true",
        "early_stopping_patience": int(training_config.get("early_stopping_patience", 10)),
        "early_stopping_min_delta": float(training_config.get("early_stopping_min_delta", 1e-4)),
    }

    # Dataset parameters
    dataset_config = config["dataset"]
    dataset_params = {
        "n_tasks": int(dataset_config["n_tasks"]),
        "n_steps": int(dataset_config["n_steps"]),
        "seed": int(dataset_config["seed"]),
    }

    # Evaluation parameters
    eval_config = config["evaluation"]
    eval_params = {
        "eval_frequency": int(eval_config["eval_frequency"]),
        "save_frequency": int(eval_config["save_frequency"]),
        "test_steps": int(eval_config["test_steps"]),
    }

    # Experiment parameters
    experiment_config = config["experiment"]
    experiment_params = {
        "config_name": experiment_config["config_name"],
        "save_results": experiment_config["save_results"].lower() == "true",
        "model_type": experiment_config.get("model_type", "normal"),
    }

    # System parameters
    system_config = config["system"]
    device = system_config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    system_params = {"device": device}

    return {
        "model": model_params,
        "training": training_params,
        "dataset": dataset_params,
        "evaluation": eval_params,
        "experiment": experiment_params,
        "system": system_params,
    }


def kl_div(p, q, eps=1e-10):
    """
    Compute KL divergence KL(p || q) - how much information is lost when using q to approximate p.
    
    Args:
        p: Reference distribution (true distribution)
        q: Approximating distribution (model prediction)
        eps: Small epsilon for numerical stability
        
    Returns:
        KL divergence as a scalar
    """
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return torch.sum(p * torch.log(p / q)).item()
