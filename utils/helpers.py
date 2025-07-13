"""
Helper utilities for MixerEncoding.
"""

import random
import numpy as np
import torch
from typing import Optional, Union, List
import os
import json
import yaml


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config: dict, filepath: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError("Unsupported file format. Use .json or .yaml")


def load_config(filepath: str) -> dict:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .yaml")


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory
        experiment_name: Experiment name
        
    Returns:
        Path to experiment directory
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Accuracy percentage
    """
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Calculate multiple metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    _, predicted = torch.max(predictions, 1)
    
    # Convert to numpy
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predicted, average='weighted')
    conf_matrix = confusion_matrix(targets, predicted)
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'confusion_matrix': conf_matrix
    }


def create_lr_scheduler(optimizer, scheduler_type: str, **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        **kwargs: Scheduler parameters
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_type.lower() == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_optimizer(optimizer_type: str, model_params, **kwargs):
    """
    Create optimizer.
    
    Args:
        optimizer_type: Type of optimizer
        model_params: Model parameters
        **kwargs: Optimizer parameters
        
    Returns:
        PyTorch optimizer
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model_params, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model_params, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model_params, **kwargs)
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(model_params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}") 