"""
Device management utilities.
"""

import torch
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device.
    
    Args:
        device: Device specification ('cuda', 'cpu', 'auto')
        
    Returns:
        torch.device instance
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_device(device: str) -> torch.device:
    """
    Set the default device.
    
    Args:
        device: Device specification
        
    Returns:
        torch.device instance
    """
    torch_device = get_device(device)
    
    if torch_device.type == "cuda":
        # Set default tensor type to cuda
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        # Set default tensor type to cpu
        torch.set_default_tensor_type('torch.FloatTensor')
    
    return torch_device


def get_gpu_info() -> dict:
    """
    Get GPU information.
    
    Returns:
        Dictionary containing GPU information
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    gpu_info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "current": torch.cuda.current_device(),
        "name": torch.cuda.get_device_name(),
        "memory": {
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
        }
    }
    
    return gpu_info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def to_device(data, device: torch.device):
    """
    Move data to specified device.
    
    Args:
        data: Data to move (tensor, list, dict, etc.)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    else:
        return data 