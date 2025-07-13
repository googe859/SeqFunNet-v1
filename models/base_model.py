"""
Base model class for all models in MixerEncoding.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BaseModel(nn.Module, ABC):
    """
    Base class for all models in MixerEncoding.
    
    This class provides a unified interface for all models and includes
    common functionality like model complexity analysis and feature extraction.
    """
    
    def __init__(self, num_classes: int, **kwargs):
        """
        Initialize the base model.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_config = kwargs
        
        # Register model parameters for tracking
        self._register_model_info()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (logits, softmax_output)
        """
        pass
    
    def _register_model_info(self):
        """Register model information for tracking."""
        self.model_info = {
            "num_classes": self.num_classes,
            "config": self.model_config,
            "total_params": 0,
            "trainable_params": 0,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including parameter counts.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.model_info.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        })
        
        return self.model_info
    
    def analyze_complexity(self, input_shape: Tuple[int, ...] = (1, 1, 60, 60)) -> Dict[str, Any]:
        """
        Analyze model complexity including FLOPs and memory usage.
        
        Args:
            input_shape: Input tensor shape for analysis
            
        Returns:
            Dictionary containing complexity metrics
        """
        try:
            from thop import profile, clever_format
            
            # Create dummy input
            device = next(self.parameters()).device
            dummy_input = torch.randn(input_shape).to(device)
            
            # Calculate FLOPs
            flops, params = profile(self, inputs=(dummy_input,), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
            
            # Measure inference time
            self.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = self(dummy_input)
                
                # Measure inference time
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                
                if device.type == 'cuda':
                    start_time.record()
                    for _ in range(100):
                        _ = self(dummy_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    avg_latency = start_time.elapsed_time(end_time) / 100  # ms
                else:
                    import time
                    start_time = time.time()
                    for _ in range(100):
                        _ = self(dummy_input)
                    end_time = time.time()
                    avg_latency = (end_time - start_time) * 1000 / 100  # ms
            
            # Measure memory usage
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = self(dummy_input)
                memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            
            return {
                "flops": flops,
                "params": params,
                "latency_ms": avg_latency,
                "fps": 1000 / avg_latency,
                "memory_mb": memory_usage,
                **self.get_model_info()
            }
            
        except ImportError:
            print("Warning: thop not installed. FLOPs calculation skipped.")
            return {
                "flops": "N/A",
                "params": "N/A", 
                "latency_ms": "N/A",
                "fps": "N/A",
                "memory_mb": "N/A",
                **self.get_model_info()
            }
    
    def extract_features(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """
        Extract features from a specific layer or the final layer.
        
        Args:
            x: Input tensor
            layer_name: Name of the layer to extract features from (None for final layer)
            
        Returns:
            Feature tensor
        """
        # Default implementation - override in subclasses if needed
        with torch.no_grad():
            logits, _ = self.forward(x)
            return logits
    
    def get_feature_dim(self) -> int:
        """
        Get the dimension of the final feature representation.
        
        Returns:
            Feature dimension
        """
        # Default implementation - override in subclasses
        return self.num_classes
    
    def save_model(self, filepath: str, include_optimizer: bool = False, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
            include_optimizer: Whether to save optimizer state
            optimizer: Optimizer to save (if include_optimizer is True)
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config,
            'model_info': self.get_model_info(),
        }
        
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None) -> 'BaseModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model instance
        model = cls(**checkpoint['model_config'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        print(f"Model loaded from {filepath}")
        return model
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters by type.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                non_trainable_params += param.numel()
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        param_counts = self.count_parameters()
        return (f"{self.__class__.__name__}(num_classes={self.num_classes}, "
                f"total_params={param_counts['total']:,}, "
                f"trainable_params={param_counts['trainable']:,})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__() 