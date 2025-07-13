"""
Default configuration for MixerEncoding.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class DefaultConfig:
    """Default configuration for training and evaluation."""
    
    # Model parameters
    model_name: str = "SeqFunNet"
    num_classes: int = 6
    depth: int = 4
    patch_size: int = 15
    dim: int = 30
    expansion_factor: int = 4
    dropout: float = 0.1
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    
    # Data parameters
    dataset: str = "bird_6"
    feature_type: str = "mfcc"
    input_height: int = 60
    input_width: int = 60
    num_workers: int = 4
    
    # Device and hardware
    device: str = "auto"  # "auto", "cuda", "cpu"
    gpu_ids: List[int] = None
    mixed_precision: bool = True
    
    # Logging and saving
    save_dir: str = "results"
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 1
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "mixerencoding"
    wandb_entity: str = None
    
    # Evaluation
    test_samples: Optional[int] = None
    metrics: List[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DefaultConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> "DefaultConfig":
        """Update config with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


# Predefined configurations for common use cases
@dataclass
class QuickConfig(DefaultConfig):
    """Quick training configuration for testing."""
    epochs: int = 10
    batch_size: int = 16
    save_interval: int = 5


@dataclass
class FullConfig(DefaultConfig):
    """Full training configuration for production."""
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    mixed_precision: bool = True


@dataclass
class AblationConfig(DefaultConfig):
    """Configuration for ablation studies."""
    epochs: int = 50
    batch_size: int = 32
    save_interval: int = 5
    test_samples: int = 1000 