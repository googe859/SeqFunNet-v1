"""
Model configurations for different architectures.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str
    num_classes: int
    input_height: int = 60
    input_width: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {key: value for key, value in self.__dict__.items()}


@dataclass
class SeqFunNetConfig(ModelConfig):
    """Configuration for SeqFunNet model."""
    depth: int = 4
    patch_size: int = 15
    dim: int = 30
    expansion_factor: int = 4
    dropout: float = 0.1
    transformer_d_model: int = 60
    transformer_nhead: int = 5
    transformer_layers: int = 2
    transformer_dropout: float = 0.5
    gam_rate: int = 4
    mlp_hidden_dims: list = None
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 128, 120]


@dataclass
class MLPMixerConfig(ModelConfig):
    """Configuration for MLP-Mixer model."""
    depth: int = 8
    patch_size: int = 15
    dim: int = 30
    expansion_factor: int = 4
    dropout: float = 0.1


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Transformer model."""
    d_model: int = 60
    nhead: int = 5
    num_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.5
    activation: str = "relu"


@dataclass
class ModelFusionConfig(ModelConfig):
    """Configuration for Model Fusion."""
    depth: int = 8
    patch_size: int = 15
    dim: int = 30
    input_width: int = 30
    transformer_d_model: int = 60
    transformer_nhead: int = 3
    transformer_layers: int = 3
    transformer_dropout: float = 0.15


@dataclass
class EncoderMLPConfig(ModelConfig):
    """Configuration for Encoder-MLP model."""
    transformer_d_model: int = 50
    transformer_nhead: int = 5
    transformer_layers: int = 2
    transformer_dropout: float = 0.5
    mlp_hidden_dims: list = None
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 128, 30]


@dataclass
class EncoderLinearConfig(ModelConfig):
    """Configuration for Encoder-Linear model."""
    d_model: int = 12
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.5


class ModelConfigs:
    """Factory class for model configurations."""
    
    @staticmethod
    def get_config(model_name: str, num_classes: int, **kwargs) -> ModelConfig:
        """Get model configuration by name."""
        configs = {
            "SeqFunNet": SeqFunNetConfig,
            "MLPMixer": MLPMixerConfig,
            "Transformer": TransformerConfig,
            "ModelFusion": ModelFusionConfig,
            "EncoderMLP": EncoderMLPConfig,
            "EncoderLinear": EncoderLinearConfig,
        }
        
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")
        
        config_class = configs[model_name]
        return config_class(name=model_name, num_classes=num_classes, **kwargs)
    
    @staticmethod
    def get_all_configs(num_classes: int) -> Dict[str, ModelConfig]:
        """Get all model configurations."""
        configs = {}
        for model_name in ["SeqFunNet", "MLPMixer", "Transformer", "ModelFusion", "EncoderMLP", "EncoderLinear"]:
            configs[model_name] = ModelConfigs.get_config(model_name, num_classes)
        return configs


# Predefined configurations for common datasets
BIRD_6_CONFIGS = {
    "SeqFunNet": SeqFunNetConfig(
        name="SeqFunNet", num_classes=6, depth=4, patch_size=15, dim=30
    ),
    "MLPMixer": MLPMixerConfig(
        name="MLPMixer", num_classes=6, depth=8, patch_size=15, dim=30
    ),
    "Transformer": TransformerConfig(
        name="Transformer", num_classes=6, d_model=60, nhead=5, num_layers=2
    ),
}

MARINE_MAMMALS_15_CONFIGS = {
    "SeqFunNet": SeqFunNetConfig(
        name="SeqFunNet", num_classes=15, depth=6, patch_size=15, dim=40
    ),
    "MLPMixer": MLPMixerConfig(
        name="MLPMixer", num_classes=15, depth=10, patch_size=15, dim=40
    ),
    "Transformer": TransformerConfig(
        name="Transformer", num_classes=15, d_model=60, nhead=6, num_layers=3
    ),
}

URBAN_SOUND_10_CONFIGS = {
    "SeqFunNet": SeqFunNetConfig(
        name="SeqFunNet", num_classes=10, depth=4, patch_size=15, dim=30
    ),
    "MLPMixer": MLPMixerConfig(
        name="MLPMixer", num_classes=10, depth=8, patch_size=15, dim=30
    ),
    "Transformer": TransformerConfig(
        name="Transformer", num_classes=10, d_model=60, nhead=5, num_layers=2
    ),
} 