"""
MLP-Mixer implementation for audio classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple
from einops.layers.torch import Rearrange, Reduce

from .base_model import BaseModel


class PreNormResidual(nn.Module):
    """Pre-norm residual block."""
    
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x)) + x


def FeedForward(dim: int, expansion_factor: int = 4, dropout: float = 0., dense: type = nn.Linear) -> nn.Module:
    """
    Create a feed-forward network.
    
    Args:
        dim: Input dimension
        expansion_factor: Expansion factor for hidden dimension
        dropout: Dropout rate
        dense: Linear layer type (nn.Linear or nn.Conv1d)
        
    Returns:
        Feed-forward network
    """
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


class MLPMixer(nn.Module):
    """
    MLP-Mixer architecture.
    
    Reference: "MLP-Mixer: An all-MLP Architecture for Vision"
    """
    
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        expansion_factor: int = 4,
        dropout: float = 0.,
        in_chans: int = 1
    ):
        """
        Initialize MLP-Mixer.
        
        Args:
            image_size: Size of input image (assumed square)
            patch_size: Size of patches
            dim: Hidden dimension
            depth: Number of mixer layers
            expansion_factor: Expansion factor for MLPs
            dropout: Dropout rate
            in_chans: Number of input channels
        """
        super().__init__()
        
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        
        # Use Conv1d for token mixing, Linear for channel mixing
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        
        self.mixer = nn.Sequential(
            # Patch embedding
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * in_chans, dim),
            
            # Mixer layers
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
            
            # Final normalization and pooling
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, dim)
        """
        return self.mixer(x)


class MLPMixerModel(BaseModel):
    """
    Complete MLP-Mixer model for audio classification.
    """
    
    def __init__(
        self,
        num_classes: int,
        image_size: int = 60,
        patch_size: int = 15,
        dim: int = 30,
        depth: int = 8,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        in_chans: int = 1
    ):
        """
        Initialize MLP-Mixer model.
        
        Args:
            num_classes: Number of output classes
            image_size: Size of input image (assumed square)
            patch_size: Size of patches
            dim: Hidden dimension
            depth: Number of mixer layers
            expansion_factor: Expansion factor for MLPs
            dropout: Dropout rate
            in_chans: Number of input channels
        """
        super().__init__(num_classes, 
                        image_size=image_size,
                        patch_size=patch_size,
                        dim=dim,
                        depth=depth,
                        expansion_factor=expansion_factor,
                        dropout=dropout,
                        in_chans=in_chans)
        
        self.mlp_mixer = MLPMixer(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            expansion_factor=expansion_factor,
            dropout=dropout,
            in_chans=in_chans
        )
        
        # Classification head
        self.classifier = nn.Linear(dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (logits, softmax_output)
        """
        # Ensure input has correct shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Extract features
        features = self.mlp_mixer(x)
        
        # Classification
        logits = self.classifier(features)
        softmax_out = self.softmax(logits)
        
        return logits, softmax_out
    
    def extract_features(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """
        Extract features from the MLP-Mixer.
        
        Args:
            x: Input tensor
            layer_name: Layer name (ignored for this model)
            
        Returns:
            Feature tensor
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        with torch.no_grad():
            return self.mlp_mixer(x)
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.model_config['dim']


# Convenience function for creating MLP-Mixer models
def create_mlp_mixer(
    num_classes: int,
    image_size: int = 60,
    patch_size: int = 15,
    dim: int = 30,
    depth: int = 8,
    **kwargs
) -> MLPMixerModel:
    """
    Create an MLP-Mixer model with specified parameters.
    
    Args:
        num_classes: Number of output classes
        image_size: Size of input image
        patch_size: Size of patches
        dim: Hidden dimension
        depth: Number of mixer layers
        **kwargs: Additional arguments
        
    Returns:
        MLP-Mixer model
    """
    return MLPMixerModel(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        **kwargs
    ) 