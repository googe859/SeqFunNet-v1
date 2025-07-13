"""
Transformer-only model for audio classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base_model import BaseModel


class TransformerOnly(BaseModel):
    """
    Transformer-only model for audio classification.
    
    This model uses only Transformer architecture for feature extraction
    and classification, without MLP-Mixer or other components.
    """
    
    def __init__(
        self,
        num_classes: int,
        d_model: int = 60,
        nhead: int = 5,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.5,
        activation: str = "relu",
        input_height: int = 60,
        input_width: int = 60
    ):
        """
        Initialize Transformer-only model.
        
        Args:
            num_classes: Number of output classes
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            activation: Activation function
            input_height: Input feature height
            input_width: Input feature width
        """
        super().__init__(num_classes,
                        d_model=d_model,
                        nhead=nhead,
                        num_layers=num_layers,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation,
                        input_height=input_height,
                        input_width=input_width)
        
        # MaxPool for dimension reduction
        self.maxpool = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])
        
        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=num_layers
        )
        
        # Local attention for additional feature refinement
        self.local_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Transformer-only model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (logits, softmax_output)
        """
        # Ensure input has correct shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Apply maxpool for dimension reduction
        x_maxpool = self.maxpool(x)  # (B, 1, H, W//2)
        
        # Remove channel dimension and permute for transformer
        x_reduced = torch.squeeze(x_maxpool, 1)  # (B, H, W//2)
        x_trans = x_reduced.permute(2, 0, 1)  # (W//2, B, H)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x_trans)
        
        # Apply local attention
        local_attention_output, _ = self.local_attention(
            transformer_output, 
            transformer_output, 
            transformer_output
        )
        
        # Global average pooling
        transformer_embedding = torch.mean(transformer_output, dim=0)  # (B, d_model)
        local_attention_embedding = torch.mean(local_attention_output, dim=0)  # (B, d_model)
        
        # Combine embeddings
        combined_embedding = transformer_embedding + local_attention_embedding
        
        # Classification
        logits = self.classifier(combined_embedding)
        softmax_out = self.softmax(logits)
        
        return logits, softmax_out
    
    def extract_features(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """
        Extract features from specific layers.
        
        Args:
            x: Input tensor
            layer_name: Layer name to extract features from
                       Options: 'transformer', 'attention', 'combined'
            
        Returns:
            Feature tensor
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        with torch.no_grad():
            x_maxpool = self.maxpool(x)
            x_reduced = torch.squeeze(x_maxpool, 1)
            x_trans = x_reduced.permute(2, 0, 1)
            
            if layer_name == "transformer":
                transformer_output = self.transformer_encoder(x_trans)
                return torch.mean(transformer_output, dim=0)
            
            elif layer_name == "attention":
                transformer_output = self.transformer_encoder(x_trans)
                local_attention_output, _ = self.local_attention(
                    transformer_output, transformer_output, transformer_output
                )
                return torch.mean(local_attention_output, dim=0)
            
            else:
                # Return final combined features
                logits, _ = self.forward(x)
                return logits
    
    def get_feature_dim(self) -> int:
        """Get the dimension of the final feature representation."""
        return self.model_config['d_model']


# Convenience function for creating Transformer-only models
def create_transformer_only(
    num_classes: int,
    d_model: int = 60,
    nhead: int = 5,
    num_layers: int = 2,
    **kwargs
) -> TransformerOnly:
    """
    Create a Transformer-only model with specified parameters.
    
    Args:
        num_classes: Number of output classes
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        **kwargs: Additional arguments
        
    Returns:
        Transformer-only model
    """
    return TransformerOnly(
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        **kwargs
    ) 