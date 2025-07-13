"""
SeqFunNet: Sequential Function Network for audio classification.

This model combines Transformer, MLP-Mixer, and GAM attention mechanisms
for effective audio feature learning and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base_model import BaseModel
from .attention import GAM, MultiHeadAttention
from .mlp_mixer import MLPMixer


class SeqFunNet(BaseModel):
    """
    SeqFunNet: A hybrid model combining Transformer, MLP-Mixer, and GAM attention.
    
    This model processes audio features through multiple branches:
    1. Transformer branch for temporal feature extraction
    2. MLP-Mixer branch for spatial feature mixing
    3. GAM branch for channel and spatial attention
    4. Feature fusion for final classification
    """
    
    def __init__(
        self,
        num_classes: int,
        depth: int = 4,
        patch_size: int = 15,
        dim: int = 30,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        transformer_d_model: int = 60,
        transformer_nhead: int = 5,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.5,
        gam_rate: int = 4,
        mlp_hidden_dims: Optional[list] = None
    ):
        """
        Initialize SeqFunNet.
        
        Args:
            num_classes: Number of output classes
            depth: MLP-Mixer depth
            patch_size: MLP-Mixer patch size
            dim: MLP-Mixer dimension
            expansion_factor: MLP-Mixer expansion factor
            dropout: Dropout rate
            transformer_d_model: Transformer model dimension
            transformer_nhead: Number of transformer attention heads
            transformer_layers: Number of transformer layers
            transformer_dropout: Transformer dropout rate
            gam_rate: GAM attention reduction rate
            mlp_hidden_dims: Hidden dimensions for MLP branch
        """
        super().__init__(num_classes,
                        depth=depth,
                        patch_size=patch_size,
                        dim=dim,
                        expansion_factor=expansion_factor,
                        dropout=dropout,
                        transformer_d_model=transformer_d_model,
                        transformer_nhead=transformer_nhead,
                        transformer_layers=transformer_layers,
                        transformer_dropout=transformer_dropout,
                        gam_rate=gam_rate,
                        mlp_hidden_dims=mlp_hidden_dims)
        
        # Set default MLP hidden dimensions
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [64, 128, 120]
        
        # ========== Transformer Branch ========== #
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=64,
            dropout=transformer_dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=transformer_layers
        )
        
        self.local_attn = MultiHeadAttention(
            embed_dim=transformer_d_model,
            num_heads=transformer_nhead
        )
        
        # ========== GAM Branch ========== #
        # 60x345经过maxpool后变成60x172，所以GAM输入通道是172
        self.gam = GAM(172, 172, rate=gam_rate)
        
        # ========== MLP Branch ========== #
        # 60x172经过GAM后展平：60*172 = 10320
        self.mlp = nn.Sequential(
            nn.Linear(in_features=10320, out_features=mlp_hidden_dims[0]),
            nn.BatchNorm1d(mlp_hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.Linear(mlp_hidden_dims[1], mlp_hidden_dims[2])
        )
        
        # ========== MLP-Mixer Branch ========== #
        self.mlp_mixer = MLPMixer(
            image_size=60,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            expansion_factor=expansion_factor,
            dropout=dropout
        )
        self.mixer_proj = nn.Linear(dim, mlp_hidden_dims[2])  # 投影到120维
        
        # ========== Final Classification Head ========== #
        self.classifier = nn.Linear(mlp_hidden_dims[2], num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SeqFunNet.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (logits, softmax_output)
        """
        # Ensure input has correct shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # ========== 1. MaxPool2d只对宽度做池化 ========== #
        x_maxpool = self.transformer_maxpool(x)
        
        # ========== 2. Transformer Branch ========== #
        # Remove channel dim: (B, 1, 60, W//2) --> (B, 60, W//2)
        x_maxpool_reduced = torch.squeeze(x_maxpool, 1)
        x_trans = x_maxpool_reduced.permute(2, 0, 1)  # (W//2, B, 60)
        
        # Transformer processing
        transformer_output = self.transformer_encoder(x_trans)
        transformer_embedding = torch.mean(transformer_output, dim=0)  # (B, 60)
        
        # Local attention
        local_attention_output, _ = self.local_attn(
            transformer_output, 
            transformer_output, 
            transformer_output
        )
        local_attention_embedding = torch.mean(local_attention_output, dim=0)  # (B, 60)
        
        # ========== 3. GAM Branch ========== #
        x1 = x_trans.unsqueeze(0)  # Add batch dimension for GAM
        x2 = self.gam(x1)
        
        # ========== 4. MLP Branch ========== #
        x2 = x2.reshape(-1, 10320)  # [B, 10320]
        mlp_out = self.mlp(x2)  # [B, 120]
        
        # ========== 5. MLP-Mixer Branch ========== #
        # 将矩形输入调整为正方形用于MLP-Mixer
        x_square = F.adaptive_avg_pool2d(x, (60, 60))
        mixer_out = self.mlp_mixer(x_square)  # (B, dim)
        mixer_out = self.mixer_proj(mixer_out)  # (B, 120)
        
        # ========== 6. Feature Fusion and Final Classification ========== #
        # Combine transformer embeddings
        transformer_combined = torch.cat([transformer_embedding, local_attention_embedding], dim=1)  # (B, 120)
        
        # Final feature fusion
        complete_embedding = mlp_out + mixer_out + transformer_combined  # (B, 120)
        
        # Classification
        logits = self.classifier(complete_embedding)
        softmax_out = self.softmax(logits)
        
        return logits, softmax_out
    
    def extract_features(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """
        Extract features from specific layers or the final layer.
        
        Args:
            x: Input tensor
            layer_name: Layer name to extract features from
                       Options: 'transformer', 'mlp', 'mixer', 'combined'
            
        Returns:
            Feature tensor
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        with torch.no_grad():
            if layer_name == "transformer":
                # Extract transformer features
                x_maxpool = self.transformer_maxpool(x)
                x_maxpool_reduced = torch.squeeze(x_maxpool, 1)
                x_trans = x_maxpool_reduced.permute(2, 0, 1)
                transformer_output = self.transformer_encoder(x_trans)
                return torch.mean(transformer_output, dim=0)
            
            elif layer_name == "mlp":
                # Extract MLP features
                x_maxpool = self.transformer_maxpool(x)
                x_maxpool_reduced = torch.squeeze(x_maxpool, 1)
                x_trans = x_maxpool_reduced.permute(2, 0, 1)
                x1 = x_trans.unsqueeze(0)
                x2 = self.gam(x1)
                x2 = x2.reshape(-1, 10320)
                return self.mlp(x2)
            
            elif layer_name == "mixer":
                # Extract MLP-Mixer features
                x_square = F.adaptive_avg_pool2d(x, (60, 60))
                mixer_out = self.mlp_mixer(x_square)
                return self.mixer_proj(mixer_out)
            
            else:
                # Extract final combined features
                logits, _ = self.forward(x)
                return logits
    
    def get_feature_dim(self) -> int:
        """Get the dimension of the final feature representation."""
        return self.model_config.get('mlp_hidden_dims', [64, 128, 120])[-1]


# Convenience function for creating SeqFunNet models
def create_seq_fun_net(
    num_classes: int,
    depth: int = 4,
    patch_size: int = 15,
    dim: int = 30,
    **kwargs
) -> SeqFunNet:
    """
    Create a SeqFunNet model with specified parameters.
    
    Args:
        num_classes: Number of output classes
        depth: MLP-Mixer depth
        patch_size: MLP-Mixer patch size
        dim: MLP-Mixer dimension
        **kwargs: Additional arguments
        
    Returns:
        SeqFunNet model
    """
    return SeqFunNet(
        num_classes=num_classes,
        depth=depth,
        patch_size=patch_size,
        dim=dim,
        **kwargs
    ) 