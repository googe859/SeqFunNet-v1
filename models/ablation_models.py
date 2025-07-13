"""
Ablation study models for SeqFunNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base_model import BaseModel
from .attention import GAM
from .mlp_mixer import MLPMixer


class SeqFunNetAblation(BaseModel):
    """
    Ablation study model for SeqFunNet.
    
    This model allows systematic removal of components to study their contribution:
    - Transformer branch
    - MLP-Mixer branch  
    - GAM attention
    - Feature fusion strategies
    """
    
    def __init__(
        self,
        num_classes: int,
        use_transformer: bool = True,
        use_mixer: bool = True,
        use_gam: bool = True,
        use_mlp: bool = True,
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
        mlp_hidden_dims: Optional[list] = None,
        fusion_strategy: str = "add"  # "add", "concat", "weighted"
    ):
        """
        Initialize SeqFunNet ablation model.
        
        Args:
            num_classes: Number of output classes
            use_transformer: Whether to use transformer branch
            use_mixer: Whether to use MLP-Mixer branch
            use_gam: Whether to use GAM attention
            use_mlp: Whether to use MLP branch
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
            fusion_strategy: Feature fusion strategy
        """
        super().__init__(num_classes,
                        use_transformer=use_transformer,
                        use_mixer=use_mixer,
                        use_gam=use_gam,
                        use_mlp=use_mlp,
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
                        mlp_hidden_dims=mlp_hidden_dims,
                        fusion_strategy=fusion_strategy)
        
        # Set default MLP hidden dimensions
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [64, 128, 120]
        
        self.use_transformer = use_transformer
        self.use_mixer = use_mixer
        self.use_gam = use_gam
        self.use_mlp = use_mlp
        self.fusion_strategy = fusion_strategy
        
        # ========== Transformer Branch ========== #
        if use_transformer:
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
            
            self.local_attn = nn.MultiheadAttention(
                embed_dim=transformer_d_model,
                num_heads=transformer_nhead
            )
        
        # ========== GAM Branch ========== #
        if use_gam:
            self.gam = GAM(172, 172, rate=gam_rate)
        
        # ========== MLP Branch ========== #
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=10320, out_features=mlp_hidden_dims[0]),
                nn.BatchNorm1d(mlp_hidden_dims[0]),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
                nn.Linear(mlp_hidden_dims[1], mlp_hidden_dims[2])
            )
        
        # ========== MLP-Mixer Branch ========== #
        if use_mixer:
            self.mlp_mixer = MLPMixer(
                image_size=60,
                patch_size=patch_size,
                dim=dim,
                depth=depth,
                expansion_factor=expansion_factor,
                dropout=dropout
            )
            self.mixer_proj = nn.Linear(dim, mlp_hidden_dims[2])
        
        # ========== Feature Fusion ========== #
        self._setup_fusion()
        
        # ========== Final Classification Head ========== #
        self.classifier = nn.Linear(self.final_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
    
    def _setup_fusion(self):
        """Setup feature fusion based on enabled components."""
        feature_dims = []
        
        if self.use_transformer:
            feature_dims.append(120)  # transformer_combined dimension
        
        if self.use_mlp:
            feature_dims.append(120)  # mlp output dimension
        
        if self.use_mixer:
            feature_dims.append(120)  # mixer output dimension
        
        if not feature_dims:
            raise ValueError("At least one component must be enabled")
        
        self.final_dim = sum(feature_dims)
        
        # Setup fusion weights if using weighted fusion
        if self.fusion_strategy == "weighted" and len(feature_dims) > 1:
            self.fusion_weights = nn.Parameter(torch.ones(len(feature_dims)))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ablation model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (logits, softmax_output)
        """
        # Ensure input has correct shape
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        features = []
        
        # ========== Transformer Branch ========== #
        if self.use_transformer:
            x_maxpool = self.transformer_maxpool(x)
            x_maxpool_reduced = torch.squeeze(x_maxpool, 1)
            x_trans = x_maxpool_reduced.permute(2, 0, 1)
            
            transformer_output = self.transformer_encoder(x_trans)
            transformer_embedding = torch.mean(transformer_output, dim=0)
            
            local_attention_output, _ = self.local_attn(
                transformer_output, transformer_output, transformer_output
            )
            local_attention_embedding = torch.mean(local_attention_output, dim=0)
            
            transformer_combined = torch.cat([transformer_embedding, local_attention_embedding], dim=1)
            features.append(transformer_combined)
        
        # ========== MLP Branch ========== #
        if self.use_mlp:
            if self.use_transformer:
                x1 = x_trans.unsqueeze(0)
            else:
                # If no transformer, create input for MLP
                x_maxpool = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(x)
                x_maxpool_reduced = torch.squeeze(x_maxpool, 1)
                x_trans = x_maxpool_reduced.permute(2, 0, 1)
                x1 = x_trans.unsqueeze(0)
            
            if self.use_gam:
                x2 = self.gam(x1)
            else:
                x2 = x1
            
            x2 = x2.reshape(-1, 10320)
            mlp_out = self.mlp(x2)
            features.append(mlp_out)
        
        # ========== MLP-Mixer Branch ========== #
        if self.use_mixer:
            x_square = F.adaptive_avg_pool2d(x, (60, 60))
            mixer_out = self.mlp_mixer(x_square)
            mixer_out = self.mixer_proj(mixer_out)
            features.append(mixer_out)
        
        # ========== Feature Fusion ========== #
        if self.fusion_strategy == "add" and len(features) > 1:
            complete_embedding = sum(features)
        elif self.fusion_strategy == "concat":
            complete_embedding = torch.cat(features, dim=1)
        elif self.fusion_strategy == "weighted" and len(features) > 1:
            weights = F.softmax(self.fusion_weights, dim=0)
            complete_embedding = sum(w * f for w, f in zip(weights, features))
        else:
            complete_embedding = features[0]
        
        # Classification
        logits = self.classifier(complete_embedding)
        softmax_out = self.softmax(logits)
        
        return logits, softmax_out
    
    def get_ablation_info(self) -> dict:
        """
        Get information about the ablation configuration.
        
        Returns:
            Dictionary containing ablation information
        """
        return {
            "use_transformer": self.use_transformer,
            "use_mixer": self.use_mixer,
            "use_gam": self.use_gam,
            "use_mlp": self.use_mlp,
            "fusion_strategy": self.fusion_strategy,
            "final_dim": self.final_dim,
            "model_name": self._get_model_name()
        }
    
    def _get_model_name(self) -> str:
        """Get descriptive name for the ablation configuration."""
        components = []
        if self.use_transformer:
            components.append("Transformer")
        if self.use_mixer:
            components.append("Mixer")
        if self.use_gam:
            components.append("GAM")
        if self.use_mlp:
            components.append("MLP")
        
        return f"SeqFunNet-{'+'.join(components)}-{self.fusion_strategy.capitalize()}"


# Predefined ablation configurations
def create_ablation_configs(num_classes: int) -> dict:
    """
    Create predefined ablation configurations.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Dictionary of ablation configurations
    """
    configs = {
        "full": {
            "use_transformer": True,
            "use_mixer": True,
            "use_gam": True,
            "use_mlp": True,
            "fusion_strategy": "add"
        },
        "no_transformer": {
            "use_transformer": False,
            "use_mixer": True,
            "use_gam": True,
            "use_mlp": True,
            "fusion_strategy": "add"
        },
        "no_mixer": {
            "use_transformer": True,
            "use_mixer": False,
            "use_gam": True,
            "use_mlp": True,
            "fusion_strategy": "add"
        },
        "no_gam": {
            "use_transformer": True,
            "use_mixer": True,
            "use_gam": False,
            "use_mlp": True,
            "fusion_strategy": "add"
        },
        "no_mlp": {
            "use_transformer": True,
            "use_mixer": True,
            "use_gam": True,
            "use_mlp": False,
            "fusion_strategy": "add"
        },
        "transformer_only": {
            "use_transformer": True,
            "use_mixer": False,
            "use_gam": False,
            "use_mlp": False,
            "fusion_strategy": "add"
        },
        "mixer_only": {
            "use_transformer": False,
            "use_mixer": True,
            "use_gam": False,
            "use_mlp": False,
            "fusion_strategy": "add"
        },
        "concat_fusion": {
            "use_transformer": True,
            "use_mixer": True,
            "use_gam": True,
            "use_mlp": True,
            "fusion_strategy": "concat"
        },
        "weighted_fusion": {
            "use_transformer": True,
            "use_mixer": True,
            "use_gam": True,
            "use_mlp": True,
            "fusion_strategy": "weighted"
        }
    }
    
    return configs


def create_ablation_model(config_name: str, num_classes: int, **kwargs) -> SeqFunNetAblation:
    """
    Create ablation model from predefined configuration.
    
    Args:
        config_name: Name of the ablation configuration
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        Ablation model
    """
    configs = create_ablation_configs(num_classes)
    
    if config_name not in configs:
        raise ValueError(f"Unknown ablation config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name]
    config.update(kwargs)
    
    return SeqFunNetAblation(num_classes=num_classes, **config) 