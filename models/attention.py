"""
Attention mechanisms for audio classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GAM(nn.Module):
    """
    Global Attention Module (GAM).
    
    Combines channel attention and spatial attention for enhanced feature representation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, rate: int = 4):
        """
        Initialize GAM.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            rate: Reduction rate for attention computation
        """
        super().__init__()
        
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor with attention applied
        """
        b, c, h, w = x.shape
        
        # Channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        
        # Spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Wrapper around PyTorch's MultiheadAttention with additional functionality.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False
    ):
        """
        Initialize MultiHeadAttention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Parallel attention heads
            dropout: Dropout probability
            bias: Add bias as module parameter
            add_bias_kv: Add bias to the key and value sequences
            add_zero_attn: Add a new batch of zeros to the key and value sequences
            kdim: Total number of features in key
            vdim: Total number of features in value
            batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first
        )
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            query: Query embeddings of shape (L, N, E_q) or (N, L, E_q)
            key: Key embeddings of shape (S, N, E_k) or (N, S, E_k)
            value: Value embeddings of shape (S, N, E_v) or (N, S, E_v)
            key_padding_mask: If specified, a mask of shape (N, S) indicating which elements should be ignored
            need_weights: If specified, returns attention weights
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions
            average_attn_weights: If true, indicates that the returned attn_weights should be averaged across heads
            is_causal: If specified, applies a causal mask as attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        return self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal
        )
    
    def get_attention_weights(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Get attention weights without computing output.
        
        Args:
            query: Query embeddings
            key: Key embeddings
            value: Value embeddings
            **kwargs: Additional arguments
            
        Returns:
            Attention weights
        """
        with torch.no_grad():
            _, weights = self.forward(
                query=query,
                key=key,
                value=value,
                need_weights=True,
                **kwargs
            )
        return weights


class SelfAttention(nn.Module):
    """
    Self-attention module for sequence modeling.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True
    ):
        """
        Initialize SelfAttention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            batch_first: Whether input is batch-first
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor with self-attention applied
        """
        # Apply self-attention
        attn_out, _ = self.attention(query=x, key=x, value=x)
        
        # Residual connection and normalization
        out = self.norm(x + self.dropout(attn_out))
        
        return out


class CrossAttention(nn.Module):
    """
    Cross-attention module for attending to different sequences.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True
    ):
        """
        Initialize CrossAttention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            batch_first: Whether input is batch-first
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            
        Returns:
            Output tensor with cross-attention applied
        """
        # Apply cross-attention
        attn_out, _ = self.attention(query=query, key=key, value=value)
        
        # Residual connection and normalization
        out = self.norm(query + self.dropout(attn_out))
        
        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Input tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :] 