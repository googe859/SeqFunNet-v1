"""
Tests for MixerEncoding models.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models import SeqFunNet, MLPMixerModel, TransformerOnly, create_ablation_model
# 删除参数量相关的导入


class TestModels:
    """Test cases for models."""
    
    @pytest.fixture
    def dummy_input(self):
        """Create dummy input tensor."""
        return torch.randn(2, 1, 60, 120)
    
    @pytest.fixture
    def dummy_targets(self):
        """Create dummy target tensor."""
        return torch.randint(0, 6, (2,))
    
    def test_seq_fun_net(self, dummy_input, dummy_targets):
        """Test SeqFunNet model."""
        model = SeqFunNet(num_classes=6, depth=4, patch_size=15)
        
        # Test forward pass
        logits, probs = model(dummy_input)
        
        assert logits.shape == (2, 6)
        assert probs.shape == (2, 6)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
        
        # Test feature extraction
        features = model.extract_features(dummy_input)
        assert features.shape == (2, 6)
    
    def test_mlp_mixer(self, dummy_input, dummy_targets):
        """Test MLPMixer model."""
        model = MLPMixerModel(num_classes=6, depth=4, patch_size=15)
        
        # Test forward pass
        logits, probs = model(dummy_input)
        
        assert logits.shape == (2, 6)
        assert probs.shape == (2, 6)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    
    def test_transformer_only(self, dummy_input, dummy_targets):
        """Test TransformerOnly model."""
        model = TransformerOnly(num_classes=6, d_model=60, nhead=5)
        
        # Test forward pass
        logits, probs = model(dummy_input)
        
        assert logits.shape == (2, 6)
        assert probs.shape == (2, 6)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    
    def test_ablation_models(self, dummy_input):
        """Test ablation models."""
        ablation_configs = [
            "full",
            "no_transformer",
            "no_mixer",
            "no_gam",
            "no_mlp"
        ]
        
        for config_name in ablation_configs:
            model = create_ablation_model(config_name, num_classes=6)
            
            # Test forward pass
            logits, probs = model(dummy_input)
            
            assert logits.shape == (2, 6)
            assert probs.shape == (2, 6)
            
            # Test ablation info
            info = model.get_ablation_info()
            assert "model_name" in info
            assert "use_transformer" in info
            assert "use_mixer" in info
            assert "use_gam" in info
            assert "use_mlp" in info
    
    def test_different_input_shapes(self):
        """Test models with different input shapes."""
        model = SeqFunNet(num_classes=6, depth=4, patch_size=15)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 1, 60, 120)
            logits, probs = model(x)
            assert logits.shape == (batch_size, 6)
            assert probs.shape == (batch_size, 6)
    
    def test_model_training_mode(self, dummy_input, dummy_targets):
        """Test model in training mode."""
        model = SeqFunNet(num_classes=6, depth=4, patch_size=15)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits, probs = model(dummy_input)
        loss = criterion(logits, dummy_targets)
        
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert logits.shape == (2, 6)
    
    def test_model_eval_mode(self, dummy_input):
        """Test model in evaluation mode."""
        model = SeqFunNet(num_classes=6, depth=4, patch_size=15)
        
        model.eval()
        with torch.no_grad():
            logits, probs = model(dummy_input)
            
            assert logits.shape == (2, 6)
            assert probs.shape == (2, 6)
            assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        params = count_parameters(model)
        assert params == 55  # 10*5 + 5 (weights + bias)
    
    def test_get_model_size_mb(self):
        """Test model size calculation."""
        model = torch.nn.Linear(1000, 1000)
        size_mb = get_model_size_mb(model)
        assert size_mb > 0
        assert isinstance(size_mb, float)


if __name__ == "__main__":
    pytest.main([__file__]) 