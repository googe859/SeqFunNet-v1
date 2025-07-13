"""
Basic usage example for MixerEncoding.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models import SeqFunNet, MLPMixerModel, TransformerOnly, create_ablation_model
# 删除参数量相关的导入

def example_1_basic_models():
    print("=" * 50)
    print("Example 1: Basic Model Usage")
    print("=" * 50)
    batch_size = 4
    x = torch.randn(batch_size, 1, 60, 120)
    print("\n1. SeqFunNet Model:")
    seq_fun_net = SeqFunNet(num_classes=6, depth=4, patch_size=15)
    logits, probs = seq_fun_net(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Output probabilities shape: {probs.shape}")
    print("\n2. MLP-Mixer Model:")
    mlp_mixer = MLPMixerModel(num_classes=6, depth=4, patch_size=15)
    logits, probs = mlp_mixer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Output probabilities shape: {probs.shape}")
    print("\n3. Transformer Only Model:")
    transformer = TransformerOnly(num_classes=6, d_model=60, nhead=5)
    logits, probs = transformer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Output probabilities shape: {probs.shape}")

def example_2_ablation_study():
    print("\n" + "=" * 50)
    print("Example 2: Ablation Study Models")
    print("=" * 50)
    x = torch.randn(2, 1, 60, 120)
    ablation_configs = ["full", "no_transformer", "no_mixer", "no_gam", "no_mlp"]
    for config_name in ablation_configs:
        print(f"\n{config_name.upper().replace('_', ' ')}:")
        model = create_ablation_model(config_name, num_classes=6)
        info = model.get_ablation_info()
        print(f"   Model name: {info['model_name']}")
        print(f"   Components: Transformer={info['use_transformer']}, "
              f"Mixer={info['use_mixer']}, GAM={info['use_gam']}, MLP={info['use_mlp']}")
        print(f"   Fusion strategy: {info['fusion_strategy']}")
        print(f"   Final dimension: {info['final_dim']}")
        logits, probs = model(x)
        print(f"   Output shape: {logits.shape}")

def example_3_feature_extraction():
    print("\n" + "=" * 50)
    print("Example 3: Feature Extraction")
    print("=" * 50)
    model = SeqFunNet(num_classes=6, depth=4, patch_size=15)
    x = torch.randn(1, 1, 60, 120)
    print("\nFeature extraction from SeqFunNet:")
    features = model.extract_features(x)
    print(f"   Final features shape: {features.shape}")
    transformer_features = model.extract_features(x, layer_name="transformer")
    print(f"   Transformer features shape: {transformer_features.shape}")
    attention_features = model.extract_features(x, layer_name="attention")
    print(f"   Attention features shape: {attention_features.shape}")

def example_4_model_comparison():
    print("\n" + "=" * 50)
    print("Example 4: Model Comparison")
    print("=" * 50)
    models = {
        "SeqFunNet": SeqFunNet(num_classes=6, depth=4, patch_size=15),
        "MLPMixer": MLPMixerModel(num_classes=6, depth=4, patch_size=15),
        "TransformerOnly": TransformerOnly(num_classes=6, d_model=60, nhead=5)
    }
    x = torch.randn(1, 1, 60, 120)
    print("\nModel Comparison:")
    print(f"{'Model':<15} {'Output Shape'}")
    print("-" * 30)
    for name, model in models.items():
        with torch.no_grad():
            if hasattr(model, 'forward'):
                output = model(x)
                if isinstance(output, tuple):
                    output_shape = output[0].shape
                else:
                    output_shape = output.shape
            else:
                output_shape = "N/A"
        print(f"{name:<15} {output_shape}")

def example_5_training_setup():
    print("\n" + "=" * 50)
    print("Example 5: Training Setup")
    print("=" * 50)
    model = SeqFunNet(num_classes=6, depth=4, patch_size=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 4
    x = torch.randn(batch_size, 1, 60, 120)
    y = torch.randint(0, 6, (batch_size,))
    print("\nTraining setup:")
    print(f"   Model: {type(model).__name__}")
    print(f"   Optimizer: {type(optimizer).__name__}")
    print(f"   Loss function: {type(criterion).__name__}")
    print(f"   Input shape: {x.shape}")
    print(f"   Target shape: {y.shape}")
    model.train()
    optimizer.zero_grad()
    logits, probs = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Predictions shape: {logits.shape}")
    print(f"   Probabilities shape: {probs.shape}")

def main():
    print("MixerEncoding - Basic Usage Examples")
    print("=" * 60)
    try:
        example_1_basic_models()
        example_2_ablation_study()
        example_3_feature_extraction()
        example_4_model_comparison()
        example_5_training_setup()
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error in examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 