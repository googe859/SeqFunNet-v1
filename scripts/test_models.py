#!/usr/bin/env python3
"""
Test script to verify all models work correctly.
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models import SeqFunNet, MLPMixerModel, TransformerOnly, create_ablation_model
# 删除参数量相关的导入

def test_model(model, model_name, input_shape=(1, 1, 60, 120)):
    print(f"\nTesting {model_name}...")
    x = torch.randn(input_shape)
    # 删除参数量相关输出
    try:
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward'):
                output = model(x)
                if isinstance(output, tuple):
                    logits, probs = output
                else:
                    logits = output
                    probs = None
                print(f"  Output shape: {logits.shape}")
                if probs is not None:
                    print(f"  Probabilities shape: {probs.shape}")
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(x)
                    print(f"  Features shape: {features.shape}")
                print(f"  ✓ {model_name} works correctly!")
                return True
            else:
                print(f"  ✗ {model_name} has no forward method")
                return False
    except Exception as e:
        print(f"  ✗ {model_name} failed: {str(e)}")
        return False

def main():
    print("Testing MixerEncoding Models")
    print("=" * 40)
    test_configs = [
        {"name": "SeqFunNet", "model_class": SeqFunNet, "kwargs": {"num_classes": 6, "depth": 4, "patch_size": 15}},
        {"name": "MLPMixer", "model_class": MLPMixerModel, "kwargs": {"num_classes": 6, "depth": 4, "patch_size": 15}},
        {"name": "TransformerOnly", "model_class": TransformerOnly, "kwargs": {"num_classes": 6, "d_model": 60, "nhead": 5}}
    ]
    ablation_configs = ["full", "no_transformer", "no_mixer", "no_gam", "no_mlp", "transformer_only", "mixer_only", "concat_fusion", "weighted_fusion"]
    results = {}
    for config in test_configs:
        model = config["model_class"](**config["kwargs"])
        success = test_model(model, config["name"])
        results[config["name"]] = success
    for config_name in ablation_configs:
        try:
            model = create_ablation_model(config_name, num_classes=6)
            success = test_model(model, f"Ablation_{config_name}")
            results[f"Ablation_{config_name}"] = success
        except Exception as e:
            print(f"\n✗ Ablation_{config_name} failed to create: {str(e)}")
            results[f"Ablation_{config_name}"] = False
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    passed = 0
    total = len(results)
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {model_name}")
        if success:
            passed += 1
    print(f"\nPassed: {passed}/{total}")
    if passed == total:
        print("🎉 All models work correctly!")
    else:
        print("⚠️  Some models have issues. Check the output above.")
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 