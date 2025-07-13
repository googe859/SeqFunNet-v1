#!/usr/bin/env python3
"""
Ablation study script for SeqFunNet.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models import create_ablation_configs

def run_ablation_study(dataset="bird_6", epochs=50, batch_size=32):
    """
    Run complete ablation study.
    
    Args:
        dataset: Dataset name
        epochs: Number of epochs per experiment
        batch_size: Batch size
    """
    ablation_configs = [
        "full",
        "no_transformer", 
        "no_mixer",
        "no_gam",
        "no_mlp",
        "transformer_only",
        "mixer_only",
        "concat_fusion",
        "weighted_fusion"
    ]
    
    results = {}
    
    for config_name in ablation_configs:
        print(f"\n{'='*50}")
        print(f"Running ablation: {config_name}")
        print(f"{'='*50}")
        
        # Create experiment name
        experiment_name = f"ablation_{config_name}_{dataset}"
        
        # Build command
        cmd = [
            sys.executable, "-m", "src.experiments.train",
            "--model", f"ablation_{config_name}",
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--experiment_name", experiment_name
        ]
        
        try:
            # Run training
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = time.time()
            
            # Parse results (this is simplified - you might want to parse log files)
            results[config_name] = {
                "status": "success",
                "time": end_time - start_time,
                "output": result.stdout
            }
            
            print(f"✓ {config_name} completed in {end_time - start_time:.1f}s")
            
        except subprocess.CalledProcessError as e:
            results[config_name] = {
                "status": "failed",
                "error": e.stderr,
                "output": e.stdout
            }
            print(f"✗ {config_name} failed: {e.stderr}")
    
    # Save results summary
    save_ablation_summary(results, dataset)
    
    return results

def save_ablation_summary(results, dataset):
    """
    Save ablation study summary.
    
    Args:
        results: Results dictionary
        dataset: Dataset name
    """
    summary_path = Path("results") / f"ablation_summary_{dataset}.txt"
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, "w") as f:
        f.write(f"Ablation Study Summary - {dataset}\n")
        f.write("=" * 50 + "\n\n")
        
        for config_name, result in results.items():
            f.write(f"Configuration: {config_name}\n")
            f.write(f"Status: {result['status']}\n")
            
            if result['status'] == "success":
                f.write(f"Time: {result['time']:.1f}s\n")
            else:
                f.write(f"Error: {result['error']}\n")
            
            f.write("-" * 30 + "\n")
    
    print(f"\nAblation summary saved to: {summary_path}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SeqFunNet ablation study")
    parser.add_argument("--dataset", type=str, default="bird_6", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--config", type=str, default=None, help="Single ablation config to run")
    
    args = parser.parse_args()
    
    if args.config:
        # Run single ablation
        print(f"Running single ablation: {args.config}")
        cmd = [
            sys.executable, "-m", "src.experiments.train",
            "--model", f"ablation_{args.config}",
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--experiment_name", f"ablation_{args.config}_{args.dataset}"
        ]
        
        subprocess.run(cmd)
    else:
        # Run complete ablation study
        results = run_ablation_study(args.dataset, args.epochs, args.batch_size)
        
        # Print summary
        print("\n" + "="*50)
        print("ABLATION STUDY SUMMARY")
        print("="*50)
        
        for config_name, result in results.items():
            status = "✓" if result['status'] == "success" else "✗"
            print(f"{status} {config_name}: {result['status']}")

if __name__ == "__main__":
    main() 