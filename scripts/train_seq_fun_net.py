#!/usr/bin/env python3
"""
Training script for SeqFunNet model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.experiments.train import main

if __name__ == "__main__":
    # Override default arguments for SeqFunNet
    import argparse
    parser = argparse.ArgumentParser(description="Train SeqFunNet")
    
    parser.add_argument("--dataset", type=str, default="bird_6", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--depth", type=int, default=4, help="MLP-Mixer depth")
    parser.add_argument("--patch_size", type=int, default=15, help="Patch size")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    
    args = parser.parse_args()
    
    # Set model to SeqFunNet
    sys.argv = [
        "train.py",
        "--model", "SeqFunNet",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--depth", str(args.depth),
        "--patch_size", str(args.patch_size)
    ]
    
    if args.experiment_name:
        sys.argv.extend(["--experiment_name", args.experiment_name])
    
    main() 