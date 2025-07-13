#!/usr/bin/env python3
"""
Main training script for MixerEncoding.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import SeqFunNet, MLPMixerModel, TransformerOnly, create_ablation_model
from data import AudioDataModule
from training import Trainer
from utils.logger import setup_logging, get_logger
from utils.device import get_device
from utils.helpers import set_seed

logger = get_logger(__name__)


def create_model(model_name: str, num_classes: int, **kwargs):
    """
    Create model based on name.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if model_name == "SeqFunNet":
        return SeqFunNet(num_classes=num_classes, **kwargs)
    elif model_name == "MLPMixer":
        return MLPMixerModel(num_classes=num_classes, **kwargs)
    elif model_name == "TransformerOnly":
        return TransformerOnly(num_classes=num_classes, **kwargs)
    elif model_name.startswith("ablation_"):
        config_name = model_name.replace("ablation_", "")
        return create_ablation_model(config_name, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MixerEncoding models")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="SeqFunNet", 
                       choices=["SeqFunNet", "MLPMixer", "TransformerOnly"] + 
                               [f"ablation_{config}" for config in ["full", "no_transformer", "no_mixer", "no_gam", "no_mlp", "transformer_only", "mixer_only", "concat_fusion", "weighted_fusion"]],
                       help="Model to train")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--depth", type=int, default=4, help="MLP-Mixer depth")
    parser.add_argument("--patch_size", type=int, default=15, help="MLP-Mixer patch size")
    parser.add_argument("--dim", type=int, default=30, help="MLP-Mixer dimension")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="bird_6", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="dataset_npy", help="Data path")
    parser.add_argument("--feature_type", type=str, default="mfcc", help="Feature type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler")
    
    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--save_dir", type=str, default="results", help="Save directory")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Setup logging
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.dataset}_{args.depth}_{args.patch_size}"
    
    log_file = os.path.join(args.save_dir, args.experiment_name, "training.log")
    setup_logging(level=getattr(logging, args.log_level), log_file=log_file)
    
    logger.info(f"Starting training with model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Experiment: {args.experiment_name}")
    
    # Create data module
    data_path = os.path.join(args.data_path, args.dataset)
    data_module = AudioDataModule(
        data_path=data_path,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    num_classes = data_module.get_num_classes()
    
    logger.info(f"Dataset: {args.dataset}, Classes: {num_classes}")
    logger.info(f"Train samples: {len(train_loader.dataset) if train_loader else 0}")
    logger.info(f"Val samples: {len(val_loader.dataset) if val_loader else 0}")
    logger.info(f"Test samples: {len(test_loader.dataset) if test_loader else 0}")
    
    # Create model
    model_kwargs = {
        "depth": args.depth,
        "patch_size": args.patch_size,
        "dim": args.dim,
    }
    
    model = create_model(args.model, num_classes, **model_kwargs)
    model.to(device)
    
    logger.info(f"Model created: {model}")
    
    # Create optimizer
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create scheduler
    if args.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler.lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        scheduler = None
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        epochs=args.epochs
    )
    
    # Train
    history = trainer.train()
    
    # Evaluate
    if test_loader:
        test_metrics = trainer.evaluate()
        logger.info(f"Final test accuracy: {test_metrics['accuracy']:.2f}%")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 