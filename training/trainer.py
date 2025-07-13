"""
Trainer class for model training and evaluation.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logger import get_logger
from ..utils.device import get_device
from ..evaluation.metrics import calculate_metrics
from ..evaluation.visualization import plot_training_curves, plot_confusion_matrix

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class for model training and evaluation.
    
    This class handles the complete training pipeline including:
    - Training loop with validation
    - Model checkpointing
    - Learning rate scheduling
    - Early stopping
    - Metrics tracking
    - Visualization
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        save_dir: str = "results",
        experiment_name: str = "experiment",
        **kwargs
    ):
        """
        Initialize Trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            criterion: Loss function
            device: Device to train on
            save_dir: Directory to save results
            experiment_name: Name of the experiment
            **kwargs: Additional arguments
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device or get_device()
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        
        # Training parameters
        self.epochs = kwargs.get('epochs', 100)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.save_interval = kwargs.get('save_interval', 10)
        self.log_interval = kwargs.get('log_interval', 10)
        self.eval_interval = kwargs.get('eval_interval', 1)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Setup
        self.model.to(self.device)
        self._setup_save_dir()
        self._setup_defaults()
    
    def _setup_save_dir(self):
        """Setup save directory."""
        self.experiment_dir = os.path.join(self.save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = os.path.join(self.experiment_dir, "models")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def _setup_defaults(self):
        """Setup default optimizer, scheduler, and criterion if not provided."""
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum().item()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing validation metrics
        """
        if self.val_loader is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        metrics = calculate_metrics(all_targets, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            **metrics
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Save directory: {self.experiment_dir}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.eval_interval == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Save checkpoint
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if self._should_stop_early(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        self.save_checkpoint(self.epochs, train_metrics, val_metrics, is_final=True)
        
        # Generate plots
        self._generate_training_plots()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
    
    def _should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_metrics: Validation metrics
            
        Returns:
            True if training should stop
        """
        if self.early_stopping_patience <= 0:
            return False
        
        current_val_loss = val_metrics['loss']
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = len(self.train_losses)
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience
    
    def save_checkpoint(self, epoch: int, train_metrics: Dict[str, float], 
                       val_metrics: Dict[str, float], is_final: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            is_final: Whether this is the final checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'learning_rates': self.learning_rates
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.models_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_metrics['accuracy'] > self.best_val_accuracy:
            self.best_val_accuracy = val_metrics['accuracy']
            best_model_path = os.path.join(self.models_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved with validation accuracy: {self.best_val_accuracy:.2f}%")
        
        # Save final model
        if is_final:
            final_model_path = os.path.join(self.models_dir, "final_model.pth")
            torch.save(checkpoint, final_model_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training history
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
            self.train_accuracies = history.get('train_accuracies', [])
            self.val_accuracies = history.get('val_accuracies', [])
            self.learning_rates = history.get('learning_rates', [])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader (uses self.test_loader if None)
            
        Returns:
            Dictionary containing test metrics
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        if test_loader is None:
            logger.warning("No test loader provided")
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output, probabilities = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(all_targets, all_predictions, all_probabilities)
        metrics.update({
            'loss': avg_loss,
            'accuracy': accuracy
        })
        
        # Save results
        self._save_evaluation_results(metrics, all_targets, all_predictions, all_probabilities)
        
        logger.info(f"Test Results: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return metrics
    
    def _save_evaluation_results(self, metrics: Dict[str, float], targets: List[int], 
                                predictions: List[int], probabilities: List[np.ndarray]):
        """Save evaluation results to file."""
        results_path = os.path.join(self.logs_dir, "test_results.txt")
        
        with open(results_path, 'w') as f:
            f.write("Test Results\n")
            f.write("=" * 50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        # Save predictions
        predictions_path = os.path.join(self.logs_dir, "test_predictions.npz")
        np.savez(predictions_path, 
                targets=targets, 
                predictions=predictions, 
                probabilities=probabilities)
    
    def _generate_training_plots(self):
        """Generate training plots."""
        if not self.train_losses:
            return
        
        # Training curves
        plot_training_curves(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            train_accuracies=self.train_accuracies,
            val_accuracies=self.val_accuracies,
            learning_rates=self.learning_rates,
            save_path=os.path.join(self.plots_dir, "training_curves.png")
        )
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = f"Model: {self.model.__class__.__name__}\n"
        summary += f"Total parameters: {total_params:,}\n"
        summary += f"Trainable parameters: {trainable_params:,}\n"
        summary += f"Device: {self.device}\n"
        
        return summary 