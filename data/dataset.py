"""
Dataset classes for audio classification.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any, List
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AudioDataset(Dataset):
    """
    Dataset class for audio classification.
    
    Supports loading pre-processed audio features (MFCC, Mel-spectrogram, etc.)
    from numpy files with automatic train/val/test splitting.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        feature_type: str = "mfcc",
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize AudioDataset.
        
        Args:
            data_path: Path to the dataset directory
            split: Dataset split ('train', 'val', 'test')
            feature_type: Type of audio features ('mfcc', 'mel', 'spectrogram')
            transform: Optional transform to apply to features
            target_transform: Optional transform to apply to labels
            normalize: Whether to normalize features
            **kwargs: Additional arguments
        """
        self.data_path = data_path
        self.split = split
        self.feature_type = feature_type
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        
        # Load data
        self.features, self.labels = self._load_data()
        
        # Apply transforms
        if self.normalize:
            self._normalize_features()
        
        logger.info(f"Loaded {split} dataset: {len(self.features)} samples")
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels from files.
        
        Returns:
            Tuple of (features, labels)
        """
        # Construct file paths
        base_path = os.path.join(self.data_path, self.feature_type)
        
        # Smart loading with fallback options
        feature_file = self._smart_load_path(base_path, f"{self.split}")
        label_file = self._smart_load_path(base_path, f"{self.split}_label")
        
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        # Load data
        features = np.load(feature_file, allow_pickle=True)
        labels = np.load(label_file, allow_pickle=True)
        
        # Ensure correct shapes
        if features.ndim == 2:
            # Add channel dimension if needed
            features = features.reshape(-1, 1, features.shape[0], features.shape[1])
        elif features.ndim == 3:
            # Add channel dimension
            features = features.reshape(-1, 1, features.shape[1], features.shape[2])
        
        # Ensure labels are 1D
        labels = labels.reshape(-1)
        
        return features, labels
    
    def _smart_load_path(self, base_path: str, name: str) -> str:
        """
        Smart path resolution with fallback options.
        
        Args:
            base_path: Base directory path
            name: File name without extension
            
        Returns:
            Full file path
        """
        # Try different naming conventions
        possible_paths = [
            os.path.join(base_path, f"{name}_{self.feature_type}.npy"),
            os.path.join(base_path, f"{name}.npy"),
            os.path.join(base_path, f"{name}_{self.feature_type}_features.npy"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return the first option as default (will raise error if not found)
        return possible_paths[0]
    
    def _normalize_features(self):
        """Normalize features to zero mean and unit variance."""
        if len(self.features) > 0:
            # Compute statistics across all samples
            features_flat = self.features.reshape(-1, self.features.shape[-2] * self.features.shape[-1])
            mean = np.mean(features_flat, axis=0, keepdims=True)
            std = np.std(features_flat, axis=0, keepdims=True)
            
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            
            # Normalize
            features_normalized = (features_flat - mean) / std
            self.features = features_normalized.reshape(self.features.shape)
            
            logger.info(f"Normalized features: mean={mean.mean():.4f}, std={std.mean():.4f}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        features = self.features[idx]
        label = self.labels[idx]
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label]).squeeze()
        
        # Apply transforms
        if self.transform is not None:
            features = self.transform(features)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return features, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.
        
        Returns:
            Class weights tensor
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        
        # Compute inverse frequency weights
        weights = total_samples / (len(unique_labels) * counts)
        return torch.FloatTensor(weights)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get class distribution.
        
        Returns:
            Dictionary mapping class indices to counts
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique_labels, counts))


class AudioDataModule:
    """
    Data module for managing train/val/test splits and dataloaders.
    """
    
    def __init__(
        self,
        data_path: str,
        feature_type: str = "mfcc",
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize AudioDataModule.
        
        Args:
            data_path: Path to the dataset directory
            feature_type: Type of audio features
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            shuffle: Whether to shuffle training data
            drop_last: Whether to drop last incomplete batch
            transform: Optional transform to apply to features
            target_transform: Optional transform to apply to labels
            normalize: Whether to normalize features
            **kwargs: Additional arguments
        """
        self.data_path = data_path
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize dataloaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self._setup_datasets()
        self._setup_dataloaders()
    
    def _setup_datasets(self):
        """Setup train, validation, and test datasets."""
        try:
            self.train_dataset = AudioDataset(
                data_path=self.data_path,
                split="train",
                feature_type=self.feature_type,
                transform=self.transform,
                target_transform=self.target_transform,
                normalize=self.normalize
            )
        except FileNotFoundError:
            logger.warning("Train dataset not found")
        
        try:
            self.val_dataset = AudioDataset(
                data_path=self.data_path,
                split="val",
                feature_type=self.feature_type,
                transform=self.transform,
                target_transform=self.target_transform,
                normalize=self.normalize
            )
        except FileNotFoundError:
            logger.warning("Validation dataset not found")
        
        try:
            self.test_dataset = AudioDataset(
                data_path=self.data_path,
                split="test",
                feature_type=self.feature_type,
                transform=self.transform,
                target_transform=self.target_transform,
                normalize=self.normalize
            )
        except FileNotFoundError:
            logger.warning("Test dataset not found")
    
    def _setup_dataloaders(self):
        """Setup dataloaders for all splits."""
        if self.train_dataset is not None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=True
            )
        
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True
            )
        
        if self.test_dataset is not None:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True
            )
    
    def get_dataloaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """
        Get train, validation, and test dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        Get class weights from training dataset.
        
        Returns:
            Class weights tensor or None if training dataset not available
        """
        if self.train_dataset is not None:
            return self.train_dataset.get_class_weights()
        return None
    
    def get_num_classes(self) -> int:
        """
        Get number of classes from training dataset.
        
        Returns:
            Number of classes
        """
        if self.train_dataset is not None:
            return len(np.unique(self.train_dataset.labels))
        elif self.val_dataset is not None:
            return len(np.unique(self.val_dataset.labels))
        elif self.test_dataset is not None:
            return len(np.unique(self.test_dataset.labels))
        else:
            raise ValueError("No dataset available to determine number of classes")
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get input shape from training dataset.
        
        Returns:
            Input shape tuple
        """
        if self.train_dataset is not None:
            sample_features, _ = self.train_dataset[0]
            return sample_features.shape
        else:
            raise ValueError("No training dataset available to determine input shape")
    
    def __str__(self) -> str:
        """String representation of the data module."""
        info = f"AudioDataModule(data_path={self.data_path}, feature_type={self.feature_type})"
        if self.train_dataset:
            info += f"\n  Train: {len(self.train_dataset)} samples"
        if self.val_dataset:
            info += f"\n  Val: {len(self.val_dataset)} samples"
        if self.test_dataset:
            info += f"\n  Test: {len(self.test_dataset)} samples"
        return info 