"""
Configuration management utilities.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    Configuration manager for MixerEncoding.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        file_ext = Path(config_path).suffix.lower()
        
        if file_ext == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
        
        return self.config
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration (optional)
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ValueError("No config path specified")
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        file_ext = Path(config_path).suffix.lower()
        
        if file_ext == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config.update(config_dict)
    
    def merge(self, other_config: Dict[str, Any]) -> None:
        """
        Merge configuration with another dictionary.
        
        Args:
            other_config: Configuration to merge
        """
        self._merge_dict(self.config, other_config)
    
    def _merge_dict(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Recursively merge dictionaries.
        
        Args:
            base: Base dictionary
            update: Update dictionary
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dict(base[key], value)
            else:
                base[key] = value
    
    def validate(self, required_keys: list) -> bool:
        """
        Validate configuration has required keys.
        
        Args:
            required_keys: List of required keys
            
        Returns:
            True if all required keys are present
        """
        missing_keys = []
        
        for key in required_keys:
            if key not in self.config:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    manager = ConfigManager(filepath)
    return manager.to_dict()


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save file
    """
    manager = ConfigManager()
    manager.update(config)
    manager.save_config(filepath)


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "model": {
            "name": "SeqFunNet",
            "num_classes": 6,
            "depth": 4,
            "patch_size": 15,
            "dim": 30,
            "expansion_factor": 4,
            "dropout": 0.1
        },
        "data": {
            "dataset": "bird_6",
            "data_path": "dataset_npy",
            "feature_type": "mfcc",
            "batch_size": 32,
            "num_workers": 4,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15
        },
        "training": {
            "epochs": 100,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adam",
            "scheduler": "cosine",
            "early_stopping_patience": 10,
            "save_best_only": True
        },
        "experiment": {
            "name": None,
            "save_dir": "results",
            "device": "auto",
            "seed": 42,
            "use_wandb": False
        },
        "logging": {
            "level": "INFO",
            "save_logs": True,
            "log_interval": 10
        }
    }


def create_model_config(model_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create model-specific configuration.
    
    Args:
        model_name: Name of the model
        **kwargs: Model parameters
        
    Returns:
        Model configuration
    """
    base_config = create_default_config()
    
    if model_name == "SeqFunNet":
        base_config["model"].update({
            "name": "SeqFunNet",
            "use_transformer": True,
            "use_mixer": True,
            "use_gam": True,
            "use_mlp": True,
            "fusion_strategy": "add"
        })
    elif model_name == "MLPMixer":
        base_config["model"].update({
            "name": "MLPMixer",
            "image_size": 60,
            "patch_size": 15,
            "dim": 30,
            "depth": 4,
            "expansion_factor": 4,
            "dropout": 0.1
        })
    elif model_name == "TransformerOnly":
        base_config["model"].update({
            "name": "TransformerOnly",
            "d_model": 60,
            "nhead": 5,
            "num_layers": 2,
            "dim_feedforward": 64,
            "dropout": 0.5
        })
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Update with provided kwargs
    base_config["model"].update(kwargs)
    
    return base_config


def create_ablation_config(ablation_type: str, **kwargs) -> Dict[str, Any]:
    """
    Create ablation study configuration.
    
    Args:
        ablation_type: Type of ablation
        **kwargs: Additional parameters
        
    Returns:
        Ablation configuration
    """
    base_config = create_model_config("SeqFunNet")
    
    if ablation_type == "no_transformer":
        base_config["model"]["use_transformer"] = False
    elif ablation_type == "no_mixer":
        base_config["model"]["use_mixer"] = False
    elif ablation_type == "no_gam":
        base_config["model"]["use_gam"] = False
    elif ablation_type == "no_mlp":
        base_config["model"]["use_mlp"] = False
    elif ablation_type == "concat_fusion":
        base_config["model"]["fusion_strategy"] = "concat"
    elif ablation_type == "weighted_fusion":
        base_config["model"]["fusion_strategy"] = "weighted"
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    # Update with provided kwargs
    base_config["model"].update(kwargs)
    
    return base_config 