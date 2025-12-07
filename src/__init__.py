"""
Kenyan Food 13 Classification - Modular Source Code

This package contains modular implementations for the Kaggle competition.
All modules support seamless execution on local machines and Kaggle environments.

Quick Start:
    >>> from src.config import get_config
    >>> from src.datamodule import KenyanFood13DataModule
    >>> from src.model import KenyanFood13Classifier
    >>> from src.trainer import train_model
    >>>
    >>> # Get configurations (auto-detects environment)
    >>> train_cfg, data_cfg, sys_cfg = get_config()
    >>>
    >>> # Create data module
    >>> data_module = KenyanFood13DataModule(data_cfg)
    >>> data_module.setup()
    >>>
    >>> # Create model
    >>> model = KenyanFood13Classifier(train_cfg, data_module.num_classes)
    >>>
    >>> # Train
    >>> train_model(train_cfg, data_cfg, sys_cfg, model, data_module)
"""

__version__ = "2.0.0"  # Phase 2: Full modular architecture

# Expose main classes for easier imports
from .config import (
    TrainingConfiguration,
    DataConfiguration,
    SystemConfiguration,
    get_config,
    is_kaggle_environment,
    get_environment_name
)

from .dataset import KenyanFood13Dataset
from .datamodule import KenyanFood13DataModule
from .model import KenyanFood13Classifier
from .trainer import train_model
from .utils import calculate_dataset_mean_std, get_imagenet_stats

__all__ = [
    # Configuration
    'TrainingConfiguration',
    'DataConfiguration',
    'SystemConfiguration',
    'get_config',
    'is_kaggle_environment',
    'get_environment_name',
    # Data
    'KenyanFood13Dataset',
    'KenyanFood13DataModule',
    # Model
    'KenyanFood13Classifier',
    # Training
    'train_model',
    # Utils
    'calculate_dataset_mean_std',
    'get_imagenet_stats',
]
