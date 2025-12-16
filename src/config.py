"""
Configuration classes for training, data, and system settings.
Automatically detects Kaggle environment and adjusts paths accordingly.
"""

import os
from dataclasses import dataclass
import torch


def is_kaggle_environment() -> bool:
    """
    Detect if code is running in Kaggle environment.

    Returns:
        bool: True if running in Kaggle, False otherwise
    """
    return os.path.exists('/kaggle/input')


def get_environment_name() -> str:
    """
    Get the current environment name.

    Returns:
        str: 'kaggle' or 'local'
    """
    return 'kaggle' if is_kaggle_environment() else 'local'


@dataclass
class TrainingConfiguration:
    """
    Configuration for the training process.

    Attributes:
        batch_size: Number of samples per batch
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        momentum: Momentum factor for SGD optimizer
        log_interval: Logging frequency (steps)
        random_seed: Random seed for reproducibility
        optimizer: Optimizer type ('sgd', 'adam', 'adamw')
        weight_decay: L2 regularization factor
        use_scheduler: Whether to use learning rate scheduler
        scheduler: Scheduler type ('step', 'cosine', 'reduce_on_plateau')
        lr_step_size: Step size for StepLR scheduler
        lr_gamma: Multiplicative factor for learning rate decay
        model_name: Base model architecture
        pretrained: Whether to use pretrained weights
        precision: Training precision ('float32', 'float16', 'bfloat16')
        fine_tune_start: Layer index to start fine-tuning from
        freeze_pct: Percentage of parameters to freeze (0.0-1.0, default 0.6)
    """
    batch_size: int = 16  # OPTIMIZED: Reduced for memory efficiency with freeze_pct=0.0 (all layers trainable)
    learning_rate: float = 0.00007  # FINE-TUNED: Reduced for smoother convergence (was 0.0001, caused oscillation)
    # HISTORY: 0.001 → 0.0001 (stable) → 0.0003 (for freeze_pct=0.2) → 0.0001 (for freeze_pct=0.0) → 0.00007 (for stability)
    # Use 0.0003 with freeze_pct=0.2, use 0.0001 with freeze_pct=0.0, use 0.00007 for smoothest convergence
    # ORIGINAL: learning_rate: float = 0.001
    num_epochs: int = 10
    momentum: float = 0.9
    log_interval: int = 10
    random_seed: int = 42
    freeze_pct: float = 0.0  # RECOMMENDED: Unfreeze all layers for best accuracy (was 0.2)
    # HISTORY: 0.6 → 0.2 (good for 82% acc) → 0.0 (target 85%+ acc)
    # Set to 0.2 for faster training, 0.0 for highest accuracy

    # Optimizer configuration
    optimizer: str = "adamw"  # IMPROVED: Changed from 'sgd' - AdamW better for fine-tuning
    # ORIGINAL: optimizer: str = "sgd"  # SGD was too slow for this task
    weight_decay: float = 0.01  # IMPROVED: Increased from 0.0001 for better regularization

    # Learning rate scheduler configuration
    use_scheduler: bool = True
    scheduler: str = "cosine"  # IMPROVED: Changed from 'step' - smoother decay, better for fine-tuning
    # ORIGINAL: scheduler: str = "step"  # StepLR was too aggressive (gamma=0.1)
    lr_step_size: int = 5  # For StepLR: step size for learning rate decay
    lr_gamma: float = 0.1  # For StepLR: multiplicative factor of learning rate decay

    model_name: str = "efficientnetv2"  # CHANGE 3: Changed from 'googlenet' (more powerful model)
    # ORIGINAL: model_name: str = "googlenet"
    pretrained: bool = True  # use pretrained weights for the base model
    precision: str = "float32"  # precision for training: float32, float16, bfloat16
    fine_tune_start: int = 5  # layer from which to start fine-tuning

    # Early stopping configuration
    use_early_stopping: bool = True  # whether to use early stopping
    early_stop_monitor: str = "valid/acc"  # metric to monitor for early stopping
    early_stop_patience: int = 15  # INCREASED: More patience for Phase 2 (RandAugment needs longer convergence)
    # HISTORY: 7 (stopped too early at epoch 20) → 15 (allow more exploration)
    early_stop_mode: str = "max"  # 'min' for loss, 'max' for accuracy


@dataclass
class DataConfiguration:
    """
    Configuration for dataset and data loading.
    Automatically adjusts paths based on environment (local vs Kaggle).

    Attributes:
        annotations_file: Path to train.csv annotation file
        img_dir: Path to images directory
        input_size: Input image size (height/width for square images)
        num_workers: Number of data loading workers
        batch_size: Batch size for data loaders
    """
    annotations_file: str = None
    img_dir: str = None
    input_size: int = 384  # Default for EfficientNetV2 (auto-overridden by model-specific preprocessing)
    # NOTE: When using model_name in DataModule, this is automatically overridden:
    # - ResNet50/GoogleNet: 224x224
    # - EfficientNetV2: 384x384
    num_workers: int = 0  # number of workers for data loading (set to 0 to avoid JAX fork deadlock)
    batch_size: int = 32  # batch size for training and validation

    def __post_init__(self):
        """Set default paths based on environment if not explicitly provided."""
        if self.annotations_file is None:
            if is_kaggle_environment():
                self.annotations_file = "/kaggle/input/opencv-pytorch-project-2-classification-round-3/train.csv"
            else:
                self.annotations_file = "../data/kenyan-food-13/train.csv"

        if self.img_dir is None:
            if is_kaggle_environment():
                self.img_dir = "/kaggle/input/opencv-pytorch-project-2-classification-round-3/images/images"
            else:
                self.img_dir = "../data/kenyan-food-13/images/images"


@dataclass
class SystemConfiguration:
    """
    Configuration for system resources and output paths.

    Attributes:
        device: Computing device ('cuda' or 'cpu')
        output_dir: Directory for saving outputs (checkpoints, logs)
    """
    device: str = None
    output_dir: str = None

    def __post_init__(self):
        """Set default values based on environment if not explicitly provided."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.output_dir is None:
            if is_kaggle_environment():
                self.output_dir = "/kaggle/working/output"
            else:
                self.output_dir = "./output"


def get_config(environment: str = None,
               local_mode: bool = None,
               **overrides):
    """
    Convenience function to get all configurations with optional overrides.

    Args:
        environment: Force environment ('kaggle' or 'local'). If None, auto-detect.
        local_mode: Legacy parameter for backward compatibility
        **overrides: Override specific configuration parameters

    Returns:
        tuple: (TrainingConfiguration, DataConfiguration, SystemConfiguration)

    Example:
        >>> train_cfg, data_cfg, sys_cfg = get_config(num_epochs=20, batch_size=64)
    """
    # Handle legacy local_mode parameter
    if local_mode is not None:
        print("Warning: 'local_mode' parameter is deprecated. Use 'environment' instead.")
        environment = 'local' if local_mode else 'kaggle'

    # Create configurations
    train_config = TrainingConfiguration()
    data_config = DataConfiguration()
    system_config = SystemConfiguration()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(train_config, key):
            setattr(train_config, key, value)
        elif hasattr(data_config, key):
            setattr(data_config, key, value)
        elif hasattr(system_config, key):
            setattr(system_config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter '{key}' ignored.")

    # Log environment detection
    detected_env = get_environment_name()
    if environment and environment != detected_env:
        print(f"Warning: Detected '{detected_env}' environment, but '{environment}' was specified.")

    return train_config, data_config, system_config
