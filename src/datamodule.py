"""
PyTorch Lightning DataModule for Kenyan Food 13 Classification.
Manages dataset creation, transforms, and dataloaders.
"""

import os
import pytorch_lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import KenyanFood13Dataset
from .config import is_kaggle_environment


class KenyanFood13DataModule(L.LightningDataModule):
    """
    LightningDataModule for Kenyan Food 13 dataset.

    Handles data loading, transformation pipelines, and train/validation split.
    Automatically applies data augmentation to training set.

    Args:
        data_config (DataConfiguration): Configuration object with dataset parameters
        model_name (str, optional): Model name to use for preprocessing ('resnet50', 'efficientnetv2', 'googlenet')
            If provided, uses model-specific preprocessing parameters.
        mean (list, optional): RGB channel means for normalization.
            If None and model_name is None, uses ImageNet defaults [0.485, 0.456, 0.406]
        std (list, optional): RGB channel standard deviations for normalization.
            If None and model_name is None, uses ImageNet defaults [0.229, 0.224, 0.225]

    Attributes:
        num_classes (int): Number of classes (available after calling setup())
        train_dataset (Dataset): Training dataset (available after setup())
        val_dataset (Dataset): Validation dataset (available after setup())

    Example:
        >>> from src.config import DataConfiguration
        >>> data_config = DataConfiguration()
        >>> data_module = KenyanFood13DataModule(data_config, model_name='efficientnetv2')
        >>> data_module.setup()
        >>> train_loader = data_module.train_dataloader()
    """

    def __init__(self, data_config, model_name=None, mean=None, std=None):
        super().__init__()
        self.data_config = data_config
        self.model_name = model_name

        # Handle local mode annotation file naming
        local_mode = not is_kaggle_environment()
        # if local_mode:
        #     print("Running in local mode - loading data from local paths")
        #     base, ext = os.path.splitext(self.data_config.annotations_file)
        #     self.data_config.annotations_file = f"{base}_local{ext}"

        # Validate annotation file exists
        if not os.path.exists(self.data_config.annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {self.data_config.annotations_file}")

        self.train_dataset = None
        self.val_dataset = None
        self.local_mode = local_mode

        # Get model-specific preprocessing if model_name is provided
        if model_name:
            from .model import KenyanFood13Classifier
            _, input_size, model_mean, model_std = KenyanFood13Classifier.get_model_preprocessing(model_name)
            # Override data_config input_size if using model-specific preprocessing
            if self.data_config.input_size != input_size:
                print(f"[INFO] Using model-specific input size: {input_size} (was {self.data_config.input_size})")
                self.data_config.input_size = input_size
            self.mean = mean if mean is not None else model_mean
            self.std = std if std is not None else model_std
            print(f"[INFO] Using model-specific preprocessing: mean={self.mean}, std={self.std}")
        else:
            # Mean and Std for normalization - use provided values or defaults (ImageNet stats)
            self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
            self.std = std if std is not None else [0.229, 0.224, 0.225]

    def setup(self, stage=None):
        """
        Setup datasets with appropriate transforms.

        Args:
            stage (str, optional): 'fit', 'validate', 'test', or 'predict'.
                If None, sets up all datasets.

        Note:
            Training set uses augmentation transforms, validation uses basic transforms.
        """
        # Parse input size
        if isinstance(self.data_config.input_size, int):
            img_height = self.data_config.input_size
            img_width = self.data_config.input_size
        elif isinstance(self.data_config.input_size, tuple) and len(self.data_config.input_size) == 2:
            img_height = self.data_config.input_size[0]
            img_width = self.data_config.input_size[1]
        else:
            raise ValueError("input_size must be an int or a tuple of two ints (height, width)")

        # Common transforms for validation (no augmentation)
        common_transforms = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Augmentation transforms for training
        # PHASE 2: Added RandAugment for stronger augmentation
        aug_transforms = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            # RandAugment: Randomly applies augmentations (rotation, color, sharpness, etc.)
            # num_ops=2: apply 2 random operations per image
            # magnitude=9: strength of augmentations (0-30, 9 is moderate-strong)
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Create datasets
        self.train_dataset = KenyanFood13Dataset(
            annotations_file=self.data_config.annotations_file,
            img_dir=self.data_config.img_dir,
            train=True,
            transform=aug_transforms,
            local_mode=self.local_mode
        )

        self.val_dataset = KenyanFood13Dataset(
            annotations_file=self.data_config.annotations_file,
            img_dir=self.data_config.img_dir,
            train=False,
            transform=common_transforms,
            local_mode=self.local_mode
        )

        # Store number of classes
        self.num_classes = self.train_dataset.num_classes

    def train_dataloader(self):
        """
        Create training dataloader.

        Returns:
            DataLoader: Training data loader with shuffling enabled

        Raises:
            RuntimeError: If setup() hasn't been called yet
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup() before requesting train_dataloader.")

        # Use num_workers=0 on Kaggle to avoid os.fork() deadlock with multi-GPU training
        import os
        num_workers = 0 if os.path.exists('/kaggle') else self.data_config.num_workers

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=False  # Disable persistent workers to avoid fork issues
        )

    def val_dataloader(self):
        """
        Create validation dataloader.

        Returns:
            DataLoader: Validation data loader without shuffling

        Raises:
            RuntimeError: If setup() hasn't been called yet
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup() before requesting val_dataloader.")

        # Use num_workers=0 on Kaggle to avoid os.fork() deadlock with multi-GPU training
        import os
        num_workers = 0 if os.path.exists('/kaggle') else self.data_config.num_workers

        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=False  # Disable persistent workers to avoid fork issues
        )
