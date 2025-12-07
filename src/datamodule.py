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
        mean (list, optional): RGB channel means for normalization.
            If None, uses ImageNet defaults [0.485, 0.456, 0.406]
        std (list, optional): RGB channel standard deviations for normalization.
            If None, uses ImageNet defaults [0.229, 0.224, 0.225]

    Attributes:
        num_classes (int): Number of classes (available after calling setup())
        train_dataset (Dataset): Training dataset (available after setup())
        val_dataset (Dataset): Validation dataset (available after setup())

    Example:
        >>> from src.config import DataConfiguration
        >>> data_config = DataConfiguration()
        >>> data_module = KenyanFood13DataModule(data_config)
        >>> data_module.setup()
        >>> train_loader = data_module.train_dataloader()
    """

    def __init__(self, data_config, mean=None, std=None):
        super().__init__()
        self.data_config = data_config

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
        aug_transforms = transforms.Compose([
            transforms.RandomResizedCrop((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
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

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers
        )

    def val_dataloader(self):
        """
        Create validation dataloader.

        Returns:
            DataLoader: Validation data loader (no shuffling)

        Raises:
            RuntimeError: If setup() hasn't been called yet
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup() before requesting val_dataloader.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers
        )
