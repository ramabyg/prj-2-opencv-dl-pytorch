"""
PyTorch Lightning model for Kenyan Food 13 Classification.
Uses transfer learning with GoogleNet as base model.
"""

import torch
import torch.nn as nn
import torchvision
import lightning as L
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall
)

from .config import TrainingConfiguration


class KenyanFood13Classifier(L.LightningModule):
    """
    Lightning module for food classification using transfer learning.

    Uses a pre-trained GoogleNet model with a modified final layer for
    the specific number of food classes. Includes comprehensive metric
    tracking for both training and validation.

    Args:
        training_config (TrainingConfiguration): Training configuration object
        num_classes (int): Number of food classes to classify

    Attributes:
        model: The underlying neural network (GoogleNet)
        criterion: Loss function (CrossEntropyLoss)
        Various metrics for train/validation tracking

    Example:
        >>> from src.config import TrainingConfiguration
        >>> config = TrainingConfiguration()
        >>> model = KenyanFood13Classifier(config, num_classes=13)
        >>> predictions = model(images)
    """

    def __init__(self, training_config: TrainingConfiguration, num_classes: int):
        super(KenyanFood13Classifier, self).__init__()
        self.save_hyperparameters()

        # Store training configuration
        self.training_config = training_config
        self.num_classes = num_classes

        # Load base model
        if training_config.model_name == "googlenet":
            self.model = torchvision.models.googlenet(pretrained=training_config.pretrained)
            # Replace the final layer for our number of classes
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Model {training_config.model_name} not supported.")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training metrics
        self.train_mean_loss = MeanMetric()
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.train_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.train_recall = MulticlassRecall(num_classes=num_classes, average='macro')

        # Validation metrics
        self.val_mean_loss = MeanMetric()
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=num_classes, average='macro')

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch

        Returns:
            torch.Tensor: Loss value for this batch
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        self.train_mean_loss.update(loss)
        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_f1.update(preds, labels)

        # Log step-level metrics
        self.log('train/loss', self.train_mean_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log('train/precision', self.train_precision.compute(), on_epoch=True, prog_bar=True)
        self.log('train/recall', self.train_recall.compute(), on_epoch=True, prog_bar=True)
        self.log('train/f1', self.train_f1.compute(), on_epoch=True, prog_bar=True)
        self.log('step', self.current_epoch, on_epoch=True, prog_bar=True)

        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        self.val_mean_loss.update(loss)
        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        # Log validation metrics
        self.log('valid/loss', self.val_mean_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid/recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self):
        """Log epoch-level validation metrics."""
        self.log('valid/precision', self.val_precision.compute(), on_epoch=True, prog_bar=True)
        self.log('valid/recall', self.val_recall.compute(), on_epoch=True, prog_bar=True)
        self.log('valid/f1', self.val_f1.compute(), on_epoch=True, prog_bar=True)
        self.log('step', self.current_epoch, on_epoch=True, prog_bar=True)

        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Optimizer or dict with optimizer and scheduler configuration
        """
        # Create optimizer based on configuration
        if self.training_config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.training_config.learning_rate,
                momentum=self.training_config.momentum,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.training_config.optimizer} not supported.")

        # Configure learning rate scheduler if enabled
        if self.training_config.use_scheduler:
            if self.training_config.scheduler.lower() == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.training_config.lr_step_size,
                    gamma=self.training_config.lr_gamma
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1
                    }
                }
            elif self.training_config.scheduler.lower() == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.training_config.num_epochs
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1
                    }
                }
            elif self.training_config.scheduler.lower() == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=self.training_config.lr_gamma,
                    patience=3
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "valid/acc",
                        "interval": "epoch",
                        "frequency": 1
                    }
                }
            else:
                raise ValueError(f"Scheduler {self.training_config.scheduler} not supported.")

        # Return optimizer only if no scheduler is configured
        return optimizer
