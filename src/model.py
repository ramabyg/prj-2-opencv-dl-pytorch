"""
PyTorch Lightning model for Kenyan Food 13 Classification.
Uses transfer learning with GoogleNet as base model.
"""

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as L
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall
)

# Fix SSL certificate verification issues when downloading pre-trained weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from .config import TrainingConfiguration


from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

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

    @staticmethod
    def get_model_preprocessing(model_name: str):
        """
        Get the preprocessing transforms used during pre-training for a specific model.

        Args:
            model_name: Name of the model ('resnet50', 'efficientnetv2', 'googlenet')

        Returns:
            tuple: (preprocessing_transforms, input_size, mean, std)
        """
        model_name = model_name.lower()
        if model_name == "resnet50":
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2
            return weights.transforms(), 224, weights.transforms().mean, weights.transforms().std
        elif model_name == "efficientnetv2":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            return weights.transforms(), 384, weights.transforms().mean, weights.transforms().std
        elif model_name == "googlenet":
            from torchvision.models import GoogLeNet_Weights
            weights = GoogLeNet_Weights.IMAGENET1K_V1
            return weights.transforms(), 224, weights.transforms().mean, weights.transforms().std
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def __init__(self, training_config: TrainingConfiguration, num_classes: int):
        super(KenyanFood13Classifier, self).__init__()
        self.save_hyperparameters()

        # Store training configuration
        self.training_config = training_config
        self.num_classes = num_classes

        # Load base model

        model_name = training_config.model_name.lower()
        freeze_pct = getattr(training_config, 'freeze_pct', 0.6)
        if model_name == "resnet50":
            if training_config.pretrained:
                print("Loading pre-trained ResNet50 weights...")
                from torchvision.models import ResNet50_Weights
                self.model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                print("[OK] Pre-trained ResNet50 weights loaded successfully")
            else:
                print("Initializing ResNet50 from scratch (no pre-trained weights)")
                self.model = torchvision.models.resnet50(weights=None)

            print("Freezing early layers, unfreezing later layers for fine-tuning...")
            all_params = list(self.model.parameters())
            freeze_until = int(len(all_params) * freeze_pct)
            for i, param in enumerate(all_params):
                param.requires_grad = i >= freeze_until

            # Count layers (modules) frozen/unfrozen
            all_layers = list(self.model.children())
            total_layers = len(all_layers)
            freeze_layer_until = int(total_layers * freeze_pct)
            print(f"  Layers frozen: {freeze_layer_until} / {total_layers}")

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  Trainable: {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)")

            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        elif model_name == "efficientnetv2":
            if training_config.pretrained:
                print("Loading pre-trained EfficientNetV2-S weights...")
                self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
                print("[OK] Pre-trained EfficientNetV2-S weights loaded successfully")
            else:
                print("Initializing EfficientNetV2-S from scratch (no pre-trained weights)")
                self.model = efficientnet_v2_s(weights=None)

            print("Freezing early layers, unfreezing later layers for fine-tuning...")
            all_params = list(self.model.parameters())
            freeze_until = int(len(all_params) * freeze_pct)
            for i, param in enumerate(all_params):
                param.requires_grad = i >= freeze_until

            all_layers = list(self.model.children())
            total_layers = len(all_layers)
            freeze_layer_until = int(total_layers * freeze_pct)
            print(f"  Layers frozen: {freeze_layer_until} / {total_layers}")

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  Trainable: {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)")

            # EfficientNetV2-S classifier is a Sequential, last layer is Linear
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

        elif model_name == "googlenet":
            # Use new torchvision API (weights parameter instead of pretrained)
            if training_config.pretrained:
                print("Loading pre-trained GoogleNet weights...")
                from torchvision.models import GoogLeNet_Weights
                self.model = torchvision.models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
                print("[OK] Pre-trained weights loaded successfully")
            else:
                print("Initializing GoogleNet from scratch (no pre-trained weights)")
                self.model = torchvision.models.googlenet(weights=None)

            # Replace the final layer for our number of classes
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Model {training_config.model_name} not supported. Use 'resnet50' or 'googlenet'.")

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

        # Log validation metrics (sync_dist=True for multi-GPU)
        self.log('valid/loss', self.val_mean_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid/acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid/precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('valid/recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('valid/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        """Log epoch-level validation metrics and model parameters."""
        self.log('valid/precision', self.val_precision.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid/recall', self.val_recall.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid/f1', self.val_f1.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('step', self.current_epoch, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log model parameter histograms to TensorBoard (shows weight distributions)
        if self.logger:
            for name, params in self.named_parameters():
                if params.requires_grad:
                    # Log weight distributions
                    self.logger.experiment.add_histogram(f'weights/{name}', params, self.current_epoch)
                    # Log gradient distributions (if gradients exist)
                    if params.grad is not None:
                        self.logger.experiment.add_histogram(f'gradients/{name}', params.grad, self.current_epoch)

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
