"""
Training orchestration for Kenyan Food 13 Classification.
Sets up PyTorch Lightning Trainer with callbacks and logging.
"""

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .config import TrainingConfiguration, DataConfiguration, SystemConfiguration


def train_model(
    training_config: TrainingConfiguration,
    data_config: DataConfiguration,
    system_config: SystemConfiguration,
    model: L.LightningModule,
    data_module: L.LightningDataModule
):
    """
    Train and validate a PyTorch Lightning model.

    Sets up trainer with callbacks (checkpointing, early stopping),
    TensorBoard logging, and appropriate precision/device settings.

    Args:
        training_config (TrainingConfiguration): Training parameters
        data_config (DataConfiguration): Data loading parameters
        system_config (SystemConfiguration): System/hardware configuration
        model (LightningModule): The model to train
        data_module (LightningDataModule): Data module with train/val loaders

    Returns:
        tuple: (trained_model, data_module, checkpoint_callback)

    Raises:
        ValueError: If model or data_module is None

    Example:
        >>> from src.config import get_config
        >>> from src.model import KenyanFood13Classifier
        >>> from src.datamodule import KenyanFood13DataModule
        >>>
        >>> train_cfg, data_cfg, sys_cfg = get_config()
        >>> data_module = KenyanFood13DataModule(data_cfg)
        >>> data_module.setup()
        >>> model = KenyanFood13Classifier(train_cfg, data_module.num_classes)
        >>>
        >>> trained_model, _, ckpt = train_model(
        ...     train_cfg, data_cfg, sys_cfg, model, data_module
        ... )
    """
    # Set random seed for reproducibility
    L.seed_everything(training_config.random_seed)

    # Validate inputs
    if not model:
        raise ValueError(
            "Model must be provided for training. "
            "Please initialize the model before calling this function."
        )
    if not data_module:
        raise ValueError("Data module is required to run the model")

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=system_config.output_dir,
        filename="{epoch}-{valid/acc:.4f}",
        save_top_k=3,
        monitor="valid/acc",
        mode="max",
        auto_insert_metric_name=False,
        save_weights_only=True,
        verbose=True
    )

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="valid/acc",
        patience=3,
        mode="max",
        verbose=True
    )

    # TensorBoard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=system_config.output_dir,
        name="kenyan_food_logs",
        version=None,  # Auto-incrementing version
        default_hp_metric=False
    )

    # Map precision string to PyTorch Lightning expected value
    precision_map = {
        "float32": 32,
        "float16": 16,
        "bfloat16": "bf16"
    }
    trainer_precision = precision_map.get(training_config.precision, 32)

    # Map device to accelerator type
    accelerator = "gpu" if system_config.device == "cuda" else "cpu"

    # Create trainer
    trainer = L.Trainer(
        max_epochs=training_config.num_epochs,
        accelerator=accelerator,
        devices="auto",  # Use all available devices
        precision=trainer_precision,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_logger,
        default_root_dir=system_config.output_dir,
        log_every_n_steps=training_config.log_interval,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train and validate
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    trainer.fit(model, datamodule=data_module)

    print("\n" + "="*60)
    print("Running Final Validation")
    print("="*60)
    trainer.validate(model, datamodule=data_module)

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print("="*60 + "\n")

    return model, data_module, checkpoint_callback


def training_validation(
    training_config: TrainingConfiguration,
    data_config: DataConfiguration,
    system_config: SystemConfiguration,
    model: L.LightningModule,
    data_module: L.LightningDataModule
):
    """
    Legacy wrapper for train_model() for backward compatibility.

    This function maintains the original naming convention used in the notebook.
    New code should use train_model() directly.

    Args:
        training_config: Training configuration
        data_config: Data configuration
        system_config: System configuration
        model: Lightning module to train
        data_module: Lightning data module

    Returns:
        tuple: Same as train_model()
    """
    return train_model(training_config, data_config, system_config, model, data_module)
