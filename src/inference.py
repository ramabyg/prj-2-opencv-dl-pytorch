"""
Inference module for generating predictions on test data.
Creates submission.csv for Kaggle competition.
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .dataset import KenyanFood13Dataset
from .config import is_kaggle_environment, TrainingConfiguration, DataConfiguration, SystemConfiguration

# Register safe globals for PyTorch 2.6+ checkpoint loading
# This allows loading checkpoints that contain our custom configuration classes
torch.serialization.add_safe_globals([
    TrainingConfiguration,
    DataConfiguration,
    SystemConfiguration
])


def generate_predictions(
    model,
    test_csv_path,
    test_img_dir,
    checkpoint_path,
    output_csv_path="submission.csv",
    batch_size=32,
    num_workers=0,
    model_name="efficientnetv2",
    device="cuda"
):
    """
    Generate predictions for test data and create submission CSV.

    Args:
        model: PyTorch Lightning model instance
        test_csv_path: Path to test.csv file
        test_img_dir: Directory containing test images
        checkpoint_path: Path to trained model checkpoint (.ckpt)
        output_csv_path: Path to save submission.csv
        batch_size: Batch size for inference
        num_workers: Number of data loader workers
        model_name: Model name for preprocessing
        device: Device to run inference on ('cuda' or 'cpu')

    Returns:
        DataFrame: Submission dataframe with image_id and predicted labels
    """

    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model from checkpoint
    print(f"[INFO] Loading model from checkpoint: {checkpoint_path}")
    model = model.__class__.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.eval()
    model.to(device)
    print("[OK] Model loaded successfully")

    # Get model-specific preprocessing
    from .model import KenyanFood13Classifier
    _, input_size, mean, std = KenyanFood13Classifier.get_model_preprocessing(model_name)

    print(f"[INFO] Using preprocessing for {model_name}: size={input_size}, mean={mean}, std={std}")

    # Create test transforms (no augmentation, just normalization)
    test_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Create test dataset
    print(f"[INFO] Loading test dataset from {test_csv_path}")
    test_dataset = KenyanFood13Dataset(
        annotations_file=test_csv_path,
        img_dir=test_img_dir,
        train=False,
        test=True,  # Important: test mode
        transform=test_transforms,
        local_mode=not is_kaggle_environment()
    )

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for submission
        num_workers=num_workers,
        persistent_workers=False
    )

    print(f"[INFO] Test dataset size: {len(test_dataset)}")
    print(f"[INFO] Starting inference...")

    # Run inference
    all_image_ids = []
    all_predictions = []

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Generating predictions"):
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Get predicted class indices
            predictions = logits.argmax(dim=1).cpu().numpy()

            # Store results
            all_image_ids.extend(image_ids)
            all_predictions.extend(predictions.tolist())

    print(f"[OK] Generated {len(all_predictions)} predictions")

    # Get class index to name mapping (reverse of train dataset mapping)
    # We need to load train dataset briefly to get the class mapping
    print("[INFO] Loading class mapping from training data...")
    train_csv = test_csv_path.replace("test", "train")

    # Check if train CSV exists
    if not os.path.exists(train_csv):
        # Try alternative paths
        if is_kaggle_environment():
            train_csv = "/kaggle/input/opencv-pytorch-project-2-classification-round-3/train.csv"
        else:
            train_csv = "../data/kenyan-food-13/train.csv"

    if os.path.exists(train_csv):
        # Get class mapping from train dataset
        temp_dataset = KenyanFood13Dataset(
            annotations_file=train_csv,
            img_dir=test_img_dir,  # Directory doesn't matter, we just need mapping
            train=True,
            transform=None
        )
        idx_to_class = {v: k for k, v in temp_dataset.class_to_idx.items()}
        print(f"[OK] Loaded class mapping with {len(idx_to_class)} classes")

        # Convert predictions from indices to class names
        predicted_labels = [idx_to_class[idx] for idx in all_predictions]
    else:
        print("[WARN] Could not load class mapping, using numeric indices")
        predicted_labels = all_predictions

    # Create submission dataframe
    submission_df = pd.DataFrame({
        'image_id': all_image_ids,
        'label': predicted_labels
    })

    # Save to CSV
    submission_df.to_csv(output_csv_path, index=False)
    print(f"[OK] Submission saved to: {output_csv_path}")
    print(f"[INFO] Submission shape: {submission_df.shape}")
    print(f"[INFO] Sample predictions:")
    print(submission_df.head(10))

    # Show prediction distribution
    if isinstance(predicted_labels[0], str):
        print(f"\n[INFO] Prediction distribution:")
        print(submission_df['label'].value_counts().head(10))

    return submission_df


def create_submission(
    checkpoint_path,
    test_csv_path=None,
    test_img_dir=None,
    output_csv_path="submission.csv",
    model_config=None,
    batch_size=32
):
    """
    Convenience function to create submission from checkpoint.
    Automatically detects Kaggle environment and sets paths.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_csv_path: Path to test.csv (auto-detected if None)
        test_img_dir: Path to test images (auto-detected if None)
        output_csv_path: Output path for submission.csv
        model_config: Training configuration (to get model_name)
        batch_size: Batch size for inference

    Returns:
        DataFrame: Submission dataframe
    """

    # Auto-detect paths for Kaggle
    if test_csv_path is None:
        if is_kaggle_environment():
            test_csv_path = "/kaggle/input/opencv-pytorch-project-2-classification-round-3/test.csv"
        else:
            test_csv_path = "../data/kenyan-food-13/test.csv"

    if test_img_dir is None:
        if is_kaggle_environment():
            test_img_dir = "/kaggle/input/opencv-pytorch-project-2-classification-round-3/images/images"
        else:
            test_img_dir = "../data/kenyan-food-13/images/images"

    # Get model configuration
    if model_config is None:
        from .config import TrainingConfiguration
        model_config = TrainingConfiguration()

    # Import model class
    from .model import KenyanFood13Classifier

    # Create a dummy model instance (will be replaced by checkpoint)
    model = KenyanFood13Classifier(model_config, num_classes=13)

    # Generate predictions
    submission_df = generate_predictions(
        model=model,
        test_csv_path=test_csv_path,
        test_img_dir=test_img_dir,
        checkpoint_path=checkpoint_path,
        output_csv_path=output_csv_path,
        batch_size=batch_size,
        num_workers=0,  # Avoid multiprocessing issues
        model_name=model_config.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return submission_df
