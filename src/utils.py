"""
Utility functions for data preprocessing and analysis.
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image


def calculate_dataset_mean_std(annotations_file,
                               img_dir,
                               img_size=(224, 224),
                               sample_size=None,
                               local_mode=False):
    """
    Calculate mean and std for the dataset.

    This function computes per-channel (RGB) mean and standard deviation
    for normalization. It can process all images or a random sample for faster
    computation.

    Args:
        annotations_file: Path to CSV with image filenames and labels
        img_dir: Directory containing images
        img_size: Tuple of (height, width) to resize images to. Default: (224, 224)
        sample_size: If provided, only use this many images for calculation.
                    None means use all images. Recommended: 1000-5000 for speed.
        local_mode: If True, appends '_local' to annotations file name.
                   This is for backward compatibility with existing code.

    Returns:
        tuple: (mean, std) as lists of 3 float values each for RGB channels

    Example:
        >>> mean, std = calculate_dataset_mean_std(
        ...     annotations_file='train.csv',
        ...     img_dir='./images',
        ...     sample_size=1000
        ... )
        >>> print(f"Mean: {mean}, Std: {std}")
        Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]

    Note:
        - Images are normalized to [0, 1] range before computing statistics
        - Uses PyTorch tensors for efficient computation
        - Invalid/corrupted images are skipped with error message
    """
    # Handle local mode file naming
    if local_mode:
        base, ext = os.path.splitext(annotations_file)
        annotations_file = f"{base}_local{ext}"
        print(f"Running in local mode - using {annotations_file}")

    # Validate paths
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # Load annotations
    img_labels = pd.read_csv(annotations_file)
    total_images = len(img_labels)

    # Use subset for faster computation if specified
    if sample_size and sample_size < total_images:
        img_labels = img_labels.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Calculating mean and std from {sample_size} sampled images (out of {total_images})...")
    else:
        print(f"Calculating mean and std from all {total_images} images...")

    means = []
    stds = []
    processed_count = 0
    error_count = 0


    for idx, row in img_labels.iterrows():
        img_filename = str(row.iloc[0])
        # Add .jpg extension if not already present
        if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_filename += '.jpg'
        img_path = os.path.join(img_dir, img_filename)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]

            # Calculate mean and std per channel
            img_tensor = torch.tensor(img_array).permute(2, 0, 1)  # C, H, W
            means.append(img_tensor.mean(dim=(1, 2)))
            stds.append(img_tensor.std(dim=(1, 2)))
            processed_count += 1

            # Progress indicator
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count}/{len(img_labels)} images...", end='\r')

        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Only print first 5 errors to avoid spam
                print(f"\nWarning: Error processing {img_path}: {e}")
            continue

    if error_count > 5:
        print(f"\n  (Suppressed {error_count - 5} additional errors)")

    # Calculate overall mean and std
    if not means:
        raise RuntimeError("No images were successfully processed. Check your data paths and image files.")

    mean = torch.stack(means).mean(dim=0).tolist()
    std = torch.stack(stds).mean(dim=0).tolist()

    print(f"\n✓ Successfully processed {processed_count} images ({error_count} errors)")
    print(f"  Calculated mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  Calculated std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    return mean, std


def get_imagenet_stats():
    """
    Get ImageNet normalization statistics (commonly used default).

    Returns:
        tuple: (mean, std) as lists for RGB channels
    """
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
