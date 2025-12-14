"""
Custom PyTorch Dataset for Kenyan Food 13 Classification.
Handles train/validation splitting and image loading.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class KenyanFood13Dataset(Dataset):
    """
    Custom Dataset for Kenyan Food 13 Classification Task.

    Automatically splits the dataset into train and validation sets with 80:20 ratio.
    The first 80% of images are used for training and the remaining 20% for validation.

    Args:
        annotations_file (str): Path to the CSV file with annotations (image_id, label columns)
        img_dir (str): Directory containing all the images
        train (bool, optional): If True, loads training set; if False, loads validation set.
            Default is True.
        transform (callable, optional): Optional transform to be applied to images
        target_transform (callable, optional): Optional transform to be applied to labels
        local_mode (bool, optional): If True, appends '_local' to annotations filename.
            Default is False.

    Attributes:
        num_classes (int): Number of unique classes in the dataset
        img_labels (DataFrame): Pandas DataFrame containing image filenames and labels
        img_dir (str): Path to image directory

    Example:
        >>> train_dataset = KenyanFood13Dataset(
        ...     annotations_file='train.csv',
        ...     img_dir='./images',
        ...     train=True,
        ...     transform=transforms.ToTensor()
        ... )
        >>> print(f"Training samples: {len(train_dataset)}")
        >>> image, label = train_dataset[0]

    Raises:
        FileNotFoundError: If annotations file or image directory doesn't exist
    """

    def __init__(self, annotations_file, img_dir, train=True, test=False, transform=None,
                 target_transform=None, local_mode=False):

        # Handle local mode file naming
        # if local_mode:
        #     base, ext = os.path.splitext(annotations_file)
        #     annotations_file = f"{base}_local{ext}"
        #     print("Running in local mode - loading data from local paths")

        # Validate paths
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Load annotations
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.local_mode = local_mode
        self.test_mode = test  # Flag for test/inference mode

        # Test mode: no labels, just image IDs
        if test:
            print(f"[INFO] Test mode: Loading {len(self.img_labels)} test samples (no labels)")
            print(f"[INFO] CSV columns: {list(self.img_labels.columns)}")
            self.num_classes = None  # Unknown for test set
            self.class_to_idx = None
            self.label_col = None
            return  # Skip label processing for test mode

        # Training/Validation mode: has labels

        # Detect label column name (try 'class', 'label', 'food_id', 'food_label')
        label_col = None
        for col_name in ['class', 'label', 'food_id', 'food_label', 'food']:
            if col_name in self.img_labels.columns:
                label_col = col_name
                break

        if label_col is None:
            # If no standard column found, show available columns and raise error
            raise ValueError(
                f"Could not find label column. Available columns: {list(self.img_labels.columns)}. "
                f"Expected one of: 'class', 'label', 'food_id', 'food_label', 'food'"
            )

        self.label_col = label_col
        print(f"[INFO] Using column '{label_col}' as label column")
        print(f"[INFO] CSV columns: {list(self.img_labels.columns)}")
        # Calculate number of classes
        num_classes = len(self.img_labels[self.label_col].unique())
        self.num_classes = num_classes


        # create a mapping from class names to integer labels
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(self.img_labels[self.label_col].unique()))}


        # Split into train and validation sets (80:20 ratio)
        split_index = int(0.8 * len(self.img_labels))
        if train:
            self.img_labels = self.img_labels.iloc[:split_index].reset_index(drop=True)
            print(f"Using {len(self.img_labels)} samples for training.")
            #print class to idx mapping for reference
            print(f"Class to index mapping: {self.class_to_idx}")
        else:
            self.img_labels = self.img_labels.iloc[split_index:].reset_index(drop=True)
            print(f"Using {len(self.img_labels)} samples for validation.")
            #print class to idx mapping for reference
            print(f"Class to index mapping: {self.class_to_idx}")

        print(f"Dataset initialized with {len(self.img_labels)} samples belonging to {num_classes} classes.")


    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int or Tensor): Index of the sample to retrieve

        Returns:
            tuple: (image, label) for train/val, (image, image_id) for test
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_filename = str(self.img_labels.iloc[idx, 0])
        # Add .jpg extension if not already present
        if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_filename += '.jpg'
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Test mode: return image_id instead of label
        if self.test_mode:
            image_id = str(self.img_labels.iloc[idx, 0])  # Return original image_id
            return image, image_id

        # Training/Validation mode: return label
        label_value = self.img_labels.iloc[idx][self.label_col]
        label = int(self.class_to_idx[label_value])
        
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.img_labels)
