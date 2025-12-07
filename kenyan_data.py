import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import lightning as L


class KenyanFood13Dataset(Dataset):
    """Top-level dataset module for pickling across DataLoader workers.

    Notes:
    - Define this class in a standalone module (not in a notebook cell) so
      multiprocessing (spawn) on Windows can import it in worker processes.
    - The CSV is expected to contain at least two columns: an image id/filename
      and a class label. By default this implementation uses the first two
      columns, but it's safer to pass explicit column names (see args).
    """

    def __init__(self,
                 annotations_file: str,
                 img_dir: str,
                 train: bool = True,
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[callable] = None,
                 id_col: Optional[str] = None,
                 class_col: Optional[str] = None):

        annotations_file = os.path.abspath(annotations_file)
        img_dir = os.path.abspath(img_dir)

        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.df = pd.read_csv(annotations_file)
        if self.df.shape[1] < 2:
            raise ValueError("Annotations CSV must contain at least two columns: id and class")

        # determine columns
        if id_col is None:
            id_col = self.df.columns[0]
        if class_col is None:
            class_col = self.df.columns[1]

        self.id_col = id_col
        self.class_col = class_col

        # ensure strings for ids and classes
        self.df[self.id_col] = self.df[self.id_col].astype(str)
        self.df[self.class_col] = self.df[self.class_col].astype(str)

        # build label mapping using the full CSV so mapping is consistent
        unique_labels = self.df[self.class_col].unique()
        self.label2idx = {label: int(idx) for idx, label in enumerate(sorted(unique_labels))}
        self.idx2label = {int(idx): label for label, idx in self.label2idx.items()}

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        # split dataset (80:20)
        total = len(self.df)
        split_index = int(0.8 * total)
        if self.train:
            self.df = self.df.iloc[:split_index].reset_index(drop=True)
        else:
            self.df = self.df.iloc[split_index:].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        img_id = str(row[self.id_col])
        img_path = os.path.join(self.img_dir, img_id if img_id.lower().endswith(('.jpg', '.jpeg', '.png')) else img_id + '.jpg')

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to open image {img_path}: {e}")

        label_str = str(row[self.class_col])
        label = self.label2idx.get(label_str)
        if label is None:
            raise KeyError(f"Label '{label_str}' not found in label2idx mapping")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, int(label)


class KenyanFood13DataModule(L.LightningDataModule):
    """Lightning DataModule for Kenyan Food 13 Dataset.

    This properly inherits from LightningDataModule so it works with Lightning Trainer.
    """

    def __init__(self, data_config, mean=None, std=None):
        super().__init__()
        self.data_config = data_config
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # no-op: data is expected to be available locally; override if downloading is needed
        return None

    def setup(self, stage: Optional[str] = None):
        # Lightning calls setup(stage=...) so accept the optional stage argument
        if isinstance(self.data_config.input_size, int):
            h = w = self.data_config.input_size
        else:
            h, w = self.data_config.input_size

        common_transforms = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        aug_transforms = transforms.Compose([
            transforms.RandomResizedCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.train_dataset = KenyanFood13Dataset(
            annotations_file=self.data_config.annotations_file,
            img_dir=self.data_config.img_dir,
            train=True,
            transform=aug_transforms
        )

        self.val_dataset = KenyanFood13Dataset(
            annotations_file=self.data_config.annotations_file,
            img_dir=self.data_config.img_dir,
            train=False,
            transform=common_transforms
        )

        self.num_classes = len(self.train_dataset.label2idx)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError('Call setup() before train_dataloader()')
        return DataLoader(self.train_dataset,
                  batch_size=self.data_config.batch_size,
                  shuffle=True,
                  num_workers=self.data_config.num_workers,
                  pin_memory=True if torch.cuda.is_available() else False,
                  persistent_workers=True if self.data_config.num_workers > 0 else False)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError('Call setup() before val_dataloader()')
        return DataLoader(self.val_dataset,
                  batch_size=self.data_config.batch_size,
                  shuffle=False,
                  num_workers=self.data_config.num_workers,
                  pin_memory=True if torch.cuda.is_available() else False,
                  persistent_workers=True if self.data_config.num_workers > 0 else False)
