"""
This file handles the end-to-end data pipeline for the fine-grained visual
categorization task on the dataset:
  1. Image indexing: Dynamically maps image filenames to filesystem paths regardless
     of directory layout.
  2. Dataset metadata reading: Parses train and test annotation files (filename and class label).
  3. Custom PyTorch Dataset: loads images, ensures RGB consistency, and applies transforms.
  4. Augmentation strategy: Implements state-of-the-art auto-augmentation (TrivialAugmentWide),
     random resized cropping, horizontal flips, and random erasing for training; deterministic
     resizing and center-cropping for evaluation.
  5. DataLoader creation: Performs stratified train-validation splitting (15% validation) to
     guarantee balanced class distributions, applies hardware optimizations (channels_last,
     pinned memory, persistent multi-worker prefetching), and returns ready-to-use PyTorch DataLoaders.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from src.config import CFG

# ImageNet standard channel mean and standard deviation
# All backbone models are pre-trained on ImageNet-1K
IMNET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMNET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Input spatial resolutions matching native pretraining of candidate architectures
# - ResNet-101: Native ImageNet resolution is 224x224
# - EfficientNet-B2: Native resolution scaled by compound scaling rule is 260x260
# - Swin-T: Standard shifted-window patch partition operates natively on 224x224
MODEL_RES: Dict[str, int] = {
    "resnet101": 224,
    "efficientnet_b2": 260,
    "swin_t": 224,
}

def build_image_index(root: Path) -> Dict[str, Path]:
    """
    Recursively scans the dataset directory and constructs a lookup table mapping
    image basenames to their absolute filesystem paths.

    """
    idx: Dict[str, Path] = {}
    # Recursively find all image files with supported extensions
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            # setdefault ensures the first discovered path is preserved if duplicates exist
            idx.setdefault(p.name, p)
    return idx

def read_list(txt_path: Path) -> List[Tuple[str, int]]:
    """
    Parses space-delimited annotation text files 
    Each valid line contains an image filename followed by an integer class label
    """
    items: List[Tuple[str, int]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Use rsplit(" ", 1) from the right to safely handle filenames containing spaces
                name, label = line.rsplit(" ", 1)
                items.append((name, int(label)))
    return items

class BirdDataset(Dataset):
    """
    PyTorch Dataset implementation for fine-grained bird classification

    Features:
      - Validates each annotation entry against the discovered filesystem image index
      - Skips missing files gracefully while reporting their count and examples
      - Performs on-demandloading from disk to minimize RAM footprint
      - Enforces 3-channel RGB representation
      - Applies data augmentations and normalization transforms on the fly
    """

    def __init__(self, items: List[Tuple[str, int]], img_index: Dict[str, Path], transform=None):
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.missing: List[str] = []

        # Resolve filesystem path for each listed sample
        for name, label in items:
            path = img_index.get(name)
            if path is None:
                # Record missing files
                self.missing.append(name)
            else:
                self.samples.append((path, label))

        if self.missing:
            print(f"[BirdDataset] Skipped {len(self.missing)} missing files, e.g. {self.missing[:3]}")

    def __len__(self) -> int:
        """Returns the total number of successfully loaded samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves the i-th sample from the dataset

        """
        path, label = self.samples[i]

        # Lazy load image and convert to 3-channel RGB to prevent channel mismatch
        img = Image.open(path).convert("RGB")

        # Apply data augmentation / normalization pipeline if provided
        if self.transform is not None:
            img = self.transform(img)

        return img, label


# Data Augmentation and Preprocessing Pipelines
def get_transforms(res: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs data augmentation pipelines for training and evaluation.

    Fine-Grained Classification Considerations:
      In fine-grained bird classification (CUB-200), subtle diagnostic marks (beak shape,
      wing bar patterns, eye rings, crown color) distinguish species. 
      - Scale range in RandomResizedCrop is restricted to (0.5, 1.0) 
      - RandomHorizontalFlip preserves biological validity (birds are horizontally symmetric)
      - TrivialAugmentWide applies auto-augmentation 
      - RandomErasing (p=0.2) randomly occludes small rectangular patches
    """
    # Training Pipeline: Heavy Regularization and Augmentation
    # Doesn't work well for Swin-T, underfitting
    train_tf = transforms.Compose([
        # Random scale and crop to provide scale invariance while preserving subject context
        transforms.RandomResizedCrop(res, scale=(0.5, 1.0)),
        # Random horizontal reflection (50% probability)
        transforms.RandomHorizontalFlip(p=0.5),
        # auto-augmentation for visual diversity
        transforms.TrivialAugmentWide(),
        # Convert PIL Image to PyTorch FloatTensor
        transforms.ToTensor(),
        # Normalize pixel values using ImageNet mean and std
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
        # Randomly occlude small patches to force global feature learning
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random"),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(res * 1.14)),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
    ])

    return train_tf, eval_tf

def make_loaders(model_name: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for train, validation, and test sets

    """
    # Determine input resolution based on experimental configuration
    res = 224 if CFG.UNIFORM_RES else MODEL_RES[model_name]
    train_tf, eval_tf = get_transforms(res)

    # Build image lookup table and parse train / test annotation files
    img_index = build_image_index(CFG.DATA_ROOT)
    train_items = read_list(CFG.TRAIN_TXT)
    test_items = read_list(CFG.TEST_TXT)

    # Instantiate full dataset objects (one with training transforms, one with eval transforms)
    full_train = BirdDataset(train_items, img_index, transform=train_tf)
    full_val = BirdDataset(train_items, img_index, transform=eval_tf)
    labels = [label for _, label in full_train.samples]

    # Perform stratified train/validation split (85% train, 15% validation)
    # Stratification ensures each of the 200 species has an equal proportion in train and val
    tr_idx, va_idx = train_test_split(
        np.arange(len(labels)),
        test_size=CFG.VAL_FRAC,
        stratify=labels,
        random_state=CFG.SEED,
    )

    # Create PyTorch Subsets for training and validation splits
    train_ds = Subset(full_train, tr_idx)
    val_ds = Subset(full_val, va_idx)
    test_ds = BirdDataset(test_items, img_index, transform=eval_tf)

    # Configure DataLoader hardware throughput settings
    nw = CFG.NUM_WORKERS
    loader_kwargs = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        # Prevent worker respawn overhead and pre-cache batches ahead of GPU execution
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    # Construct DataLoader instances
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print(f"[{model_name}] Configured DataLoaders: resolution={res}x{res} | "
          f"Train={len(train_ds)} samples | Val={len(val_ds)} samples | Test={len(test_ds)} samples")

    return train_loader, val_loader, test_loader
