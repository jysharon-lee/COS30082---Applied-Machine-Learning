from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from src.config import CFG

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MODEL_RES = {"resnet101": 224, "efficientnet_b2": 260, "swin_t": 224}


def build_image_index(root: Path):
    """basename -> full path, so images can live anywhere under DATA_ROOT."""
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            idx.setdefault(p.name, p)
    return idx


def read_list(txt_path: Path):
    items = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                name, label = line.rsplit(" ", 1)
                items.append((name, int(label)))
    return items


class BirdDataset(Dataset):
    def __init__(self, items, img_index, transform=None):
        self.transform = transform
        self.samples, self.missing = [], []
        for name, label in items:
            path = img_index.get(name)
            if path is None:  # e.g. train.txt entries ending in "xxx.jpg"
                self.missing.append(name)
            else:
                self.samples.append((path, label))
        if self.missing:
            print(f"[BirdDataset] skipped {len(self.missing)} missing files, e.g. {self.missing[:3]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(res: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(res * 1.14)),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    return train_tf, eval_tf


def make_loaders(model_name: str):
    res = 224 if CFG.UNIFORM_RES else MODEL_RES[model_name]
    train_tf, eval_tf = get_transforms(res)

    img_index = build_image_index(CFG.DATA_ROOT)
    train_items = read_list(CFG.TRAIN_TXT)
    test_items = read_list(CFG.TEST_TXT)

    full_train = BirdDataset(train_items, img_index, train_tf)
    full_val = BirdDataset(train_items, img_index, eval_tf)
    labels = [l for _, l in full_train.samples]

    tr_idx, va_idx = train_test_split(
        np.arange(len(labels)), test_size=CFG.VAL_FRAC,
        stratify=labels, random_state=CFG.SEED)

    train_ds = Subset(full_train, tr_idx)
    val_ds = Subset(full_val, va_idx)
    test_ds = BirdDataset(test_items, img_index, eval_tf)

    nw = CFG.NUM_WORKERS
    kw = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(train_ds, CFG.BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, CFG.BATCH_SIZE, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, CFG.BATCH_SIZE, shuffle=False, **kw)

    print(f"{model_name}: res={res} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_loader, val_loader, test_loader