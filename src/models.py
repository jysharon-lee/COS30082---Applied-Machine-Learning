import torch.nn as nn
from torchvision import models

from src.config import CFG, device

HEAD_KEY = {"resnet101": "fc", "efficientnet_b2": "classifier", "swin_t": "head"}


def build_model(name: str, num_classes: int = None, pretrained: bool = True):
    """ImageNet-pretrained backbone with a fresh classification head."""
    num_classes = num_classes or CFG.NUM_CLASSES

    if name == "resnet101":
        w = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet101(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif name == "efficientnet_b2":
        w = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b2(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head = m.classifier[1]
    elif name == "swin_t":
        w = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.swin_t(weights=w)
        m.head = nn.Linear(m.head.in_features, num_classes)
        head = m.head
    else:
        raise ValueError(f"Unknown model: {name}")

    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    return m.to(device).to(memory_format=__import__("torch").channels_last)


def freeze_backbone(model, name: str):
    """Train the head only (linear probe)."""
    key = HEAD_KEY[name]
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(key)


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot / 1e6, trn / 1e6