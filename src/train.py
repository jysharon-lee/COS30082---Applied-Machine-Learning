"""
This file provides the core training, validation, and evaluation pipeline for
comparing deep learning architectures (ResNet-101, EfficientNet-B2, Swin-T) on
the CUB-200-2011 dataset
"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from src.config import CFG, device, seed_everything
from src.data import make_loaders
from src.models import build_model, count_params, HEAD_KEY


DEFAULTS: Dict[str, Any] = dict(
    epochs=30,            # Maximum training epochs; sufficient for fine-tuning pretrained models
    lr_backbone=1e-4,     # Conservative learning rate for pretrained backbone weights
    lr_head=1e-3,         # 10x higher learning rate for the randomly initialized linear classifier
    weight_decay=0.05,    # L2 weight decay regularizer applied via AdamW
    warmup_epochs=2,      # Initial linear warmup epochs to stabilize early gradient updates
    label_smoothing=0.1,  # Label smoothing factor to prevent overconfident predictions
    mixup_prob=0.0,       # Probability of applying Mixup (default 0.0; 0.2-0.3 for gentle Mixup)
    layer_decay=1.0,      # Multiplier for Layer-Wise Learning Rate Decay (< 1.0 enables LLRD for Swin-T)
    patience=8,           # Early stopping patience based on validation Top-1 accuracy
    grad_clip=1.0,        # Maximum gradient norm threshold to prevent gradient explosion
)

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 200,
    mixup_alpha: float = 0.2,
    prob: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # If Mixup is disabled or the random roll exceeds probability threshold, return one-hot targets
    if prob <= 0 or np.random.rand() > prob:
        y_onehot = torch.zeros(y.size(0), num_classes, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
        return x, y_onehot

    # Convert integer targets to one-hot encoded label vectors
    y_onehot = torch.zeros(y.size(0), num_classes, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)

    # Sample convex mixing coefficient from Beta distribution
    lam = float(np.random.beta(mixup_alpha, mixup_alpha))

    # Random permutation of batch indices to pair samples
    perm = torch.randperm(x.size(0), device=x.device)

    # Compute linear convex combinations of images and target vectors
    y_mixed = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
    x_mixed = lam * x + (1.0 - lam) * x[perm]

    return x_mixed, y_mixed

def compute_avg_per_class_acc(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 200,
) -> float:
    """
    Computes the Average Accuracy Per Class as explicitly defined in Section 4
    of the assignment specification
    """
    class_accuracies: List[float] = []
    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            # Accuracy on class c = fraction of ground truth samples of class c correctly classified
            acc_c = (preds[mask] == targets[mask]).mean()
            class_accuracies.append(float(acc_c))

    # Return macro average across all evaluated classes
    return float(np.mean(class_accuracies)) if class_accuracies else 0.0


def get_swin_layer_id(param_name: str) -> int:
    """
    LLRD
    Maps parameter names of a Swin Transformer (Swin-T) to hierarchical stage IDs (0 to 4):
      - Stage 0: Patch Embedding & Stage 1 Transformer blocks (edges, colors, low-level textures)
      - Stage 1: Stage 2 Transformer blocks (local structural patterns)
      - Stage 2: Stage 3 Transformer blocks (intermediate semantic parts)
      - Stage 3: Stage 4 Transformer blocks & Final LayerNorm (high-level bird semantics)
      - Stage 4: Linear classification head
    """
    if param_name.startswith("head"):
        return 4
    if param_name.startswith("norm") or param_name.startswith("features.7"):
        return 3
    if param_name.startswith("features.5") or param_name.startswith("features.6"):
        return 2
    if param_name.startswith("features.3") or param_name.startswith("features.4"):
        return 1
    return 0  # features.0 (patch embed), features.1, features.2 (stage 1 blocks)


def make_optimizer(
    model: nn.Module,
    name: str,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    layer_decay: float = 1.0,
) -> torch.optim.AdamW:
    """
    Constructs an AdamW optimizer with decoupled parameter groups.
    """
    key = HEAD_KEY[name]

    # Swin-T with Layer-Wise Learning Rate Decay enabled
    if name == "swin_t" and layer_decay < 1.0:
        groups: Dict[int, List[nn.Parameter]] = {i: [] for i in range(5)}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lid = get_swin_layer_id(n)
            groups[lid].append(p)

        # Head receives lr_head
        param_groups = [
            {"params": groups[4], "lr": lr_head, "name": "head"},
        ]
        # Deeper stages down to shallow stages receive decayed backbone learning rates
        for i in range(3, -1, -1):
            decay_factor = layer_decay ** (3 - i)
            param_groups.append({
                "params": groups[i],
                "lr": lr_backbone * decay_factor,
                "name": f"stage_{i+1}",
            })

        print(f"[LLRD] Swin-T layer decay active (decay={layer_decay:.2f}): "
              f"Head={lr_head:.1e} | Stage4={lr_backbone:.1e} | "
              f"Stage3={lr_backbone*layer_decay:.1e} | "
              f"Stage2={lr_backbone*(layer_decay**2):.1e} | "
              f"Stage1={lr_backbone*(layer_decay**3):.1e}")

        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Standard two-group differential learning rates (Backbone vs Head)
    else:
        head_params, backbone_params = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                (head_params if n.startswith(key) else backbone_params).append(p)

        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
                {"params": head_params, "lr": lr_head, "name": "head"},
            ],
            weight_decay=weight_decay,
        )


# Learning Rate Scheduler (Linear Warmup + Cosine Annealing)
def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Creates a learning rate schedule combining:
      1. Linear Warmup: Increases learning rate linearly from 0 to 1.0 over `warmup_steps`
         to prevent erratic gradient updates from the fresh linear head early on
      2. Cosine Decay: Anneals learning rate from 1.0 down to a minimum residual of 0.01 (1%)
         following a cosine curve, facilitating convergence into a broad, stable minimum
    """
    def lr_lambda(step: int) -> float:
        # Phase 1: Linear warmup
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # Phase 2: Cosine annealing to minimum 0.01 multiplier
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _to_device(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transfers input images and labels to the active compute device (GPU) with optimizations:
      - non_blocking=True: Overlaps host-to-device memory copy with GPU compute.
      - memory_format=torch.channels_last (NHWC): Optimizes memory access patterns
        for NVIDIA Tensor Core acceleration.

    """
    x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    return x, y.to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
    label_smoothing: float = 0.1,
    mixup_prob: float = 0.0,
    limit_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Runs one complete training epoch over the training DataLoader.

    """
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for i, (x, y) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break

        # Move tensors to GPU with channels_last layout
        x, y = _to_device(x, y)

        # Apply Mixup if activated
        if mixup_prob > 0.0:
            x, y_soft = mixup_batch(x, y, num_classes=CFG.NUM_CLASSES, prob=mixup_prob)
        else:
            y_soft = y

        # Zero gradients with set_to_none=True 
        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision (AMP) forward pass
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = nn.functional.cross_entropy(out, y_soft, label_smoothing=label_smoothing)

        # Scaled backpropagation
        scaler.scale(loss).backward()

        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer update and dynamic scale factor adjustment
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler per batch for smooth cosine annealing
        scheduler.step()

        # Accumulate metrics
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        n += y.size(0)

    return loss_sum / n, correct / n

# Evaluation Function 
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Optional[nn.Module] = None,
    tta: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates the model over an entire dataset (validation or test set).

    """
    model.eval()
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    all_pred, all_true = [], []

    for x, y in loader:
        x, y = _to_device(x, y)

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)

            # Test-Time Augmentation (TTA) via horizontal flip averaging
            if tta:
                x_flip = torch.flip(x, dims=[-1])  # Flip width dimension
                out_flip = model(x_flip)
                out = (out + out_flip) / 2.0

            if criterion is not None:
                loss_sum += criterion(out, y).item() * y.size(0)

        # Compute Top-1 and Top-5 accuracy matches
        top5_idx = out.topk(5, dim=1).indices
        top1 += (top5_idx[:, 0] == y).sum().item()
        top5 += (top5_idx == y.unsqueeze(1)).any(dim=1).sum().item()
        n += y.size(0)

        all_pred.append(top5_idx[:, 0].cpu())
        all_true.append(y.cpu())

    # Concatenate all predictions and ground truth labels across batches
    preds = torch.cat(all_pred).numpy()
    targets = torch.cat(all_true).numpy()

    # Calculate assignment-specific Average Per-Class Accuracy
    avg_per_class = compute_avg_per_class_acc(preds, targets, CFG.NUM_CLASSES)

    return dict(
        loss=loss_sum / n if n > 0 else 0.0,
        top1=top1 / n if n > 0 else 0.0,
        top5=top5 / n if n > 0 else 0.0,
        avg_per_class=avg_per_class,
        preds=preds,
        targets=targets,
    )


def _atomic_save(obj: Any, path: Path) -> None:
    """
    Saves an object to disk atomically by writing to a temporary file and subsequently replacing the target path
    """
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)

def evaluate_checkpoint(
    name: str,
    tag: str = "",
    ckpt_type: str = "best",
    tta: bool = True,
) -> Tuple[Dict[str, Any], ...]:
    """
    Loads an already saved checkpoint from disk and performs comprehensive test set evaluation

    Features:
      - Runs standard test evaluation and reports Top-1, Per-Class, and Top-5 accuracy
      - Runs Test-Time Augmentation (TTA) and quantifies the exact performance delta
      - Automatically saves test prediction arrays to RESULTS_DIR for subsequent
        confusion matrix analysis and fine-grained species error investigation

    """
    run = f"{name}{('_' + tag) if tag else ''}"
    ckpt_path = CFG.CKPT_DIR / f"{run}_{ckpt_type}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    seed_everything()
    _, _, test_loader = make_loaders(name)
    model = build_model(name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))

    eval_criterion = nn.CrossEntropyLoss()
    print(f"\n==================== Evaluating {run} ({ckpt_type}) ====================")

    # Standard evaluation 
    te_std = evaluate(model, test_loader, eval_criterion, tta=False)
    print(f"Standard Test : Top-1 = {te_std['top1']:.4f} ({te_std['top1']*100:.2f}%) | "
          f"Per-Class = {te_std['avg_per_class']:.4f} ({te_std['avg_per_class']*100:.2f}%) | "
          f"Top-5 = {te_std['top5']:.4f}")

    # Save test predictions for confusion matrix / error analysis
    CFG.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CFG.RESULTS_DIR / f"{run}_test_preds.npz", preds=te_std["preds"], targets=te_std["targets"])
    print(f"Saved test predictions to: {CFG.RESULTS_DIR / f'{run}_test_preds.npz'}")

    # Test-Time Augmentation (TTA) evaluation (original + horizontal flip)
    if tta:
        te_tta = evaluate(model, test_loader, eval_criterion, tta=True)
        gain = (te_tta["top1"] - te_std["top1"]) * 100.0
        gain_pc = (te_tta["avg_per_class"] - te_std["avg_per_class"]) * 100.0
        print(f"With TTA (Flip): Top-1 = {te_tta['top1']:.4f} ({te_tta['top1']*100:.2f}%) | "
              f"Per-Class = {te_tta['avg_per_class']:.4f} ({te_tta['avg_per_class']*100:.2f}%) | "
              f"Top-5 = {te_tta['top5']:.4f} [Gain: {gain:+.2f}% Top-1, {gain_pc:+.2f}% Per-Class]")
        return te_std, te_tta

    return (te_std,)

def run_experiment(
    name: str,
    tag: str = "",
    limit_batches: Optional[int] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Executes a complete training and evaluation pipeline for a given architecture.

    """
    # 1. Merge baseline defaults with run-specific hyperparameter overrides
    hp = {**DEFAULTS, **overrides}
    run = f"{name}{('_' + tag) if tag else ''}"

    # Ensure output directories exist
    CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CFG.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    last_path = CFG.CKPT_DIR / f"{run}_last.pth"
    best_path = CFG.CKPT_DIR / f"{run}_best.pth"
    hist_path = CFG.RESULTS_DIR / f"{run}_history.csv"

    # 2. Seed random generators and initialize data loaders & model architecture
    seed_everything()
    train_loader, val_loader, test_loader = make_loaders(name)
    model = build_model(name)
    total_m, _ = count_params(model)

    steps_per_epoch = len(train_loader)

    # 3. Build optimizer and learning rate scheduler
    optimizer = make_optimizer(
        model,
        name,
        hp["lr_backbone"],
        hp["lr_head"],
        hp["weight_decay"],
        layer_decay=hp.get("layer_decay", 1.0),
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps=hp["epochs"] * steps_per_epoch,
        warmup_steps=hp["warmup_epochs"] * steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda")
    eval_criterion = nn.CrossEntropyLoss()

    # 4. Check for existing checkpoint to resume an interrupted run
    start_epoch, best_acc, bad_epochs, history = 0, 0.0, 0, []
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck["best_acc"]
        bad_epochs = ck["bad_epochs"]
        history = ck["history"]
        print(f"Resuming {run} from epoch {start_epoch} (best val top-1 so far {best_acc:.4f})")

    torch.cuda.reset_peak_memory_stats()
    print(f"{run}: {total_m:.1f}M params, {steps_per_epoch} steps/epoch, batch size {CFG.BATCH_SIZE}")

    # 5. Main Training & Validation Loop
    for epoch in range(start_epoch, hp["epochs"]):
        t0 = time.time()

        # Execute training epoch
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            hp["grad_clip"],
            label_smoothing=hp["label_smoothing"],
            mixup_prob=hp.get("mixup_prob", 0.0),
            limit_batches=limit_batches,
        )

        # Execute validation epoch
        va = evaluate(model, val_loader, eval_criterion, tta=False)
        secs = time.time() - t0

        # Extract current classification head learning rate for logging
        head_lr = next(
            (g["lr"] for g in optimizer.param_groups if g.get("name") == "head"),
            optimizer.param_groups[-1]["lr"],
        )

        # Record metrics in epoch history row
        row = dict(
            epoch=epoch + 1,
            train_loss=tr_loss,
            train_acc=tr_acc,
            val_loss=va["loss"],
            val_top1=va["top1"],
            val_avg_per_class=va["avg_per_class"],
            val_top5=va["top5"],
            epoch_sec=secs,
            lr_head=head_lr,
        )
        history.append(row)

        # Check for improvement in validation Top-1 accuracy
        improved = va["top1"] > best_acc
        if improved:
            best_acc = va["top1"]
            bad_epochs = 0
            _atomic_save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        # Save last checkpoint for crash recovery
        _atomic_save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                best_acc=best_acc,
                bad_epochs=bad_epochs,
                history=history,
            ),
            last_path,
        )

        # Save history CSV log
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(history)

        print(f"[{run}] ep {epoch+1:02d}/{hp['epochs']} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va['loss']:.3f} top1 {va['top1']:.3f} per_class {va['avg_per_class']:.3f} | "
              f"{secs:.0f}s{'  *best*' if improved else ''}")

        # Check early stopping condition
        if bad_epochs >= hp["patience"]:
            print(f"Early stopping: no val improvement for {hp['patience']} consecutive epochs.")
            break

    # 6. Final Evaluation on Unseen Test Set with Best Checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    te = evaluate(model, test_loader, eval_criterion, tta=False)
    te_tta = evaluate(model, test_loader, eval_criterion, tta=True)
    macro_f1 = f1_score(te["targets"], te["preds"], average="macro")

    # 7. Compile and Serialize Full Experiment Summary
    summary = dict(
        model=name,
        params_m=round(total_m, 2),
        best_val_top1=best_acc,
        test_top1=te["top1"],
        test_avg_per_class=te["avg_per_class"],
        test_top5=te["top5"],
        test_macro_f1=macro_f1,
        test_tta_top1=te_tta["top1"],
        test_tta_avg_per_class=te_tta["avg_per_class"],
        epochs_run=len(history),
        avg_epoch_sec=float(np.mean([h["epoch_sec"] for h in history])),
        peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        batch_size=CFG.BATCH_SIZE,
        uniform_res=CFG.UNIFORM_RES,
        hyperparams=hp,
    )

    with open(CFG.RESULTS_DIR / f"{run}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save test prediction arrays for confusion matrix and error analysis
    np.savez(CFG.RESULTS_DIR / f"{run}_test_preds.npz", preds=te["preds"], targets=te["targets"])

    print(f"\n{run} FINAL TEST RESULTS | Top-1: {te['top1']:.4f} (TTA: {te_tta['top1']:.4f}) | "
          f"Average Per-Class: {te['avg_per_class']:.4f} (TTA: {te_tta['avg_per_class']:.4f}) | "
          f"Top-5: {te['top5']:.4f} | Macro-F1: {macro_f1:.4f}")

    return summary
