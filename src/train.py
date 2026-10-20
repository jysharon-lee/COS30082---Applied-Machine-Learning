import csv, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.config import CFG, device, seed_everything
from src.data import make_loaders
from src.models import build_model, count_params, HEAD_KEY

# Identical defaults so model comparisons are fair
DEFAULTS = dict(
    epochs=30,
    lr_backbone=1e-4,
    lr_head=1e-3,
    weight_decay=0.05,
    warmup_epochs=2,
    label_smoothing=0.1,
    mixup_prob=0.0,       # 0.0 default; 0.2-0.3 for gentle Mixup
    patience=8,           # early stopping on validation top-1
    grad_clip=1.0,
)


def mixup_cutmix_batch(x, y, num_classes=200, mixup_alpha=0.2, prob=0.0):
    """Applies gentle Mixup (no CutMix) with probability `prob` to avoid occluding fine-grained bird features."""
    if prob <= 0 or np.random.rand() > prob:
        y_onehot = torch.zeros(y.size(0), num_classes, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
        return x, y_onehot

    y_onehot = torch.zeros(y.size(0), num_classes, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    perm = torch.randperm(x.size(0))
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[perm]
    x_mixed = lam * x + (1 - lam) * x[perm]
    return x_mixed, y_mixed


def compute_avg_per_class_acc(preds, targets, num_classes=200):
    """
    Computes Average accuracy per class: Ave = (1/C) * sum(T_i)
    as specified in Section 4 of the assignment brief.
    """
    accs = []
    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            accs.append((preds[mask] == targets[mask]).mean())
    return float(np.mean(accs))


def make_optimizer(model, name, lr_backbone, lr_head, weight_decay):
    key = HEAD_KEY[name]
    head, backbone = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (head if n.startswith(key) else backbone).append(p)
    return torch.optim.AdamW(
        [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}],
        weight_decay=weight_decay)


def make_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _to_device(x, y):
    x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    return x, y.to(device, non_blocking=True)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, grad_clip,
                    label_smoothing=0.1, mixup_prob=0.0, limit_batches=None):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x, y = _to_device(x, y)
        if mixup_prob > 0:
            x, y_soft = mixup_cutmix_batch(x, y, num_classes=CFG.NUM_CLASSES, prob=mixup_prob)
        else:
            y_soft = y

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y_soft, label_smoothing=label_smoothing)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion=None, tta=False):
    """Evaluates the model. If tta=True, averages logits of original and horizontally flipped images."""
    model.eval()
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = _to_device(x, y)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            if tta:
                # Test-Time Augmentation (TTA) via horizontal flip
                x_flip = torch.flip(x, dims=[-1])
                out_flip = model(x_flip)
                out = (out + out_flip) / 2.0

            if criterion is not None:
                loss_sum += criterion(out, y).item() * y.size(0)

        top5_idx = out.topk(5, dim=1).indices
        top1 += (top5_idx[:, 0] == y).sum().item()
        top5 += (top5_idx == y.unsqueeze(1)).any(1).sum().item()
        n += y.size(0)
        all_pred.append(top5_idx[:, 0].cpu())
        all_true.append(y.cpu())

    preds = torch.cat(all_pred).numpy()
    targets = torch.cat(all_true).numpy()
    avg_per_class = compute_avg_per_class_acc(preds, targets, CFG.NUM_CLASSES)

    return dict(loss=loss_sum / n, top1=top1 / n, top5=top5 / n,
                avg_per_class=avg_per_class, preds=preds, targets=targets)


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)   # avoids a corrupt checkpoint if Colab disconnects mid-write


def evaluate_checkpoint(name: str, tag: str = "", ckpt_type: str = "best", tta: bool = True):
    """
    Evaluates an already saved checkpoint on the test set.
    Provides standard test evaluation and Test-Time Augmentation (TTA) in one go.
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

    # 1. Standard evaluation
    te_std = evaluate(model, test_loader, eval_criterion, tta=False)
    print(f"Standard Test : Top-1 = {te_std['top1']:.4f} ({te_std['top1']*100:.2f}%) | "
          f"Per-Class = {te_std['avg_per_class']:.4f} ({te_std['avg_per_class']*100:.2f}%) | "
          f"Top-5 = {te_std['top5']:.4f}")

    # 2. TTA evaluation
    if tta:
        te_tta = evaluate(model, test_loader, eval_criterion, tta=True)
        gain = (te_tta["top1"] - te_std["top1"]) * 100
        gain_pc = (te_tta["avg_per_class"] - te_std["avg_per_class"]) * 100
        print(f"With TTA (Flip): Top-1 = {te_tta['top1']:.4f} ({te_tta['top1']*100:.2f}%) | "
              f"Per-Class = {te_tta['avg_per_class']:.4f} ({te_tta['avg_per_class']*100:.2f}%) | "
              f"Top-5 = {te_tta['top5']:.4f} [Gain: {gain:+.2f}% Top-1, {gain_pc:+.2f}% Per-Class]")
        return te_std, te_tta

    return te_std


def run_experiment(name, tag="", limit_batches=None, **overrides):
    """Train one model end-to-end, checkpoint every epoch, resume automatically,
    then evaluate the best checkpoint on the test set (with and without TTA)."""
    hp = {**DEFAULTS, **overrides}
    run = f"{name}{('_' + tag) if tag else ''}"
    CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CFG.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    last_path = CFG.CKPT_DIR / f"{run}_last.pth"
    best_path = CFG.CKPT_DIR / f"{run}_best.pth"
    hist_path = CFG.RESULTS_DIR / f"{run}_history.csv"

    seed_everything()
    train_loader, val_loader, test_loader = make_loaders(name)
    model = build_model(name)
    total_m, _ = count_params(model)

    steps_per_epoch = len(train_loader)
    optimizer = make_optimizer(model, name, hp["lr_backbone"], hp["lr_head"], hp["weight_decay"])
    scheduler = make_scheduler(optimizer, hp["epochs"] * steps_per_epoch, hp["warmup_epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")
    eval_criterion = nn.CrossEntropyLoss()

    start_epoch, best_acc, bad_epochs, history = 0, 0.0, 0, []
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        bad_epochs, history = ck["bad_epochs"], ck["history"]
        print(f"Resuming {run} from epoch {start_epoch} (best val top-1 so far {best_acc:.4f})")

    torch.cuda.reset_peak_memory_stats()
    print(f"{run}: {total_m:.1f}M params, {steps_per_epoch} steps/epoch, batch {CFG.BATCH_SIZE}")

    for epoch in range(start_epoch, hp["epochs"]):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler,
                                          scaler, hp["grad_clip"],
                                          label_smoothing=hp["label_smoothing"],
                                          mixup_prob=hp.get("mixup_prob", 0.0),
                                          limit_batches=limit_batches)
        va = evaluate(model, val_loader, eval_criterion, tta=False)
        secs = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=tr_loss, train_acc=tr_acc, val_loss=va["loss"],
                   val_top1=va["top1"], val_avg_per_class=va["avg_per_class"], val_top5=va["top5"],
                   epoch_sec=secs, lr_head=optimizer.param_groups[1]["lr"])
        history.append(row)

        improved = va["top1"] > best_acc
        if improved:
            best_acc, bad_epochs = va["top1"], 0
            _atomic_save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        _atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                          epoch=epoch, best_acc=best_acc, bad_epochs=bad_epochs, history=history),
                     last_path)
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader(); w.writerows(history)

        print(f"[{run}] ep {epoch+1:02d}/{hp['epochs']} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va['loss']:.3f} top1 {va['top1']:.3f} per_class {va['avg_per_class']:.3f} | "
              f"{secs:.0f}s{'  *best*' if improved else ''}")

        if bad_epochs >= hp["patience"]:
            print(f"Early stopping: no val improvement for {hp['patience']} epochs.")
            break

    # ---- Final test evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    te = evaluate(model, test_loader, eval_criterion, tta=False)
    te_tta = evaluate(model, test_loader, eval_criterion, tta=True)
    macro_f1 = f1_score(te["targets"], te["preds"], average="macro")

    summary = dict(
        model=name, params_m=round(total_m, 2), best_val_top1=best_acc,
        test_top1=te["top1"], test_avg_per_class=te["avg_per_class"],
        test_top5=te["top5"], test_macro_f1=macro_f1,
        test_tta_top1=te_tta["top1"], test_tta_avg_per_class=te_tta["avg_per_class"],
        epochs_run=len(history),
        avg_epoch_sec=float(np.mean([h["epoch_sec"] for h in history])),
        peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        batch_size=CFG.BATCH_SIZE, uniform_res=CFG.UNIFORM_RES, hyperparams=hp,
    )
    with open(CFG.RESULTS_DIR / f"{run}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(CFG.RESULTS_DIR / f"{run}_test_preds.npz", preds=te["preds"], targets=te["targets"])

    print(f"\n{run} TEST | top1 {te['top1']:.4f} (TTA: {te_tta['top1']:.4f}) | "
          f"avg_per_class {te['avg_per_class']:.4f} (TTA: {te_tta['avg_per_class']:.4f}) | "
          f"top5 {te['top5']:.4f} | macro-F1 {macro_f1:.4f}")
    return summary
