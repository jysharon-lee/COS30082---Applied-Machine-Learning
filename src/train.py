import csv, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.config import CFG, device, seed_everything
from src.data import make_loaders
from src.models import build_model, count_params, HEAD_KEY

# Identical for all three models so the comparison is fair
DEFAULTS = dict(
    epochs=30,
    lr_backbone=1e-4,
    lr_head=1e-3,
    weight_decay=0.05,
    warmup_epochs=2,
    label_smoothing=0.1,
    patience=8,           # early stopping on validation top-1
    grad_clip=1.0,
)


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


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, grad_clip, limit_batches=None):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x, y = _to_device(x, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
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
def evaluate(model, loader, criterion=None):
    model.eval()
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = _to_device(x, y)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            if criterion is not None:
                loss_sum += criterion(out, y).item() * y.size(0)
        top5_idx = out.topk(5, dim=1).indices
        top1 += (top5_idx[:, 0] == y).sum().item()
        top5 += (top5_idx == y.unsqueeze(1)).any(1).sum().item()
        n += y.size(0)
        all_pred.append(top5_idx[:, 0].cpu())
        all_true.append(y.cpu())
    return dict(loss=loss_sum / n, top1=top1 / n, top5=top5 / n,
                preds=torch.cat(all_pred).numpy(), targets=torch.cat(all_true).numpy())


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)   # avoids a corrupt checkpoint if Colab disconnects mid-write


def run_experiment(name, tag="", limit_batches=None, **overrides):
    """Train one model end-to-end, checkpoint every epoch, resume automatically,
    then evaluate the best checkpoint on the test set.
    tag: suffix for file names (use tag='smoke' for quick tests)."""
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
    criterion = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
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
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                          scaler, hp["grad_clip"], limit_batches)
        va = evaluate(model, val_loader, eval_criterion)
        secs = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=tr_loss, train_acc=tr_acc, val_loss=va["loss"],
                   val_top1=va["top1"], val_top5=va["top5"], epoch_sec=secs,
                   lr_head=optimizer.param_groups[1]["lr"])
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
              f"val loss {va['loss']:.3f} top1 {va['top1']:.3f} top5 {va['top5']:.3f} | "
              f"{secs:.0f}s{'  *best*' if improved else ''}")

        if bad_epochs >= hp["patience"]:
            print(f"Early stopping: no val improvement for {hp['patience']} epochs.")
            break

    # ---- Final test evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    te = evaluate(model, test_loader, eval_criterion)
    macro_f1 = f1_score(te["targets"], te["preds"], average="macro")
    summary = dict(
        model=name, params_m=round(total_m, 2), best_val_top1=best_acc,
        test_top1=te["top1"], test_top5=te["top5"], test_macro_f1=macro_f1,
        epochs_run=len(history),
        avg_epoch_sec=float(np.mean([h["epoch_sec"] for h in history])),
        peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        batch_size=CFG.BATCH_SIZE, uniform_res=CFG.UNIFORM_RES, hyperparams=hp,
    )
    with open(CFG.RESULTS_DIR / f"{run}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(CFG.RESULTS_DIR / f"{run}_test_preds.npz", preds=te["preds"], targets=te["targets"])

    print(f"\n{run} TEST | top1 {te['top1']:.4f} | top5 {te['top5']:.4f} | macro-F1 {macro_f1:.4f}")
    return summary