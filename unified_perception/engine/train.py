"""Training loop."""

from __future__ import annotations

import csv
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from .._compat import get_tqdm
from ..config import Config, config_to_dict, save_config
from ..data import MultiViewPoseDataset, collate_batch
from ..models import SetCriterion, UnifiedPerceptionModel
from .evaluate import run_eval

tqdm = get_tqdm()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: Config):
    train_set = MultiViewPoseDataset(cfg, split="train")
    _nw = cfg.data.num_workers
    _pf = 2 if _nw > 0 else None
    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=_nw,
            collate_fn=collate_batch,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(_nw > 0),
            prefetch_factor=_pf,
        )
    }
    try:
        val_set = MultiViewPoseDataset(cfg, split="val")
        loaders["val"] = DataLoader(
            val_set,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_batch,
        )
    except (KeyError, RuntimeError):
        loaders["val"] = None
    return loaders


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _memory_stats(model) -> dict:
    """Return number of populated memory slots in the SparseHead (last forward)."""
    head = model.head
    mem = getattr(head, "memory_embedding", None)
    if mem is None:
        return {}
    # memory_embedding shape: (B, memory_len, C); count non-zero rows per batch
    with torch.no_grad():
        populated = (mem.abs().sum(-1) > 0).float().mean().item()
    return {"memory_fill": populated}


def train(cfg: Config):
    device = resolve_device(cfg.train.device)
    set_seed(cfg.train.seed)
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.train.output_dir, "config.yaml"))

    loaders = build_loaders(cfg)
    model = UnifiedPerceptionModel(cfg).to(device)
    criterion = SetCriterion(cfg.model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # Optional training visualisation helper.
    viz_interval = getattr(cfg.train, "viz_interval", 0)
    viz_batch_cpu = None  # cached CPU batch for visualisation

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(cfg.train.output_dir)
    except Exception:  # pragma: no cover
        writer = None

    csv_path = os.path.join(cfg.train.output_dir, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "split", "loss", "loss_cls", "loss_kpt", "grad_norm", "mpjpe_mm", "f1"])

    start_epoch = 0
    global_step = 0
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        ckpt = torch.load(cfg.train.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {cfg.train.resume} at epoch {start_epoch}")

    best_mpjpe = float("inf")
    for epoch in range(start_epoch, cfg.train.max_epochs):
        model.train()
        running = {}
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}")
        for it, batch in enumerate(pbar):
            # Cache first batch for visualisation (CPU copy, no grad).
            if viz_batch_cpu is None:
                viz_batch_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}

            imgs = batch["imgs"].to(device)
            world2img = batch["world2img"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            gt_mm = batch["gt_keypoints"].to(device)
            gt_valid = batch["gt_valid"].to(device)
            gt_norm = model.normalize_keypoints(gt_mm)

            cls_scores, kpts_norm, dn_meta = model(
                imgs, world2img, intrinsics, gt_norm, gt_valid
            )
            loss, metrics = criterion(cls_scores, kpts_norm, gt_norm, gt_valid, dn_meta)

            # LR warmup.
            if global_step < cfg.train.warmup_iters:
                scale = (global_step + 1) / cfg.train.warmup_iters
                for g in optimizer.param_groups:
                    g["lr"] = cfg.train.lr * scale

            optimizer.zero_grad()
            loss.backward()
            grad_norm = _grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            global_step += 1

            metrics["grad_norm"] = grad_norm
            metrics.update(_memory_stats(model))

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + (v if isinstance(v, float) else 0.0)
            if it % cfg.train.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.3f}",
                    kpt=f"{metrics['loss_kpt']:.3f}",
                    gnorm=f"{grad_norm:.2f}",
                )
                if writer:
                    for k, v in metrics.items():
                        writer.add_scalar(f"train/{k}", v, global_step)

        # Training visualisation.
        if viz_interval > 0 and epoch % viz_interval == 0 and viz_batch_cpu is not None:
            try:
                # Import lazily (cv2 optional on headless servers).
                _tools = os.path.join(os.path.dirname(__file__), "..", "..", "tools")
                if _tools not in sys.path:
                    sys.path.insert(0, os.path.abspath(_tools))
                from train_viz import save_train_viz  # type: ignore
                model.eval()
                with torch.no_grad():
                    vb = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in viz_batch_cpu.items()}
                    viz_cls, viz_kpts, _ = model(vb["imgs"], vb["world2img"], vb["intrinsics"])
                out_img = os.path.join(
                    cfg.train.output_dir, "viz", f"epoch_{epoch:04d}.png"
                )
                avg_loss = running.get("loss", 0.0) / max(len(loaders["train"]), 1)
                n_train = len(loaders["train"])
                extra = f"loss={avg_loss:.3f}  gnorm={running.get('grad_norm', 0.0)/max(n_train, 1):.1f}"
                canvas_bgr = save_train_viz(out_img, viz_batch_cpu, viz_kpts, model,
                                            cls_scores=viz_cls, epoch=epoch, extra_text=extra)
                model.train()
                print(f"  [viz] saved {out_img}")
                if writer is not None:
                    # TensorBoard expects (C,H,W) uint8
                    import cv2 as _cv2
                    canvas_rgb = _cv2.cvtColor(canvas_bgr, _cv2.COLOR_BGR2RGB)
                    writer.add_image("train/viz", canvas_rgb.transpose(2,0,1), epoch)
            except Exception as exc:
                print(f"  [viz] skipped: {exc}")

        n = len(loaders["train"])
        train_loss = running.get("loss", 0.0) / max(n, 1)
        avg_grad = running.get("grad_norm", 0.0) / max(n, 1)
        csv_writer.writerow([epoch, "train", train_loss,
                              running.get("loss_cls", 0) / n,
                              running.get("loss_kpt", 0) / n,
                              avg_grad, "", ""])

        # Periodic evaluation.
        if loaders["val"] is not None and (epoch % cfg.train.eval_interval == 0):
            max_vb = getattr(cfg.eval, "max_eval_batches", 0) or None
            val_metrics = run_eval(model, loaders["val"], cfg, device, max_batches=max_vb)
            print(f"[epoch {epoch}] val: " + ", ".join(f"{k}={v:.3f}" for k, v in val_metrics.items()))
            if writer:
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/{k}", v, global_step)
            csv_writer.writerow([epoch, "val", "", "", "", "",
                                  val_metrics["mpjpe_mm"], val_metrics["f1"]])
            csv_file.flush()
            if val_metrics["matched_persons"] > 0 and val_metrics["mpjpe_mm"] < best_mpjpe:
                best_mpjpe = val_metrics["mpjpe_mm"]
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "cfg": config_to_dict(cfg)},
                    os.path.join(cfg.train.output_dir, "best.pt"),
                )

        if epoch % cfg.train.ckpt_interval == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "cfg": config_to_dict(cfg)},
                os.path.join(cfg.train.output_dir, "last.pt"),
            )

    csv_file.close()
    if writer:
        writer.close()
    print(f"Training complete. Best val MPJPE: {best_mpjpe:.2f} mm")
    return model