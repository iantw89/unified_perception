"""Training loop."""

from __future__ import annotations

import csv
import math
import os
import random
import time
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from .._compat import get_tqdm
from ..config import Config, config_to_dict, save_config
from ..data import MultiViewPoseDataset, ShuffleBufferDataset, collate_batch
from ..data.registry import build_provider
from ..models import SetCriterion, UnifiedPerceptionModel
from .evaluate import run_eval
from .step import CUDAPrefetcher, TrainStep
from .train_viz import save_train_viz

tqdm = get_tqdm()


def _worker_init_fn(worker_id: int) -> None:
    """Ensure FFmpeg shared DLLs are on the search path in each DataLoader worker (Windows)."""
    import sys as _sys
    import os as _os
    if _sys.platform == "win32" and hasattr(_os, "add_dll_directory"):
        ffmpeg_bin = _os.environ.get("FFMPEG_SHARED_BIN", "")
        if ffmpeg_bin and _os.path.isdir(ffmpeg_bin):
            _os.add_dll_directory(ffmpeg_bin)


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
    worker_count = cfg.data.num_workers
    prefetch_factor = 2 if worker_count > 0 else None
    init_fn = _worker_init_fn if worker_count > 0 else None

    if cfg.data.use_shuffle_buffer:
        # IterableDataset: sequences fully decoded into memory; windows assembled
        # by array-slicing.  Each DataLoader worker maintains its own buffer with
        # a background decode thread, eliminating per-window video-seek overhead.
        provider = build_provider(cfg.data.provider, cfg, "train")
        all_seq_ids = list({s[0] for s in provider.list_samples()})
        train_set = ShuffleBufferDataset(provider, all_seq_ids, cfg)
        loaders = {
            "train": DataLoader(
                train_set,
                batch_size=cfg.data.batch_size,
                num_workers=worker_count,
                collate_fn=collate_batch,
                drop_last=True,
                pin_memory=True,
                persistent_workers=(worker_count > 0),
                prefetch_factor=prefetch_factor,
                worker_init_fn=init_fn,
            )
        }
    else:
        train_set = MultiViewPoseDataset(cfg, split="train")
        loaders = {
            "train": DataLoader(
                train_set,
                batch_size=cfg.data.batch_size,
                shuffle=True,
                num_workers=worker_count,
                collate_fn=collate_batch,
                drop_last=True,
                pin_memory=True,
                persistent_workers=(worker_count > 0),
                prefetch_factor=prefetch_factor,
                worker_init_fn=init_fn,
            )
        }

    val_nw = cfg.data.num_workers // 2
    provider = build_provider(cfg.data.provider, cfg, "val")
    all_seq_ids = list({s[0] for s in provider.list_samples()})
    val_set = ShuffleBufferDataset(provider, all_seq_ids, cfg)
    loaders["val"] = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=val_nw,
        collate_fn=collate_batch,
        drop_last=True,
        prefetch_factor=prefetch_factor if val_nw > 0 else None,
        pin_memory=True,
        worker_init_fn=init_fn if val_nw > 0 else None,
    )
    return loaders


def _per_module_grad_norms(model: torch.nn.Module) -> list[tuple[str, float]]:
    """Return (name, l2-norm) for every leaf parameter with a gradient, sorted descending."""
    norms: list[tuple[str, float]] = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms.append((name, float(p.grad.data.norm(2))))
    return sorted(norms, key=lambda x: x[1], reverse=True)


def _gpu_mem_gb() -> tuple[float, float]:
    """Return (allocated_GB, reserved_GB) for the current CUDA device."""
    if torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved()  / 1e9,
        )
    return 0.0, 0.0


def train(cfg: Config):
    device = resolve_device(cfg.train.device)
    set_seed(cfg.train.seed)
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.train.output_dir, "config.yaml"))

    loaders = build_loaders(cfg)
    train_loader = (
        CUDAPrefetcher(loaders["train"], device)
        if device.type == "cuda" else loaders["train"]
    )
    model = UnifiedPerceptionModel(cfg).to(device)
    # Normalisation constants on device for uint8 → float conversion.
    # Applied after each batch is moved to device, avoiding float32 IPC traffic.
    _norm_mean = torch.tensor(cfg.data.pixel_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3, 1, 1)
    _norm_std  = torch.tensor(cfg.data.pixel_std,  dtype=torch.float32, device=device).view(1, 1, 1, 3, 1, 1)
    criterion = SetCriterion(cfg.model)

    if cfg.train.compile_model:
        if hasattr(torch, "compile"):
            torch._dynamo.config.capture_scalar_outputs = True
            print("[train] torch.compile(model)  (first steps will be slower during kernel compilation)")
            model = cast(torch.nn.Module, torch.compile(model))
            criterion = cast(torch.nn.Module, torch.compile(criterion))
        else:  # pragma: no cover
            print("[train] torch.compile not available (PyTorch < 2.0) — skipping")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # AMP: bfloat16 on CUDA for ~2× throughput on Tensor Cores.
    # bfloat16 has the same exponent range as float32 (no overflow in attention),
    # so no GradScaler is needed.
    use_amp = device.type == "cuda"

    ts = TrainStep(model, criterion, optimizer, _norm_mean, _norm_std, device, use_amp)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(cfg.train.output_dir)
    except Exception:  # pragma: no cover
        writer = None

    csv_path = os.path.join(cfg.train.output_dir, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "split", "loss", "loss_cls", "loss_kpt",
                          "grad_norm", "gpu_alloc_gb_avg", "gpu_alloc_gb_peak",
                          "mpjpe_mm", "f1"])

    start_epoch = 0
    global_step = 0
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        ckpt = torch.load(cfg.train.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {cfg.train.resume} at epoch {start_epoch}")

    best_mpjpe = float("inf")
    loss_ema: float | None = None          # global EMA for divergence detection
    _EMA_DECAY     = 0.98
    _SPIKE_FACTOR  = 8.0                   # flag if loss > 8× EMA
    _EMA_WARMUP    = 50                    # steps before spike detection activates
    # Skip threshold: only bail on non-finite gradients.
    # grad_clip already bounds the weight update magnitude; finite gnorm (even
    # large) still produces a valid normalised gradient direction.
    # Separate "warn" threshold logs the top-8 modules without skipping.
    _skip_thresh    = float("inf")         # skip only NaN/Inf
    _warn_thresh    = 2000.0   # print per-module diagnostics only above this gnorm

    epoch_peak_mem_gb: float = 0.0
    for epoch in range(start_epoch, cfg.train.max_epochs):
        model.train()
        running = {}
        epoch_peak_mem_gb = 0.0
        t_epoch_start = time.perf_counter()
        steps_this_epoch = 0
        skipped_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for it, batch in enumerate(pbar):
            # Determine whether this step needs 2-D detections for visualisation
            # so we can fold it into the single training forward pass.
            viz_batch_interval = cfg.train.viz_batch_interval
            do_viz = viz_batch_interval > 0 and global_step % viz_batch_interval == 0

            # Snapshot raw images BEFORE h2d/normalize so visualization has
            # un-normalised pixels.  CUDAPrefetcher makes batch["imgs"] a
            # float32 GPU tensor; sub_/div_ in ts.normalize() would clobber it
            # in-place, so we must copy to CPU first.
            _viz_imgs_cpu = batch["imgs"].cpu() if do_viz else None

            ts.h2d(batch)
            ts.normalize()
            ts.forward(return_2d=do_viz)
            loss_val = ts.compute_loss()

            # ---- Loss divergence check (before backward to save compute) ----
            if not math.isfinite(loss_val):
                print(
                    f"\n[step {global_step}] LOSS={loss_val} — NaN/Inf detected,"
                    f" skipping backward (epoch {epoch}, batch {it})"
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                skipped_steps += 1
                continue

            # Update loss EMA
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = _EMA_DECAY * loss_ema + (1 - _EMA_DECAY) * loss_val
            if global_step >= _EMA_WARMUP and loss_val > _SPIKE_FACTOR * loss_ema:
                print(
                    f"\n[step {global_step}] LOSS SPIKE: {loss_val:.4f} vs EMA"
                    f" {loss_ema:.4f} (×{loss_val / loss_ema:.1f})"
                )

            # LR warmup.
            if global_step < cfg.train.warmup_iters:
                scale = (global_step + 1) / cfg.train.warmup_iters
                for g in optimizer.param_groups:
                    g["lr"] = cfg.train.lr * scale

            # Free input tensors before backward — they are not needed for
            # gradient computation (YOLO is frozen; world2img/intrinsics are
            # data tensors with requires_grad=False).
            ts.free_inputs()
            grad_norm = ts.backward()

            # ---- Gradient explosion check -----------------------------------
            if not math.isfinite(grad_norm):
                module_norms = _per_module_grad_norms(model)
                print(
                    f"\n[step {global_step}] NON-FINITE GRAD gnorm={grad_norm}"
                    f" — skipping optimizer step.  Top-8 modules:"
                )
                for mn, mv in module_norms[:8]:
                    print(f"    {mv:12.2f}  {mn}")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                skipped_steps += 1
                continue

            if grad_norm >= _warn_thresh:
                module_norms = _per_module_grad_norms(model)
                print(
                    f"\n[step {global_step}] HIGH GRAD gnorm={grad_norm:.1f}"
                    f" (clip={cfg.train.grad_clip}).  Top-8 modules:"
                )
                for mn, mv in module_norms[:8]:
                    print(f"    {mv:12.2f}  {mn}")

            ts.optimize(cfg.train.grad_clip)
            global_step += 1
            steps_this_epoch += 1

            # Per-step visualisation: reuse cls_scores / kpts_norm from the
            # training forward (no second forward pass, no model.eval() toggle).
            # YOLO is always self-eval inside backbone.forward, so 2-D detections
            # are identical to eval-mode regardless of model.training state.
            if do_viz:
                try:
                    _B = batch["imgs"].shape[0]
                    _T = batch["imgs"].shape[1]
                    _N = batch["imgs"].shape[2]
                    assert ts.kpts_norm is not None and ts.cls_scores is not None
                    _kpts_cpu = ts.kpts_norm.detach().float().cpu()   # (L, B, T, Q, J, 3)
                    _cls_cpu  = ts.cls_scores.detach().float().cpu()  # (L, B, T, Q, 1)
                    # Build one viz_example per batch element.
                    # dets_2d flat layout: index = t*B*N + b*N + c
                    viz_list = []
                    for _b in range(_B):
                        _vb_cpu = {k: v[_b:_b+1].cpu() if isinstance(v, torch.Tensor) else v
                                   for k, v in batch.items()}
                        # Override imgs with the pre-normalisation snapshot.
                        if _viz_imgs_cpu is not None:
                            _vb_cpu["imgs"] = _viz_imgs_cpu[_b:_b+1]
                        # YOLO dets are plain dicts of numpy arrays — no .cpu() needed.
                        _yolo_b = [ts.dets_2d[_t * _B * _N + _b * _N + _c]
                                   for _t in range(_T) for _c in range(_N)]
                        viz_list.append((
                            _vb_cpu,
                            _kpts_cpu[:, _b:_b+1],
                            _cls_cpu[:, _b:_b+1],
                            _yolo_b,
                        ))
                    _step_out = os.path.join(
                        cfg.train.output_dir, "viz", f"step_{global_step:07d}"
                    )
                    save_train_viz(
                        _step_out, viz_list, model, epoch=epoch,
                        extra_text=f"step={global_step}  loss={loss_val:.3f}",
                    )
                except Exception as _exc:
                    print(f"  [viz/step] skipped: {_exc}")

            # ---- Track GPU memory -------------------------------------------
            alloc_gb, rsvd_gb = _gpu_mem_gb()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            epoch_peak_mem_gb = max(epoch_peak_mem_gb, peak_gb)

            ts.metrics["grad_norm"] = grad_norm
            ts.metrics["gpu_alloc_gb"] = alloc_gb

            for k, v in ts.metrics.items():
                running[k] = running.get(k, 0.0) + (v if isinstance(v, float) else 0.0)
            if it % cfg.train.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{loss_val:.3f}",
                    ema=f"{loss_ema:.3f}" if loss_ema is not None else "—",
                    kpt=f"{ts.metrics['loss_kpt']:.3f}",
                    gnorm=f"{grad_norm:.2f}",
                    mem=f"{alloc_gb:.1f}/{rsvd_gb:.1f}G",
                    peak=f"{peak_gb:.1f}G",
                )
                if writer:
                    for k, v in ts.metrics.items():
                        writer.add_scalar(f"train/{k}", v, global_step)
                    writer.add_scalar("sys/gpu_alloc_gb", alloc_gb, global_step)
                    writer.add_scalar("sys/gpu_rsvd_gb",  rsvd_gb,  global_step)
                    if loss_ema is not None:
                        writer.add_scalar("train/loss_ema", loss_ema, global_step)

        n = max(steps_this_epoch, 1)
        train_loss = running.get("loss", 0.0) / n
        avg_grad   = running.get("grad_norm", 0.0) / n
        elapsed    = time.perf_counter() - t_epoch_start
        sps        = steps_this_epoch / max(elapsed, 1e-3)
        print(
            f"[epoch {epoch}] loss={train_loss:.4f}  ema={loss_ema:.4f}"
            f"  gnorm={avg_grad:.3f}  peak_mem={epoch_peak_mem_gb:.2f} GB"
            f"  steps={steps_this_epoch}  skipped={skipped_steps}"
            f"  {sps:.1f} step/s  ({elapsed:.0f}s)"
        )
        csv_writer.writerow([epoch, "train", train_loss,
                              running.get("loss_cls", 0) / n,
                              running.get("loss_kpt",  0) / n,
                              avg_grad,
                              running.get("gpu_alloc_gb", 0) / n,
                              epoch_peak_mem_gb,
                              "", ""])

        # Periodic evaluation.  Use (epoch+1) % interval so epoch 0 is never
        # accidentally evaluated when a large interval like 9999 is set
        # (0 % N == 0 is always true).  Setting eval_interval=0 disables eval.
        if loaders["val"] is not None and cfg.train.eval_interval > 0 and (epoch + 1) % cfg.train.eval_interval == 0:
            max_vb = getattr(cfg.eval, "max_eval_batches", 0) or None
            val_metrics = run_eval(model, loaders["val"], cfg, device, max_batches=max_vb)
            print(f"[epoch {epoch}] val: " + ", ".join(f"{k}={v:.3f}" for k, v in val_metrics.items()))
            if writer:
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/{k}", v, global_step)
            csv_writer.writerow([epoch, "val", "", "", "", "", "", "",
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