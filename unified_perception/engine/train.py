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


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the eager module behind torch.compile, so checkpoints carry plain keys."""
    return getattr(model, "_orig_mod", model)


def _strip_compile_prefix(state: dict) -> dict:
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}


def _save_checkpoint(path: str, payload: dict) -> None:
    """Write atomically so killing the run mid-save never corrupts the previous file."""
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_loaders(cfg: Config):
    from ..data.streaming import (
        StreamingBatchSampler, StreamingDataset, chunk_policies, streaming_enabled, training_components,
    )
    worker_count = cfg.data.num_workers
    prefetch_factor = 2 if worker_count > 0 else None
    init_fn = _worker_init_fn if worker_count > 0 else None

    if cfg.data.use_shuffle_buffer and not cfg.data.composition_manifest:
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
    elif streaming_enabled(cfg):
        # Lanes walk consecutive windows of a clip; the model carries its
        # temporal/track state across batches (data/streaming.py).
        components = training_components(cfg)
        dataset = StreamingDataset(components, cfg.data.camera_dropout_prob, cfg.data.camera_dropout_min_views)
        loaders = {
            "train": DataLoader(
                dataset,
                batch_sampler=StreamingBatchSampler(
                    components, cfg.data.batch_size, max(1, len(dataset) // cfg.data.batch_size),
                    chunk_policies(cfg, components), seed=cfg.train.seed,
                ),
                num_workers=worker_count,
                collate_fn=collate_batch,
                pin_memory=True,
                persistent_workers=(worker_count > 0),
                prefetch_factor=prefetch_factor,
                worker_init_fn=init_fn,
            )
        }
    else:
        if cfg.data.provider == "gym_mocap_mix":
            from ..data.gym_pseudo3d import GymMocapMixDataset
            train_set = GymMocapMixDataset(cfg, split="train")
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
    if cfg.data.composition_manifest:
        val_set = MultiViewPoseDataset(cfg, split="val")
    else:
        provider = build_provider(cfg.data.provider, cfg, "val")
        all_seq_ids = list({s[0] for s in provider.list_samples()})
        val_set = ShuffleBufferDataset(provider, all_seq_ids, cfg)
    loaders["val"] = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=val_nw,
        collate_fn=collate_batch,
        drop_last=not bool(cfg.data.composition_manifest),
        prefetch_factor=prefetch_factor if val_nw > 0 else None,
        pin_memory=True,
        worker_init_fn=init_fn if val_nw > 0 else None,
    )
    if cfg.data.provider == "gym_mocap_mix" and cfg.model.track_queries:
        # Identity continuity needs whole clips: stream every held-out clip once.
        from ..data.streaming import StreamingDataset, aligned_component, evaluation_lanes, gym_component
        streams = {"val_gym": ([gym_component(cfg, "val")], None)}
        if cfg.data.gym_mix_aligned_dirs and cfg.data.gym_mix_aligned_fraction > 0:
            streams["val_aligned"] = ([aligned_component(cfg, "val", cache=True)], 512)
        for name, (components, cap) in streams.items():
            loaders[name] = DataLoader(
                StreamingDataset(components),
                batch_sampler=evaluation_lanes(components, cfg.data.batch_size, cap),
                num_workers=val_nw,
                collate_fn=collate_batch,
                prefetch_factor=prefetch_factor if val_nw > 0 else None,
                pin_memory=True,
                worker_init_fn=init_fn if val_nw > 0 else None,
            )
    elif cfg.data.provider == "gym_mocap_mix":
        # Separate held-out gym tail, evaluated alongside mocap val as "val_gym".
        from ..data.gym_pseudo3d import GymPseudo3DDataset
        loaders["val_gym"] = DataLoader(
            GymPseudo3DDataset(cfg, split="val"),
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=val_nw,
            collate_fn=collate_batch,
            drop_last=False,
            prefetch_factor=prefetch_factor if val_nw > 0 else None,
            pin_memory=True,
            worker_init_fn=init_fn if val_nw > 0 else None,
        )
        if cfg.data.gym_mix_aligned_dirs and cfg.data.gym_mix_aligned_fraction > 0:
            # Held-out single-person aligned windows (fixed random subset), as "val_aligned".
            from ..data.gym_pseudo3d import AlignedSingleDataset
            loaders["val_aligned"] = DataLoader(
                AlignedSingleDataset(cfg, split="val", limit=512),
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=val_nw,
                collate_fn=collate_batch,
                drop_last=False,
                prefetch_factor=prefetch_factor if val_nw > 0 else None,
                pin_memory=True,
                worker_init_fn=init_fn if val_nw > 0 else None,
            )
    return loaders


def _nonfinite_report(ts: "TrainStep", batch: dict, out_dir: str, step: int) -> None:
    """Describe a non-finite loss (terms, outputs, inputs) and save the batch without images."""
    terms = {k: v for k, v in (ts.metrics or {}).items() if isinstance(v, float) and not math.isfinite(v)}
    print(f"    non-finite loss terms: {terms or 'none reported'}")
    for name in ("cls_scores", "kpts_norm"):
        value = getattr(ts, name, None)
        if isinstance(value, torch.Tensor):
            bad = ~torch.isfinite(value.float())
            if bad.any():
                layers = bad.flatten(1).any(1).nonzero().flatten().tolist() if value.dim() > 1 else []
                lanes = bad.transpose(0, 1).flatten(1).any(1).nonzero().flatten().tolist() if value.dim() > 2 else []
                print(f"    {name}: {int(bad.sum())} non-finite values (decoder layers {layers}, lanes {lanes})")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            print(f"    input {key}: {int((~torch.isfinite(value)).sum())} non-finite values")
    print(f"    sequences: {list(zip(batch.get('seq', []), batch.get('start', [])))}")
    folder = os.path.join(out_dir, "nonfinite_batches")
    os.makedirs(folder, exist_ok=True)
    keep = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items() if k != "imgs"}
    torch.save({"step": step, "metrics": ts.metrics, "batch": keep}, os.path.join(folder, f"step_{step:07d}.pt"))


def _release_step(ts: "TrainStep", model: torch.nn.Module) -> None:
    """Drop a skipped step's graph and inputs, and restart streaming state (it may hold NaNs)."""
    try:
        ts.free_inputs()
    except AttributeError:
        pass   # already freed
    ts.loss = ts.cls_scores = ts.kpts_norm = ts.dn_meta = ts.dets_2d = None
    head = getattr(getattr(model, "_orig_mod", model), "head", None)
    if head is not None and hasattr(head, "reset_memory"):
        head.reset_memory()
        head.track_output = None


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


def train(cfg: Config, config_path: str | None = None, argv: list[str] | None = None):
    if cfg.train.max_steps < 0:
        raise ValueError("train.max_steps cannot be negative")
    device = resolve_device(cfg.train.device)
    if device.type == "cuda" and cfg.train.cuda_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(cfg.train.cuda_memory_fraction)
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
    criterion = SetCriterion(cfg.model, cfg.loss)

    if cfg.train.compile_model:
        if hasattr(torch, "compile"):
            torch._dynamo.config.capture_scalar_outputs = True
            print("[train] torch.compile(model)  (first steps will be slower during kernel compilation)")
            model = cast(UnifiedPerceptionModel, torch.compile(model))
            criterion = cast(torch.nn.Module, torch.compile(criterion))
        else:  # pragma: no cover
            print("[train] torch.compile not available (PyTorch < 2.0) — skipping")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        fused=device.type == "cuda",
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
    append_log = bool(cfg.train.resume and os.path.exists(cfg.train.resume) and os.path.exists(csv_path))
    csv_file = open(csv_path, "a" if append_log else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not append_log:
        csv_writer.writerow(["epoch", "split", "loss", "loss_cls", "loss_kpt",
                              "grad_norm", "gpu_alloc_gb_avg", "gpu_alloc_gb_peak",
                              "mpjpe_mm", "f1"])

    start_epoch = 0
    global_step = 0
    optimizer_steps = 0
    best_mpjpe = float("inf")
    best_extra: dict[str, float] = {}
    loss_ema: float | None = None          # global EMA for divergence detection
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        ckpt = torch.load(cfg.train.resume, map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(_strip_compile_prefix(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        # Mid-epoch checkpoints restart their (freshly shuffled) epoch.
        start_epoch = ckpt.get("epoch", 0) + (1 if ckpt.get("epoch_complete", True) else 0)
        global_step = ckpt.get("global_step", 0)
        optimizer_steps = ckpt.get("optimizer_steps", 0)
        best_mpjpe = ckpt.get("best_mpjpe", best_mpjpe)
        best_extra = dict(ckpt.get("best_extra", {}))
        loss_ema = ckpt.get("loss_ema")
        if cfg.train.max_steps and "optimizer_steps" not in ckpt:
            raise ValueError("A step-budgeted resume requires an optimizer_steps checkpoint")
        # Avoid replaying the identical sample order after every restart.
        set_seed(cfg.train.seed + optimizer_steps)
        print(f"Resumed from {cfg.train.resume} at epoch {start_epoch}, optimizer step {optimizer_steps}")

    # Immutable per-launch experiment record (engine/run_manifest.py); never fatal.
    try:
        from .run_manifest import write_launch_record
        record_path = write_launch_record(cfg, loaders, config_path=config_path, argv=argv,
                                          resumed_optimizer_steps=optimizer_steps)
        print(f"[train] launch record: {record_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[train] WARNING: launch record not written: {type(exc).__name__}: {exc}")

    def checkpoint(epoch: int, epoch_complete: bool = True) -> dict:
        return {"model": _unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "epoch_complete": epoch_complete, "global_step": global_step,
                "optimizer_steps": optimizer_steps, "best_mpjpe": best_mpjpe, "best_extra": best_extra,
                "loss_ema": loss_ema, "cfg": config_to_dict(cfg)}

    last_path = os.path.join(cfg.train.output_dir, "last.pt")
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
    deferred = bool(cfg.train.deferred_step) and ts.deferred_supported()
    pending = None   # (PendingStep, context) of the queued deferred step
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if cfg.train.max_steps and optimizer_steps >= cfg.train.max_steps:
            break
        model.train()
        running = {}
        epoch_peak_mem_gb = 0.0
        t_epoch_start = time.perf_counter()
        steps_this_epoch = 0
        skipped_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        t_ready = time.perf_counter()

        def finish(pending_step, ctx) -> bool:
            """Book-keep a deferred step from its host copy; returns True when a checkpoint is due."""
            nonlocal loss_ema, optimizer_steps, steps_this_epoch, skipped_steps, epoch_peak_mem_gb
            values = pending_step.result()
            step_no, loss_val, grad_norm = ctx["step"], values["loss"], values["grad_norm"]
            if values["skipped"]:
                bad = {k: v for k, v in values.items() if isinstance(v, float) and not math.isfinite(v)}
                print(f"\n[step {step_no}] NON-FINITE loss={loss_val} gnorm={grad_norm} — optimizer step "
                      f"skipped on device; streaming state reset.  Non-finite terms: {bad or 'none reported'}"
                      f"  sequences: {ctx['sequences']}")
                head = getattr(_unwrap(model), "head", None)
                if head is not None and hasattr(head, "reset_memory"):
                    head.reset_memory()
                    head.track_output = None
                for bank in (getattr(criterion, "reid", None), getattr(criterion, "reid_device", None)):
                    if bank is not None:
                        bank.reset()
                skipped_steps += 1
                return False
            if values.get("num_nonfinite_cost"):
                print(f"\n[step {step_no}] WARNING: {values['num_nonfinite_cost']} non-finite matching costs "
                      f"(finite loss); sequences: {ctx['sequences']}")
            optimizer_steps += 1
            steps_this_epoch += 1
            loss_ema = loss_val if loss_ema is None else _EMA_DECAY * loss_ema + (1 - _EMA_DECAY) * loss_val
            if step_no >= _EMA_WARMUP and loss_val > _SPIKE_FACTOR * loss_ema:
                print(f"\n[step {step_no}] LOSS SPIKE: {loss_val:.4f} vs EMA {loss_ema:.4f}"
                      f" (×{loss_val / loss_ema:.1f})")
            if grad_norm >= _warn_thresh:
                print(f"\n[step {step_no}] HIGH GRAD gnorm={grad_norm:.1f} (clip={cfg.train.grad_clip})")
            alloc_gb, rsvd_gb = _gpu_mem_gb()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            epoch_peak_mem_gb = max(epoch_peak_mem_gb, peak_gb)
            metrics = {k: v for k, v in values.items() if k != "skipped"}
            metrics.update(gpu_alloc_gb=alloc_gb, data_wait_s=ctx["data_wait"])
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + (float(v) if isinstance(v, (int, float)) else 0.0)
            if ctx["it"] % cfg.train.log_interval == 0:
                pbar.set_postfix(loss=f"{loss_val:.3f}", ema=f"{loss_ema:.3f}",
                                 kpt=f"{metrics.get('loss_kpt', float('nan')):.3f}", gnorm=f"{grad_norm:.2f}",
                                 mem=f"{alloc_gb:.1f}/{rsvd_gb:.1f}G", peak=f"{peak_gb:.1f}G",
                                 wait=f"{ctx['data_wait']:.2f}s")
                if writer:
                    for k, v in metrics.items():
                        writer.add_scalar(f"train/{k}", v, step_no)
                    writer.add_scalar("sys/gpu_alloc_gb", alloc_gb, step_no)
                    writer.add_scalar("sys/gpu_rsvd_gb", rsvd_gb, step_no)
                    writer.add_scalar("train/loss_ema", loss_ema, step_no)
            return bool(cfg.train.ckpt_every_steps and optimizer_steps % cfg.train.ckpt_every_steps == 0)

        def drain():
            """Finish the in-flight deferred step (before checkpoints, viz steps, epoch end)."""
            nonlocal pending
            if pending is not None:
                done, pending = pending, None
                if finish(*done):
                    _save_checkpoint(last_path, checkpoint(epoch, epoch_complete=False))

        for it, batch in enumerate(pbar):
            # Time blocked on the DataLoader since the previous step finished (input-bound if large).
            data_wait = time.perf_counter() - t_ready
            if cfg.train.max_steps and optimizer_steps + (pending is not None) >= cfg.train.max_steps:
                drain()
                if optimizer_steps >= cfg.train.max_steps:
                    break
            # Determine whether this step needs 2-D detections for visualisation
            # so we can fold it into the single training forward pass.
            viz_batch_interval = cfg.train.viz_batch_interval
            do_viz = viz_batch_interval > 0 and global_step % viz_batch_interval == 0

            if deferred and not do_viz:
                # No host synchronization in the step: its scalars are read after the next step
                # is queued (engine/step.py: TrainStep.step_deferred).
                ts.h2d(batch)
                ts.normalize()
                warmup_step = optimizer_steps + (pending is not None) if cfg.train.max_steps else global_step
                if warmup_step < cfg.train.warmup_iters:
                    for g in optimizer.param_groups:
                        g["lr"] = cfg.train.lr * (warmup_step + 1) / cfg.train.warmup_iters
                ts.forward()
                ctx = {"step": global_step, "it": it, "data_wait": data_wait,
                       "sequences": list(zip(batch.get("seq", []), batch.get("start", [])))}
                current = (ts.step_deferred(cfg.train.grad_clip), ctx)
                global_step += 1
                previous, pending = pending, current
                if previous is not None and finish(*previous):
                    drain()   # the queued step too, so the checkpoint's step count is exact
                    _save_checkpoint(last_path, checkpoint(epoch, epoch_complete=False))
                t_ready = time.perf_counter()
                continue
            drain()

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
                _nonfinite_report(ts, batch, cfg.train.output_dir, global_step)
                _release_step(ts, model)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                skipped_steps += 1
                t_ready = time.perf_counter()
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
            warmup_step = optimizer_steps if cfg.train.max_steps else global_step
            if warmup_step < cfg.train.warmup_iters:
                scale = (warmup_step + 1) / cfg.train.warmup_iters
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
                _release_step(ts, model)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                skipped_steps += 1
                t_ready = time.perf_counter()
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
            optimizer_steps += 1
            global_step += 1
            steps_this_epoch += 1
            if cfg.train.ckpt_every_steps and optimizer_steps % cfg.train.ckpt_every_steps == 0:
                _save_checkpoint(last_path, checkpoint(epoch, epoch_complete=False))

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
            ts.metrics["data_wait_s"] = data_wait

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
                    wait=f"{data_wait:.2f}s",
                )
                if writer:
                    for k, v in ts.metrics.items():
                        writer.add_scalar(f"train/{k}", v, global_step)
                    writer.add_scalar("sys/gpu_alloc_gb", alloc_gb, global_step)
                    writer.add_scalar("sys/gpu_rsvd_gb",  rsvd_gb,  global_step)
                    if loss_ema is not None:
                        writer.add_scalar("train/loss_ema", loss_ema, global_step)
            t_ready = time.perf_counter()
        drain()

        n = max(steps_this_epoch, 1)
        train_loss = running.get("loss", 0.0) / n
        avg_grad   = running.get("grad_norm", 0.0) / n
        avg_wait   = running.get("data_wait_s", 0.0) / n
        elapsed    = time.perf_counter() - t_epoch_start
        sps        = steps_this_epoch / max(elapsed, 1e-3)
        print(
            f"[epoch {epoch}] loss={train_loss:.4f}  ema={loss_ema:.4f}"
            f"  gnorm={avg_grad:.3f}  peak_mem={epoch_peak_mem_gb:.2f} GB"
            f"  steps={steps_this_epoch}  skipped={skipped_steps}"
            f"  {sps:.2f} step/s  data_wait={avg_wait:.2f}s/step  ({elapsed:.0f}s)"
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
            if (val_metrics["matched_persons"] > 0 and math.isfinite(val_metrics["mpjpe_mm"])
                    and val_metrics["mpjpe_mm"] < best_mpjpe):
                best_mpjpe = val_metrics["mpjpe_mm"]
                _save_checkpoint(os.path.join(cfg.train.output_dir, "best.pt"), checkpoint(epoch))
            for val_name in sorted(k for k in loaders if k.startswith("val_")):
                extra = run_eval(model, loaders[val_name], cfg, device, max_batches=max_vb)
                print(f"[epoch {epoch}] {val_name}: " + ", ".join(f"{k}={v:.3f}" for k, v in extra.items()))
                if writer:
                    for k, v in extra.items():
                        writer.add_scalar(f"{val_name}/{k}", v, global_step)
                csv_writer.writerow([epoch, val_name, "", "", "", "", "", "",
                                      extra["mpjpe_mm"], extra["f1"]])
                csv_file.flush()
                best_key = f"best_{val_name}"
                if (extra["matched_persons"] > 0 and math.isfinite(extra["mpjpe_mm"])
                        and extra["mpjpe_mm"] < best_extra.get(best_key, float("inf"))):
                    best_extra[best_key] = extra["mpjpe_mm"]
                    _save_checkpoint(os.path.join(cfg.train.output_dir, f"{best_key}.pt"), checkpoint(epoch))

        if epoch % cfg.train.ckpt_interval == 0 or (cfg.train.max_steps and optimizer_steps >= cfg.train.max_steps):
            _save_checkpoint(last_path, checkpoint(epoch))

    csv_file.close()
    if writer:
        writer.close()
    if cfg.train.max_steps and optimizer_steps < cfg.train.max_steps:
        raise RuntimeError(
            f"Only {optimizer_steps}/{cfg.train.max_steps} optimizer steps completed; "
            "increase train.max_epochs or investigate skipped steps."
        )
    print(f"Training complete. Best val MPJPE: {best_mpjpe:.2f} mm")
    return model