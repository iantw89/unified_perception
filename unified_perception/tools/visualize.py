#!/usr/bin/env python
"""Visualise val (or train) batches using the same dataloader, forward pass,
and rendering pipeline as the training loop.

Produces one PNG montage per batch sample in the same format as the
in-training visualisation triggered by ``viz_batch_interval``.  The output
directory and filename layout match ``save_train_viz`` exactly so the results
can be compared directly with training-time snapshots.

Usage examples
--------------
# GT + predictions (load a checkpoint):
python -m unified_perception.tools.visualize \\
    --config configs/pipe_cleaner.yaml \\
    --checkpoint runs/pipe_cleaner/best.pt \\
    --batches 4 --out runs/vis

# GT-only (no checkpoint — random weights, useful for data sanity-checks):
python -m unified_perception.tools.visualize \\
    --config configs/pipe_cleaner.yaml \\
    --batches 2 --split train
"""

from __future__ import annotations

import argparse
import os

import torch

from unified_perception.config import load_config
from unified_perception.engine.train import build_loaders, resolve_device
from unified_perception.engine.train_viz import save_train_viz
from unified_perception.models import UnifiedPerceptionModel


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualise model predictions with the training dataloader and renderer."
    )
    ap.add_argument("--config",     default=None,  help="path to config YAML")
    ap.add_argument("--checkpoint", default=None,  help="model .pt to load (random weights if omitted)")
    ap.add_argument("--split",      default="val", choices=["train", "val"])
    ap.add_argument("--batches",    type=int, default=4, help="number of batches to render")
    ap.add_argument("--out",        default="runs/vis")
    ap.add_argument("overrides",    nargs="*")
    args = ap.parse_args()

    cfg    = load_config(args.config, overrides=args.overrides)
    device = resolve_device(cfg.train.device)
    os.makedirs(args.out, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = UnifiedPerceptionModel(cfg).to(device)
    epoch = 0
    if args.checkpoint:
        ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        # Strip torch.compile prefix that may be present in saved state-dicts.
        state = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
        model.load_state_dict(state)
        epoch = ckpt.get("epoch", 0)
        print(f"Loaded checkpoint  epoch={epoch}  {args.checkpoint}")
    model.eval()

    # ── Norm constants — identical to train.py ────────────────────────────
    norm_mean = torch.tensor(
        cfg.data.pixel_mean, dtype=torch.float32, device=device
    ).view(1, 1, 1, 3, 1, 1)
    norm_std = torch.tensor(
        cfg.data.pixel_std, dtype=torch.float32, device=device
    ).view(1, 1, 1, 3, 1, 1)

    # ── Data — same loader as training ────────────────────────────────────
    loaders = build_loaders(cfg)
    loader  = loaders[args.split]
    # Note: CUDAPrefetcher is intentionally not used here.  The prefetcher
    # converts imgs to float32 on GPU before we can snapshot them for
    # save_train_viz, and our in-place sub_/div_ would then clobber the
    # batch tensor.  Without the prefetcher, .to(device, dtype=float32)
    # always allocates a new tensor, leaving batch["imgs"] as uint8 on CPU.

    # ── Inference + render ────────────────────────────────────────────────
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.batches:
            break

        # h2d + normalize — same as TrainStep.h2d() + TrainStep.normalize()
        imgs       = batch["imgs"].to(device=device, dtype=torch.float32)
        world2img  = batch["world2img"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        gt_valid   = batch["gt_valid"].to(device)
        gt_norm    = model.normalize_keypoints(batch["gt_keypoints"].to(device))

        imgs.sub_(norm_mean).div_(norm_std)

        # Forward — same autocast settings as TrainStep.forward()
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                cls_scores, kpts_norm, _, dets_2d = model(
                    imgs, world2img, intrinsics, gt_norm, gt_valid,
                    return_2d=True,
                    camera_valid=None if "camera_valid" not in batch else batch["camera_valid"].to(device),
                )

        B = imgs.shape[0]
        T = imgs.shape[1]
        N = imgs.shape[2]

        kpts_cpu = kpts_norm.detach().float().cpu()   # (L, B, T, Q, J, 3)
        cls_cpu  = cls_scores.detach().float().cpu()  # (L, B, T, Q, 1)

        # Build viz_list — identical layout to the do_viz block in train.py.
        # One (batch_cpu_slice, kpts, cls, yolo_dets) tuple per batch element.
        viz_list = []
        for b in range(B):
            vb_cpu = {
                k: v[b:b+1].cpu() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # dets_2d flat index: t * B * N + b * N + c  (matches detector.forward order)
            yolo_b = (
                [dets_2d[t * B * N + b * N + c] for t in range(T) for c in range(N)]
                if dets_2d is not None else None
            )
            viz_list.append((vb_cpu, kpts_cpu[:, b:b+1], cls_cpu[:, b:b+1], yolo_b))

        batch_out = os.path.join(args.out, f"batch_{batch_idx:04d}")
        save_train_viz(
            batch_out, viz_list, model,
            epoch=epoch,
            extra_text=f"{args.split}  batch={batch_idx}",
        )
        print(f"[{batch_idx + 1}/{args.batches}]  {B} samples  →  {batch_out}/")

    print(f"\nDone. Output in {args.out}")


if __name__ == "__main__":
    main()
