"""Evaluation loop."""

from __future__ import annotations

import torch

from unified_perception.models.detector import UnifiedPerceptionModel

from .._compat import get_tqdm
from .metrics import MetricsAccumulator

tqdm = get_tqdm()


@torch.no_grad()
def run_eval(model: UnifiedPerceptionModel, loader, cfg, device, score_threshold: float = 0.3, max_batches=None):
    model.eval()
    acc = MetricsAccumulator(
        pck_thresholds_mm=cfg.eval.pck_thresholds_mm,
        match_threshold_mm=cfg.eval.match_threshold_mm,
    )
    # Normalisation constants (float32, on device) for uint8 → float conversion.
    _norm_mean = torch.tensor(cfg.data.pixel_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3, 1, 1)
    _norm_std  = torch.tensor(cfg.data.pixel_std,  dtype=torch.float32, device=device).view(1, 1, 1, 3, 1, 1)
    for bi, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        imgs = batch["imgs"].to(device=device, dtype=torch.float32)
        imgs.sub_(_norm_mean).div_(_norm_std)
        world2img = batch["world2img"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        gt_kpts = batch["gt_keypoints"].to(device)  # (B,T,P,J,3) mm
        gt_valid = batch["gt_valid"].to(device)

        cls_scores, kpts_norm, _, _ = model(imgs, world2img, intrinsics)
        scores = cls_scores[-1, ..., 0].sigmoid()        # (B, T, Q)
        kpts_mm = model.denormalize_keypoints(kpts_norm[-1])  # (B, T, Q, J, 3)

        B, T = scores.shape[:2]
        for b in range(B):
            for t in range(T):
                keep = scores[b, t] > score_threshold
                pred = kpts_mm[b, t][keep].cpu().numpy()
                valid = gt_valid[b, t]
                gt = gt_kpts[b, t][valid].cpu().numpy()
                acc.update_frame(pred, gt)
    return acc.compute()
