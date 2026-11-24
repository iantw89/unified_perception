"""Evaluation loop."""

from __future__ import annotations

import numpy as np
import torch

from unified_perception.models.detector import UnifiedPerceptionModel

from .._compat import get_tqdm
from .metrics import IdentityAccumulator, MetricsAccumulator
from ..constants import ROOT_JOINTS, ROOT_SLICE
from ..models.supervision import project

tqdm = get_tqdm()


def _accepts(model, name: str) -> bool:
    """Whether ``model.forward`` takes keyword ``name`` (evaluation also runs test stub models)."""
    import inspect
    forward = getattr(getattr(model, "_orig_mod", model), "forward")
    params = inspect.signature(forward).parameters
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


@torch.no_grad()
def run_eval(model: UnifiedPerceptionModel, loader, cfg, device, score_threshold: float = 0.3, max_batches=None):
    """Evaluate; the model's streaming state is saved and restored around the pass."""
    head = getattr(getattr(model, "_orig_mod", model), "head", None)
    if not hasattr(head, "get_stream_state"):
        return _run_eval(model, loader, cfg, device, score_threshold, max_batches)
    saved = head.get_stream_state()
    head.reset_memory()
    try:
        return _run_eval(model, loader, cfg, device, score_threshold, max_batches)
    finally:
        head.set_stream_state(saved)


def _run_eval(model, loader, cfg, device, score_threshold, max_batches):
    model.eval()
    head = getattr(getattr(model, "_orig_mod", model), "head", None)
    acc = MetricsAccumulator(
        pck_thresholds_mm=cfg.eval.pck_thresholds_mm,
        match_threshold_mm=cfg.eval.match_threshold_mm,
    )
    identity = IdentityAccumulator()
    has_identity = False
    buckets: dict[str, MetricsAccumulator] = {}
    two_d_stats = [0, 0]  # matched, targets: people seen without a 3-D target
    visibility_hits: dict[str, list[int]] = {}
    # Normalisation constants (float32, on device) for uint8 → float conversion.
    _norm_mean = torch.tensor(cfg.data.pixel_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3, 1, 1)
    _norm_std  = torch.tensor(cfg.data.pixel_std,  dtype=torch.float32, device=device).view(1, 1, 1, 3, 1, 1)
    for bi, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        imgs = batch["imgs"]
        if imgs.dtype == torch.uint8:
            imgs = imgs.to(device)   # the model normalizes per frame, as in training (no float window copy)
        else:
            imgs = imgs.to(device=device, dtype=torch.float32)
            imgs.sub_(_norm_mean).div_(_norm_std)
        world2img = batch["world2img"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        gt_kpts = batch["gt_keypoints"].to(device)  # (B,T,P,J,3) mm
        gt_valid = batch["gt_valid"].to(device)
        gt_frame_valid = batch.get(
            "gt_keypoint_frame_valid",
            torch.ones(gt_valid.shape[:2], dtype=torch.bool, device=device),
        ).to(device=device, dtype=torch.bool)
        gt_root = batch.get("gt_root")
        if gt_root is None:
            gt_root = gt_kpts[..., ROOT_SLICE, :].mean(dim=-2)
        else:
            gt_root = gt_root.to(device)
        gt_root_valid = batch.get("gt_root_valid", gt_valid).to(device=device, dtype=torch.bool)
        gt_keypoint_valid = batch.get(
            "gt_keypoint_valid",
            gt_valid.unsqueeze(-1).expand(*gt_valid.shape, gt_kpts.shape[-2]),
        ).to(device=device, dtype=torch.bool)
        gt_keypoint_valid = gt_keypoint_valid & gt_valid.unsqueeze(-1) & gt_frame_valid[:, :, None, None]
        # Approximate GT: only people with a 3-D target are scored geometrically.  People seen by a
        # camera without a 3-D target are matched by reprojection (counted separately, never false
        # positives); people with no geometry at all credit at most as many unmatched predictions.
        located_all = gt_valid.bool() & (gt_root_valid | gt_keypoint_valid.any(-1))
        kp2d = batch.get("gt_kp2d")
        kp2d_valid = None if kp2d is None else batch["gt_kp2d_valid"].to(device=device, dtype=torch.bool)
        kp2d = None if kp2d is None else kp2d.to(device)

        camera_valid = batch.get("camera_valid")
        if camera_valid is not None:
            camera_valid = camera_valid.to(device=device, dtype=torch.bool)
        stream_reset = batch.get("stream_reset")
        stream_pad = batch.get("stream_pad")
        extra = {} if stream_reset is None else {"stream_reset": stream_reset.to(device)}
        host_valid = batch.get("camera_valid")
        if isinstance(host_valid, torch.Tensor) and not host_valid.is_cuda and _accepts(model, "camera_valid_host"):
            # Host copies: view selection without a device sync; native-aspect valid sizes.
            extra["camera_valid_host"] = host_valid.bool()
            if isinstance(batch.get("image_hw"), torch.Tensor):
                extra["image_hw_host"] = batch["image_hw"]
        cls_scores, kpts_norm, _, _ = model(imgs, world2img, intrinsics, camera_valid=camera_valid, **extra)
        track_output = getattr(head, "track_output", None)
        track_ids = None if track_output is None else track_output["track_ids"]
        scores = cls_scores[-1, ..., 0].sigmoid()        # (B, T, Q)
        kpts_mm = model.denormalize_keypoints(kpts_norm[-1])  # (B, T, Q, J, 3)

        # Everything the per-frame bookkeeping reads, copied to the host once per batch: indexing
        # device tensors frame by frame cost one synchronization per .cpu() (~10k per batch).
        def host(x):
            return None if x is None else x.cpu()
        scores, kpts_mm, kpts_last, track_ids = host(scores), host(kpts_mm), host(kpts_norm[-1].float()), host(track_ids)
        located_all, gt_valid, gt_kpts, gt_root = host(located_all), host(gt_valid), host(gt_kpts), host(gt_root)
        gt_root_valid, gt_keypoint_valid, gt_frame_valid = host(gt_root_valid), host(gt_keypoint_valid), host(gt_frame_valid)
        kp2d, kp2d_valid, world2img_h = host(kp2d), host(kp2d_valid), host(world2img)
        n_views = imgs.shape[2]
        camera_valid = host(camera_valid)

        B, T = scores.shape[:2]
        for b in range(B):
            if stream_pad is not None and bool(stream_pad[b]):
                continue
            if stream_reset is None or track_ids is None or bool(stream_reset[b]):
                identity.reset_clip(b)
            for t in range(T):
                # Track-query models report people only through their track ids.
                keep = (track_ids[b, t] >= 0) if track_ids is not None else scores[b, t] > score_threshold
                valid = located_all[b, t]
                two_d = gt_valid[b, t].bool() & ~valid
                if kp2d_valid is not None:
                    two_d &= kp2d_valid[b, t].any(-1).any(0)
                else:
                    two_d &= False
                unlocated = int((gt_valid[b, t].bool() & ~valid & ~two_d).sum())
                two_d_assoc = []
                if two_d.any() and keep.any():
                    keep, two_d_assoc = _match_two_d(
                        keep, kpts_last[b, t], kp2d[b, t], kp2d_valid[b, t], world2img_h[b, t],
                        torch.nonzero(two_d).flatten(), model.position_range, cfg.eval.reproj_match_px)
                    two_d_stats[0] += len(two_d_assoc)
                two_d_stats[1] += int(two_d.sum())
                pred = kpts_mm[b, t][keep].cpu().numpy()
                gt = gt_kpts[b, t][valid].cpu().numpy()
                pred_root = kpts_mm[b, t][:, ROOT_SLICE, :].mean(dim=-2)[keep].cpu().numpy()
                roots = gt_root[b, t][valid].cpu().numpy()
                root_valid = gt_root_valid[b, t][valid].cpu().numpy()
                joint_valid = gt_keypoint_valid[b, t][valid].cpu().numpy()
                metric_args = {
                    "keypoints_supervised": bool(gt_frame_valid[b, t]),
                    "pred_root_mm": pred_root,
                    "gt_root_mm": roots,
                    "gt_root_valid": root_valid,
                    "gt_keypoint_valid": joint_valid,
                }
                associations = acc.update_frame(pred, gt, **metric_args)
                acc.credit_ambiguous(min(unlocated, len(pred) - len(associations)))
                camera_count = n_views if camera_valid is None else int(camera_valid[b, t].sum())
                groups = [f"people_{len(gt)}", f"cameras_{camera_count}"]
                if "capture_group" in batch:
                    groups.append("capture_" + batch["capture_group"][b])
                for group in groups:
                    if group not in buckets:
                        buckets[group] = MetricsAccumulator(
                            pck_thresholds_mm=cfg.eval.pck_thresholds_mm,
                            match_threshold_mm=cfg.eval.match_threshold_mm,
                        )
                    # Same credit as the headline metric: predictions on people present without any
                    # geometry are not false positives (otherwise unlabelled frames read as a flood).
                    bucket_assoc = buckets[group].update_frame(pred, gt, **metric_args)
                    buckets[group].credit_ambiguous(min(unlocated, len(pred) - len(bucket_assoc)))
                if "instance_ids" in batch:
                    has_identity = True
                    queries = torch.nonzero(keep, as_tuple=True)[0]
                    labels = (track_ids[b, t][queries] if track_ids is not None else queries).cpu().numpy()
                    instances = batch["instance_ids"][b, t][valid.cpu()].cpu().numpy()
                    if two_d_assoc:  # 2-D-only people keep their identity bookkeeping too
                        all_queries = torch.nonzero(keep | _released(two_d_assoc, keep), as_tuple=True)[0]
                        labels_all = (track_ids[b, t][all_queries] if track_ids is not None else all_queries)
                        order = {int(q): i for i, q in enumerate(all_queries.tolist())}
                        kept_rows = [order[int(q)] for q in torch.nonzero(keep, as_tuple=True)[0].tolist()]
                        assoc = [(kept_rows[r], c) for r, c in associations]
                        assoc += [(order[q], len(instances) + i) for i, (q, _) in enumerate(two_d_assoc)]
                        extra_ids = batch["instance_ids"][b, t][[g for _, g in two_d_assoc]].cpu().numpy()
                        identity.update(labels_all.cpu().numpy(), np.concatenate([instances, extra_ids]),
                                        assoc, lane=b)
                    else:
                        identity.update(labels, instances, associations, lane=b)
                if "visibility" in batch:
                    # Maximum visibility across cameras: fully hidden means
                    # hidden in every view. It never filters supervision.
                    fractions = batch["visibility"][b, t, :, valid.cpu()].amax(dim=0).cpu().numpy()
                    matched = {g for _, g in associations}
                    for g, fraction in enumerate(fractions):
                        group = "hidden" if fraction == 0 else ("partial" if fraction < 0.95 else "clear")
                        hits = visibility_hits.setdefault(group, [0, 0])
                        hits[0] += g in matched
                        hits[1] += 1
        # Re-identification takeovers relabel provisional ids retroactively (per lane segment).
        for lane, provisional, recovered in (track_output or {}).get("aliases", []):
            if stream_pad is None or not bool(stream_pad[lane]):
                identity.alias(lane, provisional, recovered)
    metrics = acc.compute()
    if two_d_stats[1]:
        metrics["two_d_only/targets"] = two_d_stats[1]
        metrics["two_d_only/recall"] = two_d_stats[0] / two_d_stats[1]
    if has_identity:
        metrics.update(identity.compute())
        metrics["id_track_mode"] = float(track_ids is not None)
    for group, bucket in buckets.items():
        metrics.update({f"{group}/{key}": value for key, value in bucket.compute().items()})
    for group, (hits, total) in visibility_hits.items():
        metrics[f"visibility_{group}/recall"] = hits / total
        metrics[f"visibility_{group}/targets"] = total
    return metrics


def _released(two_d_assoc, keep):
    mask = torch.zeros_like(keep)
    for q, _ in two_d_assoc:
        mask[q] = True
    return mask


def _match_two_d(keep, kpts_norm, kp2d, kp2d_valid, world2img, people, position_range, threshold_px):
    """Greedy reprojection matching of reported predictions to people without a 3-D target.

    Returns (keep without the matched predictions, [(query, person)]): matched predictions are
    neither false positives nor scored in 3-D.
    """
    queries = torch.nonzero(keep).flatten()
    pr = position_range.to(kpts_norm.device)
    uv, depth = project(kpts_norm[queries], world2img, pr[:3], pr[3:] - pr[:3])   # (N, Qk, J, ·)
    target, valid = kp2d[:, people], kp2d_valid[:, people]                        # (N, G, J, ·)
    err = (uv[:, :, None] - target[:, None, ..., :2]).norm(dim=-1)                 # (N, Qk, G, J)
    err = torch.where(depth[:, :, None] > 0, err, torch.full_like(err, 1e6))
    mask = valid[:, None].float()
    cost = ((err * mask).sum(dim=(0, 3)) / mask.sum(dim=(0, 3)).clamp(min=1)).cpu()   # (Qk, G)
    keep = keep.clone()
    pairs = []
    while cost.numel() and float(cost.min()) < threshold_px:
        r, c = divmod(int(torch.argmin(cost)), cost.shape[1])
        pairs.append((int(queries[r]), int(people[c])))
        keep[queries[r]] = False
        cost[r, :] = float("inf")
        cost[:, c] = float("inf")
    return keep, pairs
