"""Supervision from approximate and incomplete ground truth.

Three kinds of positive (``gt_valid``) person per frame:

* **located**: a 3-D target exists (joints and/or root).  Targets may be noisy
  (triangulated): each carries a sigma (mm).  The keypoint/root loss ignores
  errors within ``gt_noise_tolerance * sigma`` and down-weights the rest by
  ``ref / (ref + sigma)``; exact GT (sigma = 0) reduces to plain L1.
* **2-D only**: seen by some camera but not triangulated (e.g. a dead zone where
  only one camera sees the person).  Matched to queries by reprojection cost and
  supervised by a robust (log1p) 2-D reprojection loss -- never treated as absent.
* **match-only**: no 3-D target, but an interpolated root between two label frames
  (``match_root``).  Matched to queries by root cost and trained as present (existence /
  recall), with no position loss: the interpolation is a cue for *which* query is the
  person, never a pose target (it is biased for fast motion).
* **unlocated**: known present, no geometry at all.  Never matched to detection
  queries (no cost to match on); a track slot holding the identity is still a
  positive, without position loss.

``instance_exists`` people that are not ``gt_valid`` remain ignored (models/tracking.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..constants import ROOT_JOINTS, ROOT_SLICE
from .matcher import hungarian_cost, solve_costs, to_device


@dataclass
class LossSettings:
    noise_tolerance: float = 1.0
    sigma_ref_mm: float = 30.0
    reproj_weight: float = 1.0
    reproj_tolerance_px: float = 4.0
    reproj_scale_px: float = 64.0
    reproj_triangulated_weight: float = 0.0

    @classmethod
    def from_config(cls, cfg_loss) -> "LossSettings":
        if cfg_loss is None:
            return cls()
        return cls(cfg_loss.gt_noise_tolerance, cfg_loss.gt_sigma_ref_mm, cfg_loss.reproj_weight,
                   cfg_loss.reproj_tolerance_px, cfg_loss.reproj_scale_px, cfg_loss.reproj_triangulated_weight)


def project(points_norm: torch.Tensor, world2img: torch.Tensor, range_min: torch.Tensor,
            range_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized model-space points (..., 3) -> pixels (N, ..., 2) and depth (N, ...)."""
    mm = points_norm.float() * range_size + range_min
    homog = torch.cat([mm, torch.ones_like(mm[..., :1])], dim=-1)
    flat = homog.reshape(-1, 4)
    cam = torch.einsum("nij,mj->nmi", world2img.float(), flat)
    depth = cam[..., 2]
    uv = cam[..., :2] / depth.clamp(min=1e-3)[..., None]
    shape = points_norm.shape[:-1]
    return uv.reshape(world2img.shape[0], *shape, 2), depth.reshape(world2img.shape[0], *shape)


class FrameTargets:
    """Targets of one (lane, frame); indices into the padded person axis P."""

    def __init__(self, sup: "Supervision", b: int, t: int):
        self.gts = sup.gt[b, t]
        self.valid = sup.valid[b, t]
        self.joints_valid = sup.keypoint_valid[b, t]
        self.roots = sup.root[b, t]
        self.roots_valid = sup.root_valid[b, t]
        self.sigma = None if sup.sigma is None else sup.sigma[b, t]              # (P, J) mm
        self.root_sigma = None if sup.root_sigma is None else sup.root_sigma[b, t]
        self.kp2d = None if sup.kp2d is None else sup.kp2d[b, t]                 # (N, P, J, 3)
        self.kp2d_valid = None if sup.kp2d is None else sup.kp2d_valid[b, t]     # (N, P, J)
        self.world2img = None if sup.world2img is None else sup.world2img[b, t]  # (N, 4, 4)
        self.range_min, self.range_size = sup.range_min, sup.range_size
        self.located = sup.located[b, t]
        self.has_2d = sup.has_2d[b, t]
        self.match_root = None if sup.match_root is None else sup.match_root[b, t]
        self.match_only = sup.match_only[b, t]
        self.matchable = sup.matchable[b, t]
        # Host copies (one transfer per batch): branch decisions without device syncs.
        cpu = sup.cpu()
        self.cpu_valid, self.cpu_matchable = cpu["valid"][b, t], cpu["matchable"][b, t]
        self.cpu_match_only, self.cpu_located = cpu["match_only"][b, t], cpu["located"][b, t]
        self.supervised = bool(cpu["frame_valid"][b, t])

    def reprojection_cost(self, preds: torch.Tensor, people: torch.Tensor, scale_px: float) -> torch.Tensor:
        """(Q, len(people)) mean reprojection error / scale over each person's 2-D joints."""
        uv, depth = project(preds, self.world2img, self.range_min, self.range_size)  # (N, Q, J, ·)
        target = self.kp2d[:, people]                                                 # (N, G, J, 3)
        valid = self.kp2d_valid[:, people]
        err = (uv[:, :, None] - target[None, ..., :2].transpose(0, 1)).norm(dim=-1)  # (N, Q, G, J)
        err = torch.where(depth[:, :, None] > 0, err, torch.full_like(err, 10 * scale_px))
        mask = valid[:, None].float()
        return (err * mask).sum(dim=(0, 3)) / mask.sum(dim=(0, 3)).clamp(min=1) / scale_px


class Supervision:
    """Batch targets (normalized model space) with optional approximate-GT extras."""

    def __init__(self, targets: dict, position_range: torch.Tensor | None = None):
        self.gt = targets["gt"]
        self.valid = targets["valid"].bool()
        self.keypoint_valid = targets["keypoint_valid"].bool() & self.valid[..., None]
        self.frame_valid = targets["frame_valid"].bool()
        self.root = targets["root"]
        self.root_valid = targets["root_valid"].bool() & self.valid
        self.sigma = targets.get("sigma_mm")
        self.root_sigma = targets.get("root_sigma_mm")
        self.kp2d = targets.get("kp2d")
        self.kp2d_valid = None if self.kp2d is None else targets["kp2d_valid"].bool()
        self.world2img = targets.get("world2img")
        self.match_root = targets.get("match_root")
        self.match_root_valid = None if self.match_root is None else targets["match_root_valid"].bool()
        # Person categories for the whole batch (B, T, P), computed once.
        self.located = self.keypoint_valid.any(-1) | self.root_valid
        self.has_2d = (torch.zeros_like(self.valid) if self.kp2d_valid is None
                       else self.kp2d_valid.any(-1).any(-2))
        if self.match_root is None:
            self.match_only = torch.zeros_like(self.valid)
        else:
            self.match_only = self.valid & self.match_root_valid & ~self.located & ~self.has_2d
        self.matchable = self.valid & (self.located | self.has_2d | self.match_only)
        self._cpu = None
        pr = position_range if position_range is not None else targets.get("position_range")
        if pr is None:
            self.range_min = self.range_size = None
            if self.kp2d is not None:
                raise ValueError("2-D supervision needs the position range")
        else:
            if not (isinstance(pr, torch.Tensor) and pr.device == self.gt.device):
                pr = torch.as_tensor(pr, dtype=torch.float32).to(self.gt.device, non_blocking=True)
            pr = pr.float()
            self.range_min, self.range_size = pr[:3], pr[3:] - pr[:3]

    def cpu(self) -> dict:
        """Host copies of the per-person masks (B, T, P) and ``frame_valid`` (B, T)."""
        if self._cpu is None:
            masks = torch.stack([self.valid, self.matchable, self.match_only, self.located,
                                 self.frame_valid[..., None].expand_as(self.valid)]).cpu()
            self._cpu = {"valid": masks[0], "matchable": masks[1], "match_only": masks[2],
                         "located": masks[3], "frame_valid": masks[4][..., 0]}
        return self._cpu

    def frame(self, b: int, t: int) -> FrameTargets:
        return FrameTargets(self, b, t)


def match_cost(frame: FrameTargets, logits: torch.Tensor, preds: torch.Tensor, people: list[int], costs: dict,
               scale_px: float = 64.0):
    """(cost (Q, G) or None, matchable people): the matching cost of queries against ``people``
    (P indices); 2-D-only people use reprojection cost, unlocated people are skipped.  Branches
    are decided on host masks, so nothing here waits for the device."""
    people = [g for g in people if bool(frame.cpu_matchable[g])]
    if not people:
        return None, []
    index = to_device(people, logits.device)
    extra = extra_mask = None
    roots, roots_valid = frame.roots[index], frame.roots_valid[index]
    match_only = frame.match_only[index]
    cpu_match_only = frame.cpu_match_only[people]
    if bool(cpu_match_only.any()):   # interpolated root: matching cost only (no joints -> root cost)
        roots = torch.where(match_only[:, None], frame.match_root[index].to(roots.dtype), roots)
        roots_valid = roots_valid | match_only
    if bool((~frame.cpu_located[people] & ~cpu_match_only).any()):
        extra = frame.reprojection_cost(preds, index, scale_px)
        extra_mask = ~frame.located[index] & ~match_only
    cost = hungarian_cost(
        logits, preds, frame.gts[index], costs["class"], costs["kpt"], costs["alpha"], costs["gamma"],
        gt_root=roots, gt_root_valid=roots_valid,
        gt_keypoint_valid=frame.joints_valid[index], cost_root=costs["root"],
        cost_root_labeled=costs["root_labeled"], root_only=not frame.supervised,
        extra_cost=extra, extra_mask=extra_mask,
    )
    return cost, people


def match_many(items: list, costs: dict, scale_px: float = 64.0) -> list:
    """Several independent matches (``items`` = [(frame, logits, preds, people)]) with one
    device-to-host transfer.  Returns [(query indices, person indices)]."""
    built = [match_cost(frame, logits, preds, people, costs, scale_px) for frame, logits, preds, people in items]
    solved = solve_costs([cost for cost, _ in built])
    return [(rows, [people[c] for c in cols]) for (_, people), (rows, cols) in zip(built, solved)]


def match(frame: FrameTargets, logits: torch.Tensor, preds: torch.Tensor, people: list[int], costs: dict,
          scale_px: float = 64.0) -> tuple[list[int], list[int]]:
    """Hungarian match of queries against ``people`` (P indices); 2-D-only people use reprojection
    cost, unlocated people are skipped.  Returns (query indices, person indices)."""
    return match_many([(frame, logits, preds, people)], costs, scale_px)[0]


def noisy_l1(error: torch.Tensor, sigma_norm: torch.Tensor | None, sigma_mm: torch.Tensor | None,
             settings: LossSettings) -> torch.Tensor:
    """Element-wise |error| beyond the noise tolerance, down-weighted by target uncertainty."""
    if sigma_norm is None:
        return error
    excess = (error - settings.noise_tolerance * sigma_norm).clamp(min=0)
    return excess * (settings.sigma_ref_mm / (settings.sigma_ref_mm + sigma_mm))[..., None]


def pair_losses(frame: FrameTargets, preds: torch.Tensor, queries: list[int], people: list[int],
                settings: LossSettings) -> dict:
    """Keypoint / root / reprojection loss sums and counts for matched (query, person) pairs.

    ``preds`` is (Q, J, 3) or (L, Q, J, 3) (all decoder layers at once, same pairs); sums run
    over every leading entry, counts are per layer.
    """
    zero = preds.new_zeros(())
    out = {"kpt": zero, "n_kpt": 0, "root": zero, "n_root": 0, "reproj": zero, "n_reproj": 0}
    if not queries:
        return out
    qi = torch.as_tensor(queries, device=preds.device)
    gi = torch.as_tensor(people, device=preds.device)
    p, g = preds.index_select(-3, qi), frame.gts[gi]
    size = frame.range_size
    sigma = None if frame.sigma is None else frame.sigma[gi]
    sigma_norm = None if sigma is None else sigma[..., None] / size
    joints = frame.joints_valid[gi]
    if joints.any():
        err = noisy_l1((p - g).abs(), sigma_norm, sigma, settings)
        out["kpt"] = (err * joints[..., None]).sum()
        out["n_kpt"] = int(joints.sum())
    roots_ok = frame.roots_valid[gi]
    if roots_ok.any():
        pred_root = p[..., ROOT_SLICE, :].mean(dim=-2)
        rs = None if frame.root_sigma is None else frame.root_sigma[gi]
        err = noisy_l1((pred_root - frame.roots[gi]).abs(), None if rs is None else rs[..., None] / size, rs,
                       settings)
        out["root"] = (err * roots_ok[..., None]).sum()
        out["n_root"] = int(roots_ok.sum())
    if frame.kp2d is not None and settings.reproj_weight > 0:
        weight = torch.where(frame.located[gi], settings.reproj_triangulated_weight, 1.0).float()
        valid = frame.kp2d_valid[:, gi] & (weight > 0)[None, :, None]                 # (N, M, J)
        if valid.any():
            uv, depth = project(p, frame.world2img, frame.range_min, size)            # (N, [L,] M, J, ·)
            target = frame.kp2d[:, gi]
            if p.dim() == 4:
                target = target[:, None]
            residual = (uv - target[..., :2]).norm(dim=-1)
            # Robust: log1p grows slowly for gross errors (wrong matches, bad detections), and
            # joints predicted behind the camera are left out (their projection is meaningless).
            excess = torch.log1p((residual - settings.reproj_tolerance_px).clamp(min=0) / settings.reproj_scale_px)
            mask = (valid * weight[None, :, None]).float()
            if p.dim() == 4:
                mask = mask[:, None]
            mask = mask * (depth > 0)
            out["reproj"] = (excess * target[..., 2].clamp(0, 1) * mask).sum()
            out["n_reproj"] = int(valid.sum())
    return out


def batch_pair_losses(sup: Supervision, preds: torch.Tensor, b: torch.Tensor, t: torch.Tensor,
                      q: torch.Tensor, g: torch.Tensor, settings: LossSettings,
                      pair_mask: torch.Tensor | None = None) -> dict:
    """:func:`pair_losses` for every matched pair of a batch at once.

    ``preds`` (L, B, T, Q, J, 3); ``b, t, q, g`` (M,) device index tensors of the pairs (lane,
    frame, query, person).  Sums run over all layers; counts are per layer and stay on the
    device (no host syncs).  ``pair_mask`` (M,) optionally disables pairs (dense evaluation of
    every query with a mask instead of a variable-length list).  Invalid targets may hold NaN (data/golden.py leaves unlabeled
    frames NaN): they are replaced before any arithmetic so no NaN reaches the gradients.
    """
    zero = preds.new_zeros(())
    out = {"kpt": zero, "n_kpt": zero, "root": zero, "n_root": zero, "reproj": zero, "n_reproj": zero}
    if b.shape[0] == 0:
        return out
    p = preds[:, b, t, q]                                                     # (L, M, J, 3)
    size = sup.range_size
    joints = sup.keypoint_valid[b, t, g]                                      # (M, J)
    if pair_mask is not None:
        joints = joints & pair_mask[:, None]
    err = (p - torch.where(joints[..., None], sup.gt[b, t, g], 0.0)).abs()
    if sup.sigma is not None:
        sigma = torch.where(joints, sup.sigma[b, t, g], 0.0)
        err = noisy_l1(err, sigma[..., None] / size, sigma, settings)
    out["kpt"] = torch.where(joints[..., None], err, 0.0).sum()
    out["n_kpt"] = joints.sum()
    roots_ok = sup.root_valid[b, t, g]                                        # (M,)
    if pair_mask is not None:
        roots_ok = roots_ok & pair_mask
    pred_root = p[..., ROOT_SLICE, :].mean(dim=-2)                     # (L, M, 3)
    err = (pred_root - torch.where(roots_ok[..., None], sup.root[b, t, g], 0.0)).abs()
    if sup.root_sigma is not None:
        rs = torch.where(roots_ok, sup.root_sigma[b, t, g], 0.0)
        err = noisy_l1(err, rs[..., None] / size, rs, settings)
    out["root"] = torch.where(roots_ok[..., None], err, 0.0).sum()
    out["n_root"] = roots_ok.sum()
    if sup.kp2d is not None and settings.reproj_weight > 0:
        weight = torch.where(sup.located[b, t, g], settings.reproj_triangulated_weight, 1.0).float()  # (M,)
        if pair_mask is not None:
            weight = weight * pair_mask
        valid = sup.kp2d_valid[b, t, :, g] & (weight > 0)[:, None, None]                    # (M, N, J)
        target = torch.where(valid[..., None], sup.kp2d[b, t, :, g], 0.0)                    # (M, N, J, 3)
        mm = p.float() * size + sup.range_min
        homog = torch.cat([mm, torch.ones_like(mm[..., :1])], dim=-1)                       # (L, M, J, 4)
        cam = torch.einsum("mnij,lmkj->lmnki", sup.world2img[b, t].float(), homog)          # (L, M, N, J, 4)
        depth = cam[..., 2]
        uv = cam[..., :2] / depth.clamp(min=1e-3)[..., None]
        residual = (uv - target[..., :2]).norm(dim=-1)                                       # (L, M, N, J)
        # Robust: log1p grows slowly for gross errors (wrong matches, bad detections), and
        # joints predicted behind the camera are left out (their projection is meaningless).
        excess = torch.log1p((residual - settings.reproj_tolerance_px).clamp(min=0) / settings.reproj_scale_px)
        scale = weight[:, None, None] * target[..., 2].clamp(0, 1)
        out["reproj"] = torch.where(valid & (depth > 0), excess * scale, 0.0).sum()
        out["n_reproj"] = valid.sum()
    return out
