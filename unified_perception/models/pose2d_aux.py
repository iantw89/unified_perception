"""Auxiliary 2-D pose supervision on the backbone's stride-8 features (experimental).

DINOv3 was not trained on pose; a light heatmap head on the finest FPN level, supervised with
the 2-D projections of the 3-D ground truth (and the gym's 2-D-only targets), gives the
backbone/FPN a direct keypoint signal.  Training only; evaluation skips it.

Targets per image (one camera of one frame):
  * every labelled 3-D joint projected through the camera (mocap / MOYO: exact and complete;
    gym: triangulated, down-weighted like the 3-D loss by its per-joint sigma);
  * gym people seen by a camera without a triangulation (``gt_kp2d``), at their 2-D positions.
Only frames with labels (``frame_valid``) are supervised, over the camera's valid region; the
whole image is supervised (CenterNet penalty-reduced focal loss), so detections of unlabelled
people count against the model -- acceptable because the gym labels are near-complete.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pose2DHead(nn.Module):
    def __init__(self, in_channels: int, num_joints: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GroupNorm(16, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(16, hidden), nn.GELU(),
            nn.Conv2d(hidden, num_joints, 1))
        nn.init.constant_(self.net[-1].bias, -4.6)          # sigmoid ~ 0.01: sparse heatmaps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _project(points: torch.Tensor, world2img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """points (..., P, J, 3) mm, world2img (..., 4, 4) -> pixels (..., P, J, 2), in front (..., P, J)."""
    homo = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    uvw = torch.einsum("...ij,...pkj->...pki", world2img[..., :3, :], homo)
    depth = uvw[..., 2]
    return uvw[..., :2] / depth.clamp_min(1e-3)[..., None], depth > 1.0


def heatmap_targets(points_px: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor,
                    grid_hw: tuple[int, int], stride: int, sigma_cells: float):
    """Gaussian heatmaps (M, J, h, w) peaking at 1 on each joint's cell, max over people, and the
    per-(image, joint) weight (max over that joint's instances).

    points_px (M, P, J, 2) pixels; valid (M, P, J); weight (M, P, J) in [0, 1].
    """
    M, P, J, _ = points_px.shape
    h, w = grid_hw
    cell = torch.floor(points_px / stride)                                      # integer centres
    ys = torch.arange(h, device=points_px.device, dtype=points_px.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=points_px.device, dtype=points_px.dtype).view(1, 1, 1, w)
    target = points_px.new_zeros(M, J, h, w)
    for p in range(P):
        cx, cy = cell[:, p, :, 0, None, None], cell[:, p, :, 1, None, None]          # (M, J, 1, 1)
        g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma_cells ** 2))
        target = torch.maximum(target, g * valid[:, p, :, None, None])
    plane_weight = (weight * valid).amax(dim=1)                                  # (M, J)
    return target, plane_weight


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor, plane_weight: torch.Tensor,
                       region: torch.Tensor, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss, normalized by the weighted number of peaks.

    logits / target (M, J, h, w); plane_weight (M, J) (0 = unsupervised plane); region (M, 1, h, w)
    valid-image cells.  Returns (loss, number of positive peaks).
    """
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = (target >= 1 - 1e-4).float()
    pos_term = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_term = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    mask = plane_weight[:, :, None, None] * region
    num_pos = (pos * mask).sum()
    loss = ((pos_term + neg_term) * mask).sum() / num_pos.clamp_min(1.0)
    return loss, num_pos.detach()


def sigma_weight(sigma_mm: torch.Tensor | None, like: torch.Tensor, scale_mm: float = 30.0) -> torch.Tensor:
    """3-D loss convention: weight 30 / (30 + sigma) (exact GT: sigma 0 -> 1)."""
    if sigma_mm is None:
        return torch.ones_like(like, dtype=torch.float32)
    return scale_mm / (scale_mm + sigma_mm.float().clamp_min(0))


def aux2d_loss(head: Pose2DHead, feat: torch.Tensor, stride: int, sigma_cells: float, *,
               gt_mm: torch.Tensor, keypoint_valid: torch.Tensor, frame_valid: torch.Tensor,
               world2img: torch.Tensor, camera_valid: torch.Tensor | None,
               image_hw: torch.Tensor | None, image_size: tuple[int, int],
               sigma_mm: torch.Tensor | None = None, kp2d: torch.Tensor | None = None,
               kp2d_valid: torch.Tensor | None = None):
    """Loss of ``head`` on ``feat`` (B*T*N, C, h, w), the stride-``stride`` features of every view
    of the window in (B, T, N) order.

    gt_mm (B, T, P, J, 3); keypoint_valid (B, T, P, J); frame_valid (B, T); world2img (B, T, N, 4, 4);
    camera_valid (B, T, N); image_hw (B, N, 2) valid sizes (None: full ``image_size``);
    sigma_mm (B, T, P, J); kp2d (B, T, N, P, J, >=2) pixels with kp2d_valid (B, T, N, P, J).
    """
    B, T, N = world2img.shape[:3]
    h, w = feat.shape[-2:]
    logits = head(feat).view(B, T, N, -1, h, w)
    J = logits.shape[3]
    uv, front = _project(gt_mm[:, :, None].expand(-1, -1, N, -1, -1, -1), world2img)   # (B,T,N,P,J,2)
    H, W = image_size
    if image_hw is None:
        limit = torch.tensor([W, H], device=uv.device, dtype=uv.dtype).expand(B, N, 2)
    else:
        limit = image_hw.to(device=uv.device, dtype=uv.dtype).flip(-1)                    # (B, N, [w, h])
    lim = limit[:, None, :, None, None, :]
    inside = (uv >= 0).all(-1) & (uv < lim).all(-1) & front
    labelled = frame_valid[:, :, None, None, None].bool()
    valid = keypoint_valid[:, :, None].bool() & inside & labelled
    weight = sigma_weight(sigma_mm, keypoint_valid)[:, :, None].expand_as(valid)
    if kp2d is not None and kp2d_valid is not None:
        uv2 = kp2d[..., :2].to(uv.dtype)
        inside2 = (uv2 >= 0).all(-1) & (uv2 < lim).all(-1)
        valid2 = kp2d_valid.bool() & inside2 & labelled
        uv = torch.cat([uv, uv2], 3)
        valid = torch.cat([valid, valid2], 3)
        weight = torch.cat([weight, torch.ones_like(valid2, dtype=weight.dtype)], 3)
    if camera_valid is not None:
        valid = valid & camera_valid[..., None, None].bool()
    M = B * T * N
    target, plane_w = heatmap_targets(uv.reshape(M, *uv.shape[3:]), valid.reshape(M, *valid.shape[3:]).float(),
                                      weight.reshape(M, *weight.shape[3:]), (h, w), stride, sigma_cells)
    # Supervise every joint plane of a labelled, present view: absent joints are negatives too.
    view_on = labelled[:, :, :, 0, 0].expand(B, T, N)
    if camera_valid is not None:
        view_on = view_on & camera_valid.bool()
    plane_w = torch.where(plane_w > 0, plane_w, torch.ones_like(plane_w)) * view_on.reshape(M, 1).float()
    rows = torch.arange(h, device=feat.device).view(1, h, 1) * stride
    cols = torch.arange(w, device=feat.device).view(1, 1, w) * stride
    lim_flat = limit[:, None].expand(B, T, N, 2).reshape(M, 2)
    region = ((rows < lim_flat[:, 1, None, None]) & (cols < lim_flat[:, 0, None, None]))[:, None].float()
    return focal_heatmap_loss(logits.reshape(M, J, h, w), target, plane_w, region)
