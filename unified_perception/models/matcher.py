"""Hungarian matcher for per-frame person assignment."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from ..constants import ROOT_JOINTS, ROOT_SLICE


@torch.no_grad()
def hungarian_match(*args, **kwargs):
    """Return query/target indices, using only annotated joints per target (see
    :func:`hungarian_cost` for the arguments)."""
    cost = hungarian_cost(*args, **kwargs)
    device = args[0].device
    if cost is None:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty
    row, col = linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(row, dtype=torch.long, device=device),
            torch.as_tensor(col, dtype=torch.long, device=device))


@torch.no_grad()
def hungarian_cost(
    pred_logits: torch.Tensor,
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    cost_class: float = 2.0,
    cost_kpt: float = 5.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    *,
    gt_root: torch.Tensor | None = None,
    gt_root_valid: torch.Tensor | None = None,
    gt_keypoint_valid: torch.Tensor | None = None,
    cost_root: float = 5.0,
    cost_root_labeled: float = 0.0,
    root_only: bool = False,
    extra_cost: torch.Tensor | None = None,
    extra_mask: torch.Tensor | None = None,
):
    """(Q, G) matching cost (None without targets), using only annotated joints per target.

    Targets with no valid joints use a root cost when a valid root exists, even
    on otherwise supervised frames. ``root_only`` selects that cost for every
    target on a frame without keypoint labels.
    """
    Q = pred_logits.shape[0]
    G = gt_keypoints.shape[0]
    if G == 0:
        return None

    prob = pred_logits.sigmoid().view(Q)
    neg = (1 - focal_alpha) * (prob ** focal_gamma) * (-(1 - prob).clamp(min=1e-8).log())
    pos = focal_alpha * ((1 - prob) ** focal_gamma) * (-(prob).clamp(min=1e-8).log())
    cls_cost = (pos - neg).view(Q, 1).expand(Q, G)

    if gt_keypoint_valid is None:
        joint_valid = torch.ones(
            (G, pred_keypoints.shape[-2]), dtype=torch.bool, device=pred_keypoints.device
        )
    else:
        joint_valid = gt_keypoint_valid.to(device=pred_keypoints.device, dtype=torch.bool)
    joint_count = joint_valid.sum(dim=-1)
    has_joints = joint_count > 0

    per_joint_cost = (pred_keypoints[:, None] - gt_keypoints[None]).abs().mean(dim=-1)
    kpt_cost = torch.where(
        joint_valid[None], per_joint_cost, torch.zeros_like(per_joint_cost)
    ).sum(dim=-1)
    kpt_cost = kpt_cost / joint_count.clamp(min=1).to(kpt_cost.dtype)[None]

    if gt_root is None:
        root_cost = pred_keypoints.new_zeros((Q, G))
        root_valid = torch.zeros(G, dtype=torch.bool, device=pred_keypoints.device)
    else:
        pred_root = pred_keypoints[:, ROOT_SLICE, :].mean(dim=1)
        root_cost = (pred_root[:, None] - gt_root.to(pred_root.device)[None]).abs().mean(dim=-1)
        root_valid = (
            torch.ones(G, dtype=torch.bool, device=pred_keypoints.device)
            if gt_root_valid is None
            else gt_root_valid.to(device=pred_keypoints.device, dtype=torch.bool)
        )

    use_root = root_only | ~has_joints
    kpt_component = (
        cost_kpt * kpt_cost * has_joints[None].to(kpt_cost.dtype)
        * (not root_only)
    )
    root_weight = torch.where(
        use_root,
        torch.full_like(joint_count, cost_root, dtype=pred_keypoints.dtype),
        torch.full_like(joint_count, cost_root_labeled, dtype=pred_keypoints.dtype),
    )
    root_cost = torch.where(root_valid[None], root_cost, torch.zeros_like(root_cost))
    root_component = root_weight[None] * root_cost
    cost = cost_class * cls_cost + kpt_component + root_component
    if extra_cost is not None:
        # Targets without 3-D geometry (e.g. seen by one camera only) are matched on
        # ``extra_cost`` (models/supervision.py: reprojection error) instead.
        mask = extra_mask.to(device=cost.device, dtype=torch.bool)[None]
        cost = torch.where(mask, cost_class * cls_cost + cost_kpt * extra_cost.to(cost.dtype), cost)
    return cost


def to_device(values, device, dtype=torch.long) -> torch.Tensor:
    """Small host list/tensor -> device without a stream sync (pinned, non-blocking copy;
    a pageable host-to-device copy waits for all queued GPU work first)."""
    t = values if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=dtype)
    device = torch.device(device)
    if device.type != "cuda":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


def solve_costs(costs: list) -> list:
    """Hungarian assignment of several cost matrices with a single device-to-host transfer.

    ``costs``: list of (Q_i, G_i) tensors or None.  Returns [(rows, cols)] as Python lists."""
    present = [c for c in costs if c is not None]
    out = []
    if present:
        flat = torch.cat([c.float().reshape(-1) for c in present]).cpu().numpy()
    start = 0
    for c in costs:
        if c is None:
            out.append(([], []))
            continue
        size = c.numel()
        row, col = linear_sum_assignment(flat[start:start + size].reshape(c.shape))
        start += size
        out.append((row.tolist(), col.tolist()))
    return out
