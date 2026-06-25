"""Hungarian matcher for set prediction (per-frame person assignment)."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_match(
    pred_logits: torch.Tensor,    # (Q, 1)
    pred_keypoints: torch.Tensor, # (Q, J, 3) normalised
    gt_keypoints: torch.Tensor,   # (G, J, 3) normalised
    cost_class: float = 2.0,
    cost_kpt: float = 5.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
):
    """Return (pred_idx, gt_idx) LongTensors of matched pairs."""
    Q = pred_logits.shape[0]
    G = gt_keypoints.shape[0]
    if G == 0:
        empty = torch.zeros(0, dtype=torch.long, device=pred_logits.device)
        return empty, empty

    prob = pred_logits.sigmoid().view(Q)  # (Q,)
    # Focal classification cost (deformable-DETR form).
    neg = (1 - focal_alpha) * (prob ** focal_gamma) * (-(1 - prob).clamp(min=1e-8).log())
    pos = focal_alpha * ((1 - prob) ** focal_gamma) * (-(prob).clamp(min=1e-8).log())
    cls_cost = (pos - neg).view(Q, 1).expand(Q, G)  # (Q, G)

    # Keypoint L1 cost (mean over joints & coords).
    pk = pred_keypoints.view(Q, 1, -1)
    gk = gt_keypoints.view(1, G, -1)
    kpt_cost = (pk - gk).abs().mean(dim=-1)  # (Q, G)

    cost = cost_class * cls_cost + cost_kpt * kpt_cost
    cost = cost.cpu().numpy()
    row, col = linear_sum_assignment(cost)
    pred_idx = torch.as_tensor(row, dtype=torch.long, device=pred_logits.device)
    gt_idx = torch.as_tensor(col, dtype=torch.long, device=pred_logits.device)
    return pred_idx, gt_idx
