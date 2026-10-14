"""Set-prediction losses: sigmoid-focal classification + L1 keypoint regression.

Operates over all decoder layers (auxiliary supervision) and every frame of
every batch element.  Targets are matched to predictions per frame with the
Hungarian matcher.  When the StreamPETR query-denoising (DN) groups are present
their reconstruction loss is added directly (no matching: each reconstruct query
already corresponds to a known GT person).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss
from .matcher import hungarian_match


class SetCriterion(torch.nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        m = cfg_model
        self.cost_class = m.cost_class
        self.cost_kpt = m.cost_kpt
        self.w_class = m.loss_class
        self.w_kpt = m.loss_kpt
        self.alpha = m.focal_alpha
        self.gamma = m.focal_gamma
        self.dn_weight = getattr(m, "dn_weight", 1.0)

    def forward(self, cls_scores: torch.Tensor, keypoints: torch.Tensor, gt_keypoints: torch.Tensor, gt_valid: torch.Tensor, dn_meta=None):
        """Compute the total loss and a metrics dict.

        Args:
            cls_scores: (L, B, T, Q, 1)
            keypoints:  (L, B, T, Q, J, 3) normalised [0,1]
            gt_keypoints: (B, T, P, J, 3) normalised [0,1]
            gt_valid: (B, T, P) bool
            dn_meta: optional dict of denoising predictions/targets.
        """
        L, B, T, Q, _ = cls_scores.shape
        device = cls_scores.device
        total_cls = torch.zeros((), device=device)
        total_kpt = torch.zeros((), device=device)
        num_pos = 0

        for lvl in range(L):
            for b in range(B):
                for t in range(T):
                    logits = cls_scores[lvl, b, t]        # (Q, 1)
                    preds = keypoints[lvl, b, t]          # (Q, J, 3)
                    valid = gt_valid[b, t]                # (P,)
                    gts = gt_keypoints[b, t][valid]       # (G, J, 3)
                    G = gts.shape[0]

                    target = torch.zeros((Q, 1), device=device)
                    if G > 0:
                        pred_idx, gt_idx = hungarian_match(
                            logits, preds, gts,
                            self.cost_class, self.cost_kpt, self.alpha, self.gamma,
                        )
                        target[pred_idx] = 1.0
                        matched_pred = preds[pred_idx]      # (G, J, 3)
                        matched_gt = gts[gt_idx]            # (G, J, 3)
                        total_kpt = total_kpt + F.l1_loss(
                            matched_pred, matched_gt, reduction="sum"
                        )
                        num_pos += G
                    total_cls = total_cls + sigmoid_focal_loss(
                        logits, target, self.alpha, self.gamma, reduction="sum"
                    )

        denom_pos = max(num_pos, 1)
        # cls normalised by number of positives (DETR convention); averaged over layers.
        loss_cls = self.w_class * total_cls / (denom_pos * L)
        loss_kpt = self.w_kpt * total_kpt / (denom_pos * keypoints.shape[-2] * 3 * L)
        total = loss_cls + loss_kpt
        metrics = {
            "loss": float(total.detach()),
            "loss_cls": float(loss_cls.detach()),
            "loss_kpt": float(loss_kpt.detach()),
            "num_pos": num_pos,
        }

        if dn_meta is not None:
            dn_cls_loss, dn_kpt_loss = self._dn_loss(dn_meta)
            total = total + dn_cls_loss + dn_kpt_loss
            metrics["loss"] = float(total.detach())
            metrics["dn_loss_cls"] = float(dn_cls_loss.detach())
            metrics["dn_loss_kpt"] = float(dn_kpt_loss.detach())

        return total, metrics

    def _dn_loss(self, dn):
        """Denoising loss: focal on every reconstruct query, L1 on person ones."""
        dn_cls = dn["dn_cls"]        # (L, B, T, P, 1)
        dn_kpts = dn["dn_kpts"]      # (L, B, T, P, J, 3)
        tgt = dn["target_kpts"]      # (B, T, P, J, 3)
        valid = dn["valid"]          # (B, T, P)
        person = dn["is_person"]     # (B, T, P)

        L = dn_cls.shape[0]
        J = dn_kpts.shape[-2]
        device = dn_cls.device

        cls_target = person.float().unsqueeze(-1)          # (B, T, P, 1)
        cls_weight = valid.float().unsqueeze(-1)           # supervise filled slots only
        pmask = valid & person                             # (B, T, P)
        num_pos = torch.clamp(pmask.sum().int(), min=1)

        total_cls = torch.zeros((), device=device)
        total_kpt = torch.zeros((), device=device)
        for lvl in range(L):
            fl = sigmoid_focal_loss(
                dn_cls[lvl], cls_target, self.alpha, self.gamma, reduction="none"
            )
            total_cls = total_cls + (fl * cls_weight).sum()
            if pmask.any():
                total_kpt = total_kpt + F.l1_loss(
                    dn_kpts[lvl][pmask], tgt[pmask], reduction="sum"
                )

        loss_cls = self.w_class * total_cls / (num_pos * L)
        loss_kpt = self.w_kpt * total_kpt / (num_pos * J * 3 * L)
        return self.dn_weight * loss_cls, self.dn_weight * loss_kpt