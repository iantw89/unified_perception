"""Evaluation metrics for multi-person 3D keypoints.

Predicted person queries (after a confidence threshold) are matched to ground
truth persons per frame with the Hungarian algorithm on mean keypoint distance.
Reports MPJPE (mm), detection precision/recall/F1, and PCK at several
thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class MetricsAccumulator:
    pck_thresholds_mm: List[float] = field(default_factory=lambda: [50.0, 100.0, 150.0])
    match_threshold_mm: float = 500.0

    def __post_init__(self):
        self.sum_mpjpe = 0.0
        self.num_matched_joints = 0
        self.num_matched_persons = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.pck_hits = {t: 0 for t in self.pck_thresholds_mm}

    @torch.no_grad()
    def update_frame(self, pred_kpts_mm: np.ndarray, gt_kpts_mm: np.ndarray):
        """pred_kpts_mm: (P, J, 3) selected detections; gt_kpts_mm: (G, J, 3)."""
        P = pred_kpts_mm.shape[0]
        G = gt_kpts_mm.shape[0]
        if P == 0 and G == 0:
            return
        if P == 0:
            self.fn += G
            return
        if G == 0:
            self.fp += P
            return

        # Cost = mean joint distance between each pred and gt.
        diff = pred_kpts_mm[:, None] - gt_kpts_mm[None, :]  # (P, G, J, 3)
        dist = np.linalg.norm(diff, axis=-1)                # (P, G, J)
        cost = dist.mean(axis=-1)                           # (P, G)
        row, col = linear_sum_assignment(cost)

        matched = 0
        used_pred, used_gt = set(), set()
        for r, c in zip(row, col):
            if cost[r, c] <= self.match_threshold_mm:
                matched += 1
                used_pred.add(r)
                used_gt.add(c)
                per_joint = dist[r, c]  # (J,)
                self.sum_mpjpe += float(per_joint.sum())
                self.num_matched_joints += per_joint.shape[0]
                self.num_matched_persons += 1
                for thr in self.pck_thresholds_mm:
                    self.pck_hits[thr] += int((per_joint <= thr).sum())
        self.tp += matched
        self.fp += P - len(used_pred)
        self.fn += G - len(used_gt)

    def compute(self) -> dict:
        mpjpe = self.sum_mpjpe / max(self.num_matched_joints, 1)
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        out = {
            "mpjpe_mm": mpjpe,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_persons": self.num_matched_persons,
        }
        for thr in self.pck_thresholds_mm:
            out[f"pck@{int(thr)}mm"] = self.pck_hits[thr] / max(self.num_matched_joints, 1)
        return out
