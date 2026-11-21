"""Evaluation metrics for multi-person 3D keypoints and roots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MetricsAccumulator:
    pck_thresholds_mm: List[float] = field(default_factory=lambda: [50.0, 100.0, 150.0])
    match_threshold_mm: float = 500.0

    def __post_init__(self):
        self.sum_mpjpe = 0.0
        self.num_matched_joints = 0
        self.num_matched_persons = 0
        self.sum_root_error = 0.0
        self.num_root_matches = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.frames = 0
        self.count_error = 0
        self.pck_hits = {t: 0 for t in self.pck_thresholds_mm}
        self.ambiguous = 0

    def credit_ambiguous(self, count: int):
        """Unmatched predictions that may be present people without any geometry (approximate GT):
        moved from false positives to ``ambiguous_predictions``."""
        count = max(0, min(int(count), self.fp))
        self.fp -= count
        self.ambiguous += count

    def update_frame(
        self,
        pred_kpts_mm: np.ndarray,
        gt_kpts_mm: np.ndarray,
        *,
        keypoints_supervised: bool = True,
        pred_root_mm: np.ndarray | None = None,
        gt_root_mm: np.ndarray | None = None,
        gt_root_valid: np.ndarray | None = None,
        gt_keypoint_valid: np.ndarray | None = None,
    ):
        """Update count/detection metrics, and pose/root metrics where labeled."""
        P, G = pred_kpts_mm.shape[0], gt_kpts_mm.shape[0]
        self.frames += 1
        self.count_error += abs(P - G)
        if P == 0 and G == 0:
            return []
        if P == 0:
            self.fn += G
            return []
        if G == 0:
            self.fp += P
            return []

        if gt_keypoint_valid is None:
            joint_valid = np.full((G, gt_kpts_mm.shape[1]), keypoints_supervised, dtype=bool)
        else:
            joint_valid = np.asarray(gt_keypoint_valid, dtype=bool)
        has_joints = joint_valid.any(axis=-1)
        if pred_root_mm is not None and gt_root_mm is not None:
            root_cost = np.linalg.norm(
                np.asarray(pred_root_mm)[:, None] - np.asarray(gt_root_mm)[None, :], axis=-1
            )
        else:
            root_cost = np.zeros((P, G), dtype=np.float64)
        root_valid_for_match = (np.ones(G, dtype=bool) if gt_root_valid is None
                                 else np.asarray(gt_root_valid, dtype=bool))
        root_cost = np.where(root_valid_for_match[None], root_cost, 0.0)
        if keypoints_supervised:
            dist = np.linalg.norm(pred_kpts_mm[:, None] - gt_kpts_mm[None, :], axis=-1)
            kpt_cost = np.where(joint_valid[None], dist, 0.0).sum(axis=-1) / np.maximum(joint_valid.sum(axis=-1), 1)[None]
            cost = np.where(has_joints[None], kpt_cost, root_cost)
        else:
            cost = root_cost
        row, col = linear_sum_assignment(cost)

        matched = 0
        used_pred, used_gt = set(), set()
        associations = []
        root_valid = (np.ones(G, dtype=bool) if gt_root_valid is None
                      else np.asarray(gt_root_valid, dtype=bool))
        for r, c in zip(row, col):
            if cost[r, c] <= self.match_threshold_mm:
                matched += 1
                used_pred.add(r)
                used_gt.add(c)
                associations.append((int(r), int(c)))
                self.num_matched_persons += 1
                valid_joints = joint_valid[c]
                if keypoints_supervised and valid_joints.any():
                    per_joint = np.linalg.norm(pred_kpts_mm[r] - gt_kpts_mm[c], axis=-1)[valid_joints]
                    self.sum_mpjpe += float(per_joint.sum())
                    self.num_matched_joints += per_joint.shape[0]
                    for threshold in self.pck_thresholds_mm:
                        self.pck_hits[threshold] += int((per_joint <= threshold).sum())
                if root_valid[c] and pred_root_mm is not None and gt_root_mm is not None:
                    self.sum_root_error += float(np.linalg.norm(pred_root_mm[r] - gt_root_mm[c]))
                    self.num_root_matches += 1
        self.tp += matched
        self.fp += P - len(used_pred)
        self.fn += G - len(used_gt)
        return associations

    def compute(self) -> dict:
        mpjpe = self.sum_mpjpe / self.num_matched_joints if self.num_matched_joints else float("nan")
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        out = {
            "mpjpe_mm": mpjpe,
            "root_error_mm": self.sum_root_error / self.num_root_matches if self.num_root_matches else float("nan"),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_persons": self.num_matched_persons,
            "target_persons": self.tp + self.fn,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "matched_coverage": self.tp / max(self.tp + self.fn, 1),
            "count_mae": self.count_error / max(self.frames, 1),
            "ambiguous_predictions": self.ambiguous,
        }
        for threshold in self.pck_thresholds_mm:
            out[f"pck@{int(threshold)}mm"] = (
                self.pck_hits[threshold] / self.num_matched_joints
                if self.num_matched_joints else float("nan")
            )
        return out


class IdentityAccumulator:
    """ID switches/fragments of predicted labels per ground-truth identity.

    Labels are query-slot indices for window models (reset every window) or
    model track ids for streaming track-query models (reset only at a lane's
    clip start).  Each lane keeps a segment per clip.  Re-identification
    takeovers (``alias``) relabel a provisional id retroactively within its
    segment: ``id_*`` metrics use the resolved labels (the delayed-decision
    output), ``id_*_online`` the labels as first emitted (causal).
    """

    def __init__(self):
        self.matched = 0
        self.targets = 0
        self.segments: list[dict] = []
        self.current: dict[int, dict] = {}

    def reset_clip(self, lane: int = 0):
        self.current[lane] = {"frames": [], "aliases": {}}
        self.segments.append(self.current[lane])

    def alias(self, lane: int, provisional: int, recovered: int):
        if lane in self.current:
            self.current[lane]["aliases"][provisional] = recovered

    def update(self, query_ids: np.ndarray, instance_ids: np.ndarray, associations: list[tuple[int, int]],
               lane: int = 0):
        if lane not in self.current:
            self.reset_clip(lane)
        self.targets += len(instance_ids)
        self.matched += len(associations)
        self.current[lane]["frames"].append(
            [(int(instance_ids[target]), int(query_ids[prediction])) for prediction, target in associations])

    @staticmethod
    def _count(segment: dict, resolve: bool) -> tuple[int, int, int]:
        aliases = segment["aliases"]

        def label(value):
            seen = set()
            while resolve and value in aliases and value not in seen:
                seen.add(value)
                value = aliases[value]
            return value

        switches = fragments = transitions = 0
        last, previous = {}, set()
        for frame in segment["frames"]:
            current = set()
            for identity, value in frame:
                value = label(value)
                if identity in last:
                    transitions += 1
                    switches += value != last[identity]
                if identity not in previous:
                    fragments += 1
                last[identity] = value
                current.add(identity)
            previous = current
        return switches, fragments, transitions

    def compute(self) -> dict[str, float]:
        out = {}
        for suffix, resolve in (("", True), ("_online", False)):
            counts = [self._count(segment, resolve) for segment in self.segments]
            switches, fragments, transitions = (sum(c[i] for c in counts) for i in range(3))
            out[f"id_switches{suffix}"] = switches
            out[f"id_switch_rate{suffix}"] = switches / max(transitions, 1)
            if not suffix:
                out["id_fragments"] = fragments
                out["id_association_transitions"] = transitions
        out.update({"id_reid_takeovers": sum(len(s["aliases"]) for s in self.segments),
                    "id_matched_frames": self.matched, "id_target_frames": self.targets,
                    "id_matched_coverage": self.matched / max(self.targets, 1)})
        return out
