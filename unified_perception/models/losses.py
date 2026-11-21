"""Set-prediction losses for dense and sparse 3D keypoint supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss
from ..constants import ROOT_JOINTS
from .matcher import hungarian_match
from .reid import ReIDLoss
from .matcher import to_device
from .supervision import LossSettings, Supervision, batch_pair_losses, match, pair_losses


class SetCriterion(torch.nn.Module):
    def __init__(self, cfg_model, cfg_loss=None):
        super().__init__()
        m = cfg_model
        self.cost_class = m.cost_class
        self.cost_kpt = m.cost_kpt
        self.cost_root = getattr(m, "cost_root", 5.0)
        self.cost_root_labeled = getattr(m, "cost_root_labeled", 0.0)
        self.w_class = m.loss_class
        self.w_kpt = m.loss_kpt
        self.w_root = getattr(cfg_loss, "root_weight", 0.0)
        self.alpha = m.focal_alpha
        self.gamma = m.focal_gamma
        self.dn_weight = getattr(m, "dn_weight", 1.0)
        self.settings = LossSettings.from_config(cfg_loss)
        # Track mode: reuse the final layer's matches (which created the births) for every
        # auxiliary decoder layer and evaluate all layers in one vectorized pass.
        self.share_aux_matching = bool(getattr(cfg_loss, "share_aux_matching", False))
        self.costs = {"class": m.cost_class, "kpt": m.cost_kpt, "alpha": m.focal_alpha, "gamma": m.focal_gamma,
                      "root": self.cost_root, "root_labeled": self.cost_root_labeled}
        self.reid_weight = getattr(m, "reid_loss_weight", 0.0)
        self.reid = None
        if getattr(m, "reid_dims", 0) and getattr(m, "track_queries", False):
            self.reid = ReIDLoss(m.reid_temperature, m.reid_bank_recent, m.reid_bank_anchors,
                                 m.reid_bank_stride)

    def forward(
        self,
        cls_scores: torch.Tensor,
        keypoints: torch.Tensor,
        gt_keypoints: torch.Tensor,
        gt_valid: torch.Tensor,
        dn_meta=None,
        gt_keypoint_frame_valid: torch.Tensor | None = None,
        gt_root: torch.Tensor | None = None,
        gt_root_valid: torch.Tensor | None = None,
        gt_keypoint_valid: torch.Tensor | None = None,
        track_output: dict | None = None,
        gt_keypoint_sigma: torch.Tensor | None = None,
        gt_root_sigma: torch.Tensor | None = None,
        gt_kp2d: torch.Tensor | None = None,
        gt_kp2d_valid: torch.Tensor | None = None,
        world2img: torch.Tensor | None = None,
        position_range: torch.Tensor | None = None,
        gt_match_root: torch.Tensor | None = None,
        gt_match_root_valid: torch.Tensor | None = None,
    ):
        """Compute losses. Keypoints and roots are in normalized model space.

        ``track_output`` (from a track-query head in training) fixes the query
        assignment of track slots by identity; detection queries are Hungarian
        matched per decoder layer against people no track claimed, except the
        final layer, which reuses the head's matches (they created the births).

        Approximate GT (models/supervision.py): ``gt_keypoint_sigma`` / ``gt_root_sigma``
        (mm) make the 3-D terms noise-tolerant; ``gt_kp2d`` (+ ``world2img``,
        ``position_range``) supervise people seen without a 3-D target by reprojection.
        """
        L, B, T, Q, _ = cls_scores.shape
        device = cls_scores.device
        if gt_keypoint_frame_valid is None:
            gt_keypoint_frame_valid = torch.ones((B, T), dtype=torch.bool, device=device)
        else:
            gt_keypoint_frame_valid = gt_keypoint_frame_valid.to(device=device, dtype=torch.bool)
        if gt_keypoint_valid is None:
            gt_keypoint_valid = gt_valid.bool().unsqueeze(-1).expand(
                *gt_valid.shape, gt_keypoints.shape[-2]
            )
        gt_keypoint_valid = (
            gt_keypoint_valid.to(device=device, dtype=torch.bool)
            & gt_valid.to(device=device, dtype=torch.bool).unsqueeze(-1)
            & gt_keypoint_frame_valid[:, :, None, None]
        )
        if gt_root is None:
            gt_root = gt_keypoints[..., list(ROOT_JOINTS), :].mean(dim=-2)
        if gt_root_valid is None:
            gt_root_valid = gt_valid
        gt_root_valid = gt_root_valid.to(device=device, dtype=torch.bool) & gt_valid

        sup = Supervision({
            "gt": gt_keypoints, "valid": gt_valid, "keypoint_valid": gt_keypoint_valid,
            "frame_valid": gt_keypoint_frame_valid, "root": gt_root, "root_valid": gt_root_valid,
            "sigma_mm": gt_keypoint_sigma, "root_sigma_mm": gt_root_sigma,
            "kp2d": gt_kp2d, "kp2d_valid": gt_kp2d_valid, "world2img": world2img,
            "match_root": gt_match_root, "match_root_valid": gt_match_root_valid,
        }, position_range)
        sums = {key: cls_scores.new_zeros(()) for key in ("cls", "kpt", "root", "reproj")}
        counts = {key: 0 for key in ("pos", "kpt", "root", "reproj")}
        assignments = None if track_output is None else track_output.get("assignments")
        num_track_pairs = 0
        if assignments is not None and self.share_aux_matching:
            num_track_pairs = self._shared_track_loss(sup, cls_scores, keypoints, assignments, sums, counts)
        for lvl in range(L if not (assignments is not None and self.share_aux_matching) else 0):
            for b in range(B):
                for t in range(T):
                    frame = sup.frame(b, t)
                    logits, preds = cls_scores[lvl, b, t], keypoints[lvl, b, t]
                    weight = None
                    if assignments is not None:
                        queries, people, weight, tracked = self._tracked_pairs(
                            frame, logits, preds, assignments[t][b], track_output["num_det"], lvl == L - 1)
                        if lvl == 0:
                            num_track_pairs += tracked
                    else:
                        valid = torch.nonzero(frame.valid).flatten().tolist()
                        if len(valid) > Q:
                            raise ValueError(f"{len(valid)} ground-truth people exceed {Q} prediction queries")
                        queries, people = self._window_pairs(frame, logits, preds, valid)
                    target = torch.zeros((Q, 1), device=device, dtype=logits.dtype)
                    if queries:
                        target[torch.as_tensor(queries, device=device)] = 1.0
                    focal = sigmoid_focal_loss(logits, target, self.alpha, self.gamma, reduction="none")
                    sums["cls"] = sums["cls"] + (focal if weight is None else focal * weight).sum()
                    located = [(q, g) for q, g in zip(queries, people) if bool(frame.valid[g])]
                    terms = pair_losses(frame, preds, [q for q, _ in located], [g for _, g in located],
                                        self.settings)
                    for key in ("kpt", "root", "reproj"):
                        sums[key] = sums[key] + terms[key]
                        if lvl == 0:
                            counts[key] += terms[f"n_{key}"]
                    if lvl == 0:
                        counts["pos"] += int(frame.valid.sum())
        def at_least_one(count):   # counts are host ints or (shared path) device tensors
            return count.clamp(min=1) if isinstance(count, torch.Tensor) else max(count, 1)

        loss_cls = self.w_class * sums["cls"] / (at_least_one(counts["pos"]) * L)
        loss_kpt = self.w_kpt * sums["kpt"] / (at_least_one(counts["kpt"]) * 3 * L)
        loss_root = self.w_root * sums["root"] / (at_least_one(counts["root"]) * 3 * L)
        loss_reproj = self.settings.reproj_weight * sums["reproj"] / (at_least_one(counts["reproj"]) * L)
        total = loss_cls + loss_kpt + loss_root + loss_reproj
        # Metric values are gathered as tensors and read back in one transfer at the end.
        values = {"loss_cls": loss_cls, "loss_kpt": loss_kpt, "loss_root": loss_root,
                  "num_pos": counts["pos"], "num_kpt": counts["kpt"], "num_root": counts["root"],
                  "loss_reproj": loss_reproj, "num_reproj": counts["reproj"]}
        extra = {}
        if assignments is not None:
            extra["num_track_pairs"] = num_track_pairs
        if assignments is not None and self.reid is not None and "id_codes" in track_output:
            loss_reid, anchors = self._reid_loss(track_output, assignments)
            extra["num_reid_anchors"] = anchors
            if loss_reid is not None:
                loss_reid = self.reid_weight * loss_reid
                total = total + loss_reid
                values["loss_reid"] = loss_reid

        if dn_meta is not None:
            dn_cls_loss, dn_kpt_loss = self._dn_loss(dn_meta, gt_keypoint_frame_valid)
            total = total + dn_cls_loss + dn_kpt_loss
            values["dn_loss_cls"], values["dn_loss_kpt"] = dn_cls_loss, dn_kpt_loss
        values["loss"] = total

        on_device = [k for k, v in values.items() if isinstance(v, torch.Tensor)]
        host = torch.stack([values[k].detach().float().reshape(()) for k in on_device]).tolist()
        metrics = {k: v for k, v in values.items() if not isinstance(v, torch.Tensor)}
        metrics.update(zip(on_device, host))
        for key in ("num_pos", "num_kpt", "num_root", "num_reproj"):
            metrics[key] = int(metrics[key])
        if not metrics["num_reproj"]:
            del metrics["loss_reproj"], metrics["num_reproj"]
        metrics.update(extra)
        return total, metrics

    def _shared_track_loss(self, sup, cls_scores, keypoints, assignments, sums, counts) -> int:
        """Track mode with shared matching: every (layer, lane, frame) at once.

        The final layer's assignments (track identities + the head's detection matches) apply
        to every decoder layer; ignored / invalid slots get no class loss.  Adds into ``sums``
        / ``counts`` and returns the number of track pairs.
        """
        L, B, T, Q, _ = cls_scores.shape
        device = cls_scores.device
        pairs, excluded, tracked = [[], [], [], []], [[], [], []], 0
        for t in range(T):
            for b in range(B):
                a = assignments[t][b]
                queries = list(a["track_q"]) + list(a["det_q"])
                people = list(a["track_g"]) + list(a["det_g"])
                pairs[0] += [b] * len(queries)
                pairs[1] += [t] * len(queries)
                pairs[2] += queries
                pairs[3] += people
                skip = list(a["ignore"]) + list(a["invalid"])
                excluded[0] += [b] * len(skip)
                excluded[1] += [t] * len(skip)
                excluded[2] += skip
                tracked += len(a["track_q"])
        pb, pt, pq, pg = to_device(pairs, device).unbind(0)
        target = torch.zeros((B, T, Q), device=device, dtype=cls_scores.dtype)
        target[pb, pt, pq] = 1.0
        weight = torch.ones((B, T, Q), device=device, dtype=cls_scores.dtype)
        if excluded[0]:
            eb, et, eq = to_device(excluded, device).unbind(0)
            weight[eb, et, eq] = 0.0
        focal = sigmoid_focal_loss(cls_scores[..., 0], target.expand(L, B, T, Q), self.alpha, self.gamma,
                                   reduction="none")
        sums["cls"] = sums["cls"] + (focal * weight).sum()
        terms = batch_pair_losses(sup, keypoints, pb, pt, pq, pg, self.settings)
        for key in ("kpt", "root", "reproj"):
            sums[key] = sums[key] + terms[key]
            counts[key] = counts[key] + terms[f"n_{key}"]
        counts["pos"] += int(sup.cpu()["valid"].sum())
        return tracked

    def _reid_loss(self, out, assignments):
        """Identity codes of every identity-matched query (per frame) and track slot (EMA)."""
        codes, ema, ids = out["id_codes"], out["id_ema"], out["instance_ids"].cpu()
        num_det = out["num_det"]
        B, T = codes.shape[:2]
        samples = []
        for b in range(B):
            frame_rows, query_rows, slot_frames, slot_rows, labels, slot_labels = [], [], [], [], [], []
            for t in range(T):
                a = assignments[t][b]
                for q, g in zip(list(a["track_q"]) + list(a["det_q"]), list(a["track_g"]) + list(a["det_g"])):
                    identity = int(ids[b, t, g])
                    if identity < 0:
                        continue
                    frame_rows.append(t)
                    query_rows.append(q)
                    labels.append(identity)
                    if q >= num_det:  # the track's denoised (EMA) code is a second view
                        slot_frames.append(t)
                        slot_rows.append(q - num_det)
                        slot_labels.append(identity)
            if frame_rows:
                ft, fq = to_device([frame_rows, query_rows], codes.device).unbind(0)
                vectors = codes[b, ft, fq]
                if slot_rows:
                    st, sk = to_device([slot_frames, slot_rows], codes.device).unbind(0)
                    vectors = torch.cat([vectors, ema[b, st, sk]])
                samples.append((vectors, torch.tensor(labels + slot_labels), torch.tensor(frame_rows + slot_frames)))
            else:
                samples.append((codes.new_zeros(0, codes.shape[-1]), torch.zeros(0, dtype=torch.long),
                                torch.zeros(0, dtype=torch.long)))
        return self.reid(samples, out.get("lane_reset"), T)

    def _window_pairs(self, frame, logits, preds, people):
        """Per-window (no tracks) matching.  Unlocated positives keep the historical class-only
        match so known people are not trained as absent."""
        queries, matched = match(frame, logits, preds, people, self.costs, self.settings.reproj_scale_px)
        rest = [g for g in people if g not in set(matched)]
        if rest:
            free = [q for q in range(logits.shape[0]) if q not in set(queries)]
            order = torch.argsort(logits[free, 0], descending=True).tolist()
            for g, k in zip(rest, order):
                queries.append(free[k])
                matched.append(g)
        return queries, matched

    def _tracked_pairs(self, frame, logits, preds, assign, num_det, final):
        """(queries, people, class weight, n_track_pairs) of one (layer, lane, frame).

        Track slots keep their identity assignment on every layer; detection queries are
        matched against people no track claimed (the final layer reuses the head's matches,
        which created the births); ignored and invalid slots get no class loss.
        """
        queries, people = list(assign["track_q"]), list(assign["track_g"])
        if final:
            det_q, det_g = list(assign["det_q"]), list(assign["det_g"])
        else:
            claimed = set(people)
            free = [g for g in torch.nonzero(frame.valid).flatten().tolist() if g not in claimed]
            det_q, det_g = match(frame, logits[:num_det], preds[:num_det], free, self.costs,
                                 self.settings.reproj_scale_px)
        weight = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
        excluded = list(assign["ignore"]) + list(assign["invalid"])
        if excluded:
            weight[excluded] = 0.0
        return queries + det_q, people + det_g, weight, len(assign["track_q"])

    def _dn_loss(self, dn, gt_keypoint_frame_valid=None):
        """DN focal loss plus keypoint L1, excluding unsupervised frames."""
        dn_cls = dn["dn_cls"]
        dn_kpts = dn["dn_kpts"]
        tgt = dn["target_kpts"]
        valid = dn["valid"]
        person = dn["is_person"]
        target_joint_valid = dn.get("target_kpt_valid")

        L = dn_cls.shape[0]
        J = dn_kpts.shape[-2]
        device = dn_cls.device
        cls_target = person.float().unsqueeze(-1)
        cls_weight = valid.float().unsqueeze(-1)
        pmask = valid & person
        if gt_keypoint_frame_valid is not None:
            frame_mask = gt_keypoint_frame_valid.to(device=device, dtype=torch.bool)
            cls_weight = cls_weight * frame_mask[:, :, None, None]
            pmask = pmask & frame_mask[:, :, None]
        num_pos = torch.clamp(pmask.sum().int(), min=1)

        # All layers at once, no host syncs.
        fl = sigmoid_focal_loss(dn_cls, cls_target.expand_as(dn_cls), self.alpha, self.gamma, reduction="none")
        total_cls = (fl * cls_weight).sum()
        if target_joint_valid is None:
            joint_mask = pmask[:, :, :, None].expand(*pmask.shape, J)
        else:
            joint_mask = pmask[:, :, :, None] & target_joint_valid.to(device=device, dtype=torch.bool)
        target = torch.where(joint_mask[..., None], tgt, 0.0)
        error = (dn_kpts - target).abs()
        total_kpt = torch.where(joint_mask[..., None], error, 0.0).sum()
        num_kpt = joint_mask.sum()

        loss_cls = self.w_class * total_cls / (num_pos * L)
        loss_kpt = self.w_kpt * total_kpt / (num_kpt.clamp(min=1) * 3 * L)
        return self.dn_weight * loss_cls, self.dn_weight * loss_kpt
