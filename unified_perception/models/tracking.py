"""TrackFormer-style track queries for the streaming sparse head.

A fixed number of *track slots* (``model.max_tracks``) is appended to the
learned detection queries every frame.  A valid slot carries the previous
frame's decoder output embedding and predicted root (its reference point) and
owns one identity; invalid slots are masked out of self-attention and losses.

Training (identity supervision, ``TrackManager.train_step``)
    * a slot whose identity is present (``gt_valid``) is assigned to that person
      directly -- no Hungarian matching;
    * a slot whose identity exists but is unobserved (``instance_exists`` and
      not ``gt_valid``; providers fill short all-camera gaps, ``data.track_max_gap``)
      is *ignored*: no class/keypoint loss, and -- exactly as at inference -- the
      slot keeps its last supervised embedding/position while its age grows, so
      the same slot learns to re-acquire the person when they reappear;
    * a slot whose identity is gone is trained as background and then freed;
    * people not claimed by a track are Hungarian-matched to detection queries;
      final-layer matches become new tracks for the next frame;
    * augmentation: carried tracks are dropped with ``track_drop_prob`` (detection
      queries must re-find them) and an unmatched high-scoring detection is
      injected as a false track with ``track_fp_prob`` (it must learn to die).

Inference (``TrackManager.infer_step``)
    * an active slot scoring >= ``track_keep_score`` continues and is reported;
      below it the slot becomes inactive and keeps its last embedding/position
      (TrackFormer re-identification); it is revived when it scores >=
      ``track_birth_score`` again, and freed after ``track_max_inactive`` frames;
    * detection queries scoring >= ``track_birth_score`` start new identities unless
      their root lies within ``track_nms_mm`` of an already reported person.

Re-identification (``model.reid_dims > 0``, models/reid.py): every slot keeps
an EMA of its identity codes; ended tracks that lived at least
``reid_wait_frames`` enter a per-lane gallery, and a newly born track stays
provisional for ``reid_wait_frames`` active frames before it may take over the
most similar gallery identity (cosine >= ``reid_threshold``).  Takeovers are
returned as aliases (lane, provisional id, recovered id) so that consumers can
relabel the provisional frames retroactively.

The state (:class:`TrackState`) persists across ``forward`` calls so arbitrarily
long sequences can be processed window by window; lanes are reset individually.
All state tensors are detached: gradients never cross frames through the tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch

from ..constants import ROOT_JOINTS
from .matcher import hungarian_match, to_device
from .supervision import Supervision, match, match_many

FALSE_TRACK_ID = -2  # injected false-positive track: never present in ground truth


@dataclass
class TrackState:
    embed: torch.Tensor     # (B, K, C)
    ref: torch.Tensor       # (B, K, 3) normalized root reference point
    valid: torch.Tensor     # (B, K) bool
    ids: torch.Tensor       # (B, K) long; GT instance id (training) or running id (inference)
    inactive: torch.Tensor  # (B, K) long; frames since the slot was last updated
    next_id: torch.Tensor   # (B,) long; next running id per lane (inference)
    id_ema: torch.Tensor    # (B, K, D) EMA identity code (D = 0 without re-id)
    age: torch.Tensor       # (B, K) long; active frames since birth (inference)
    pending: torch.Tensor   # (B, K) bool; provisional id awaiting gallery matching
    gallery_code: torch.Tensor   # (B, G, D) identity codes of ended tracks
    gallery_ids: torch.Tensor    # (B, G) long
    gallery_valid: torch.Tensor  # (B, G) bool
    gallery_time: torch.Tensor   # (B, G) long; insertion clock (oldest replaced first)
    clock: torch.Tensor          # (B,) long; frames processed in the lane
    app_ema: torch.Tensor        # (B, K, E) EMA of pretrained re-ID crop embeddings (E = 0 if unused)
    app_valid: torch.Tensor      # (B, K) bool; app_ema holds at least one crop embedding

    @classmethod
    def empty(cls, batch: int, slots: int, dims: int, device, dtype,
              reid_dims: int = 0, gallery: int = 0, app_dims: int = 0) -> "TrackState":
        def z(*shape, kind=torch.long, fill=0):
            return torch.full(shape, fill, dtype=kind, device=device)
        return cls(
            embed=torch.zeros(batch, slots, dims, device=device, dtype=dtype),
            ref=torch.zeros(batch, slots, 3, device=device, dtype=dtype),
            valid=z(batch, slots, kind=torch.bool), ids=z(batch, slots, fill=-1),
            inactive=z(batch, slots), next_id=z(batch),
            id_ema=torch.zeros(batch, slots, reid_dims, device=device),
            age=z(batch, slots), pending=z(batch, slots, kind=torch.bool),
            gallery_code=torch.zeros(batch, gallery, reid_dims, device=device),
            gallery_ids=z(batch, gallery, fill=-1), gallery_valid=z(batch, gallery, kind=torch.bool),
            gallery_time=z(batch, gallery), clock=z(batch),
            app_ema=torch.zeros(batch, slots, app_dims, device=device),
            app_valid=z(batch, slots, kind=torch.bool),
        )

    def reset_lanes(self, lanes: torch.Tensor) -> None:
        lanes = lanes.to(device=self.valid.device, dtype=torch.bool)
        for f in fields(self):
            tensor = getattr(self, f.name)
            tensor[lanes] = -1 if f.name in ("ids", "gallery_ids") else 0

    def clone(self) -> "TrackState":
        return TrackState(**{f.name: getattr(self, f.name).clone() for f in fields(self)})


def root_of(kpts: torch.Tensor) -> torch.Tensor:
    return kpts[..., list(ROOT_JOINTS), :].mean(dim=-2)


class TrackManager:
    """Assignment and lifecycle rules; holds no tensors of its own."""

    def __init__(self, cfg_model, position_range):
        m = cfg_model
        self.cost_class = m.cost_class
        self.cost_kpt = m.cost_kpt
        self.cost_root = getattr(m, "cost_root", 5.0)
        self.cost_root_labeled = getattr(m, "cost_root_labeled", 0.0)
        self.alpha = m.focal_alpha
        self.gamma = m.focal_gamma
        self.drop_prob = m.track_drop_prob
        self.fp_prob = m.track_fp_prob
        self.ref_noise = m.track_ref_noise
        self.birth_score = m.track_birth_score
        self.keep_score = m.track_keep_score
        self.max_inactive = m.track_max_inactive
        self.nms_mm = m.track_nms_mm
        self.costs = {"class": m.cost_class, "kpt": m.cost_kpt, "alpha": m.focal_alpha, "gamma": m.focal_gamma,
                      "root": self.cost_root, "root_labeled": self.cost_root_labeled}
        self.reid_dims = getattr(m, "reid_dims", 0)
        self.reid_wait = getattr(m, "reid_wait_frames", 0)
        self.reid_threshold = getattr(m, "reid_threshold", 1.0)
        self.gallery_size = getattr(m, "reid_gallery_size", 0) if self.reid_dims else 0
        pr = torch.as_tensor(position_range, dtype=torch.float32)
        self.range_min, self.range_size = pr[:3], pr[3:] - pr[:3]
        self.position_range = pr

    # ------------------------------------------------------------------ #
    def match_detections(self, logits, preds, gts, roots, roots_valid, joints_valid, supervised: bool):
        """Hungarian match of detection queries against ground truth (indices into the inputs)."""
        return hungarian_match(
            logits, preds, gts, self.cost_class, self.cost_kpt, self.alpha, self.gamma,
            gt_root=roots, gt_root_valid=roots_valid, gt_keypoint_valid=joints_valid,
            cost_root=self.cost_root, cost_root_labeled=self.cost_root_labeled,
            root_only=not supervised,
        )

    def _place(self, state: TrackState, b: int, slot: int, embed, ref, identity: int, noise: bool,
               code=None, app=None):
        if noise and self.ref_noise > 0:
            ref = (ref + torch.randn_like(ref) * self.ref_noise).clamp(0.0, 1.0)
        state.embed[b, slot] = embed
        state.ref[b, slot] = ref
        state.valid[b, slot] = True
        state.ids[b, slot] = identity
        state.inactive[b, slot] = 0
        state.age[b, slot] = 0
        state.pending[b, slot] = False
        if code is not None and self.reid_dims:
            state.id_ema[b, slot] = code
        if state.app_ema.shape[-1]:
            state.app_valid[b, slot] = app is not None
            state.app_ema[b, slot] = 0 if app is None else app

    def _new_state(self, state: TrackState, K: int, feat) -> TrackState:
        new = TrackState.empty(state.valid.shape[0], K, feat.shape[-1], feat.device, feat.dtype,
                               self.reid_dims, self.gallery_size, state.app_ema.shape[-1])
        for name in ("next_id", "gallery_code", "gallery_ids", "gallery_valid", "gallery_time", "clock"):
            getattr(new, name).copy_(getattr(state, name))
        return new

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def train_step(self, state: TrackState, num_det: int, cls, kpts, feat, targets: dict, t: int,
                   codes=None, ema=None, app=None):
        """Assign queries for frame ``t`` and build the next frame's tracks.

        Args:
            cls (B, Qm, 1), kpts (B, Qm, J, 3), feat (B, Qm, C): final decoder layer.
            targets: per-batch tensors with a time axis (B, T, P, ...): ``gt``, ``valid``,
                ``keypoint_valid``, ``frame_valid`` (B, T), ``root``, ``root_valid``,
                ``instance_ids`` and ``exists``.
            codes (B, Qm, D) per-frame identity codes and ema (B, K, D) the slots' updated EMA
                codes (detached), when re-identification is enabled; app = (embeddings (B, K, E),
                valid (B, K)) crop appearance of the slots.
        Returns:
            (assignments, new_state): one dict per lane; ground-truth indices are into P.

        Bookkeeping runs on host copies (one transfer per frame), the lanes' Hungarian costs
        come back in one transfer, and the next state is written with a few batched index
        operations: the per-frame loop never waits on the device otherwise.
        """
        B, Qm = cls.shape[:2]
        K = Qm - num_det
        device = cls.device
        new = self._new_state(state, K, feat)
        supervision = targets.get("_supervision")
        if supervision is None:
            supervision = targets["_supervision"] = Supervision(targets, self.position_range)
        P = targets["valid"].shape[2]
        host = torch.cat([
            targets["valid"][:, t].long().reshape(-1), targets["instance_ids"][:, t].long().reshape(-1),
            targets["exists"][:, t].long().reshape(-1), state.valid.long().reshape(-1),
            state.ids.long().reshape(-1)]).cpu()
        valid_all, ids_all, exists_all, slot_valid, slot_ids = torch.split(host, [B * P] * 3 + [B * K] * 2)
        valid_all, ids_all = valid_all.view(B, P).bool(), ids_all.view(B, P)
        exists_all = exists_all.view(B, P).bool() | valid_all
        slot_valid, slot_ids = slot_valid.view(B, K).bool(), slot_ids.view(B, K)

        lanes, items = [], []
        for b in range(B):
            valid, ids, exists = valid_all[b], ids_all[b], exists_all[b]
            id_to_gt = {int(ids[g]): g for g in torch.nonzero(valid & (ids >= 0)).flatten().tolist()}
            existing = {int(i) for i in ids[exists & (ids >= 0)].tolist()}
            track_q, track_g, background, ignore, carry = [], [], [], [], []
            invalid = [num_det + k for k in range(K) if not bool(slot_valid[b, k])]
            for k in torch.nonzero(slot_valid[b]).flatten().tolist():
                identity, query = int(slot_ids[b, k]), num_det + k
                if identity in id_to_gt:
                    track_q.append(query)
                    track_g.append(id_to_gt[identity])
                    carry.append((query, identity))
                elif identity in existing:
                    ignore.append(query)
                    carry.append((-1 - k, identity))  # stale: keep slot k's last state
                else:
                    background.append(query)
            claimed = set(track_g)
            free = [g for g in torch.nonzero(valid).flatten().tolist() if g not in claimed]
            lanes.append((track_q, track_g, background, ignore, invalid, carry, ids))
            # Located people match on 3-D cost, 2-D-only ones on reprojection, unlocated ones
            # (no geometry) cannot start a track here (models/supervision.py).
            items.append((supervision.frame(b, t), cls[b, :num_det], kpts[b, :num_det], free))
        matches = match_many(items, self.costs)

        assignments = []
        frozen, placed, false_tracks = [], [], []   # (lane, slot, source k | query, identity)
        for b, ((track_q, track_g, background, ignore, invalid, carry, ids), (det_q, det_g)) in enumerate(
                zip(lanes, matches)):
            assignments.append({
                "track_q": track_q, "track_g": track_g, "det_q": det_q, "det_g": det_g,
                "background": background, "ignore": ignore, "invalid": invalid,
            })
            # Next frame: carried tracks, then births, then an optional false track.
            births = [(q, int(ids[g])) for q, g in zip(det_q, det_g) if int(ids[g]) >= 0]
            slot = 0
            for query, identity in carry + births:
                if slot >= K:
                    break
                if self.drop_prob > 0 and torch.rand(()) < self.drop_prob:
                    continue
                if query < 0:  # unobserved: frozen embedding/reference/identity code, one frame older
                    frozen.append((b, slot, -1 - query, identity))
                else:
                    placed.append((b, slot, query, identity))
                slot += 1
            if slot < K and self.fp_prob > 0 and torch.rand(()) < self.fp_prob and len(set(det_q)) < num_det:
                false_tracks.append((b, slot, det_q))
        self._write_slots(new, state, num_det, kpts, feat, codes, ema, app, frozen, placed, false_tracks, cls)
        return assignments, new

    def _write_slots(self, new: TrackState, state: TrackState, num_det: int, kpts, feat, codes, ema, app,
                     frozen: list, placed: list, false_tracks: list, cls) -> None:
        """Batched writes of the next frame's slots (see :meth:`train_step`)."""
        device = feat.device
        reid = bool(self.reid_dims)
        has_app = bool(new.app_ema.shape[-1])
        if frozen:
            fb, fs, fk, fid = to_device(list(zip(*frozen)), device).unbind(0)
            new.embed[fb, fs] = state.embed[fb, fk]
            new.ref[fb, fs] = state.ref[fb, fk]
            new.valid[fb, fs] = True
            new.ids[fb, fs] = fid
            new.inactive[fb, fs] = state.inactive[fb, fk] + 1
            if reid:
                new.id_ema[fb, fs] = state.id_ema[fb, fk]
            if has_app:
                ok = state.app_valid[fb, fk]
                new.app_valid[fb, fs] = ok
                new.app_ema[fb, fs] = torch.where(ok[:, None], state.app_ema[fb, fk], 0.0)
        rows = []   # (lane, slot, query, identity) of slots taking a query's prediction
        rows += placed
        if false_tracks:
            # Highest-scoring detection query not matched this frame becomes a false track.
            taken = torch.zeros(len(false_tracks), num_det, dtype=torch.bool)
            for i, (_, _, det_q) in enumerate(false_tracks):
                if det_q:
                    taken[i, det_q] = True
            lanes = to_device([b for b, _, _ in false_tracks], device)
            scores = cls[lanes, :num_det, 0].float().masked_fill(to_device(taken, device), float("-inf"))
            fq = scores.argmax(dim=1)
        if rows or false_tracks:
            if rows:
                qb, qs, qq, qid = to_device(list(zip(*rows)), device).unbind(0)
            else:
                qb = qs = qq = qid = torch.zeros(0, dtype=torch.long, device=device)
            n_rows = qb.shape[0]
            if false_tracks:
                fslots = to_device([s for _, s, _ in false_tracks], device)
                qb, qs = torch.cat([qb, lanes]), torch.cat([qs, fslots])
                qq = torch.cat([qq, fq])
                qid = torch.cat([qid, torch.full_like(fq, FALSE_TRACK_ID)])
            ref = root_of(kpts[qb, qq])
            if self.ref_noise > 0:
                ref = (ref + torch.randn_like(ref) * self.ref_noise).clamp(0.0, 1.0)
            new.embed[qb, qs] = feat[qb, qq]
            new.ref[qb, qs] = ref.to(new.ref.dtype)
            new.valid[qb, qs] = True
            new.ids[qb, qs] = qid
            new.inactive[qb, qs] = 0
            is_slot = qq >= num_det
            slot_index = (qq - num_det).clamp(min=0)
            if reid:
                # Track queries carry their EMA code, detections (and false tracks) their frame code.
                code = codes[qb, qq]
                code = torch.where(is_slot[:, None], ema[qb, slot_index], code)
                new.id_ema[qb, qs] = code
            if has_app:
                ok = torch.zeros_like(is_slot)
                emb = new.app_ema.new_zeros(qb.shape[0], new.app_ema.shape[-1])
                if app is not None:
                    real = torch.arange(qb.shape[0], device=device) < n_rows   # false tracks: none
                    ok = is_slot & real & app[1][qb, slot_index]
                    emb = torch.where(ok[:, None], app[0][qb, slot_index], emb)
                new.app_valid[qb, qs] = ok
                new.app_ema[qb, qs] = emb

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer_step(self, state: TrackState, num_det: int, cls, kpts, feat, codes=None, ema=None, app=None):
        """Return (reported ids (B, Qm) long, -1 = not reported; new state; aliases).

        ``aliases`` lists (lane, provisional id, recovered id) gallery takeovers.
        """
        B, Qm = cls.shape[:2]
        scores = cls[..., 0].float().sigmoid()
        roots_mm = root_of(kpts).float() * self.range_size.to(kpts.device) + self.range_min.to(kpts.device)
        reported = torch.full((B, Qm), -1, dtype=torch.long, device=cls.device)
        new = state.clone()
        new.clock += 1
        reid = self.reid_dims > 0 and self.gallery_size > 0
        aliases = []
        for b in range(B):
            kept_roots = []
            for k in torch.nonzero(state.valid[b]).flatten().tolist():
                query, score = num_det + k, float(scores[b, num_det + k])
                active = int(state.inactive[b, k]) == 0
                if score >= (self.keep_score if active else self.birth_score):
                    new.embed[b, k] = feat[b, query]
                    new.ref[b, k] = root_of(kpts[b, query])
                    new.inactive[b, k] = 0
                    new.age[b, k] = state.age[b, k] + 1
                    if app is not None:
                        new.app_ema[b, k], new.app_valid[b, k] = app[0][b, k], app[1][b, k]
                    if reid:
                        new.id_ema[b, k] = ema[b, k]
                        if bool(state.pending[b, k]) and int(new.age[b, k]) >= self.reid_wait:
                            self._match_gallery(new, b, k, aliases)
                    reported[b, query] = new.ids[b, k]
                    kept_roots.append(roots_mm[b, query])
                else:
                    new.inactive[b, k] = state.inactive[b, k] + 1
                    if int(new.inactive[b, k]) > self.max_inactive:
                        if reid and int(state.age[b, k]) >= self.reid_wait:
                            self._remember(new, b, k)
                        new.valid[b, k] = False
                        new.ids[b, k] = -1
                        new.inactive[b, k] = 0
                        new.pending[b, k] = False
            order = torch.argsort(scores[b, :num_det], descending=True).tolist()
            for query in order:
                if float(scores[b, query]) < self.birth_score:
                    break
                root = roots_mm[b, query]
                if kept_roots and torch.stack(kept_roots).sub(root).norm(dim=-1).min() < self.nms_mm:
                    continue
                free = torch.nonzero(~new.valid[b]).flatten()
                if not len(free):
                    break
                identity, slot = int(new.next_id[b]), int(free[0])
                new.next_id[b] += 1
                self._place(new, b, slot, feat[b, query], root_of(kpts[b, query]), identity, noise=False,
                            code=codes[b, query] if reid else None)
                new.age[b, slot] = 1
                new.pending[b, slot] = reid
                if reid and self.reid_wait <= 1 and bool(new.pending[b, slot]):
                    self._match_gallery(new, b, slot, aliases)
                reported[b, query] = new.ids[b, slot]
                kept_roots.append(root)
        return reported, new, aliases

    def _remember(self, state: TrackState, b: int, k: int) -> None:
        """Move an ended track's identity code into the lane's gallery (oldest entry replaced)."""
        free = torch.nonzero(~state.gallery_valid[b]).flatten()
        g = int(free[0]) if len(free) else int(torch.argmin(state.gallery_time[b]))
        state.gallery_code[b, g] = state.id_ema[b, k]
        state.gallery_ids[b, g] = state.ids[b, k]
        state.gallery_valid[b, g] = True
        state.gallery_time[b, g] = state.clock[b]

    def _match_gallery(self, state: TrackState, b: int, k: int, aliases: list) -> None:
        """Provisional track k may take over an ended (gallery) or still-inactive identity."""
        state.pending[b, k] = False
        code = torch.nn.functional.normalize(state.id_ema[b, k], dim=-1)
        held = state.ids[b][state.valid[b]]
        gallery_ok = state.gallery_valid[b] & ~torch.isin(state.gallery_ids[b], held)
        inactive_ok = state.valid[b] & (state.inactive[b] > 0)
        inactive_ok[k] = False
        sims = torch.cat([
            (state.gallery_code[b] @ code).masked_fill(~gallery_ok, -2.0),
            (torch.nn.functional.normalize(state.id_ema[b], dim=-1) @ code).masked_fill(~inactive_ok, -2.0),
        ])
        if not len(sims):
            return
        best = int(torch.argmax(sims))
        if float(sims[best]) < self.reid_threshold:
            return
        G = state.gallery_code.shape[1]
        if best < G:
            recovered = int(state.gallery_ids[b, best])
            state.gallery_valid[b, best] = False
        else:  # merge with a track that went inactive and never revived
            other = best - G
            recovered = int(state.ids[b, other])
            state.valid[b, other] = False
            state.ids[b, other] = -1
            state.inactive[b, other] = 0
            state.pending[b, other] = False
        aliases.append((b, int(state.ids[b, k]), recovered))
        state.ids[b, k] = recovered


def track_key_mask(
    base: Optional[torch.Tensor], num_queries_before: int, slots_valid: torch.Tensor,
    memory_len: int, num_heads: int,
) -> torch.Tensor:
    """Self-attention mask (B*H, Q, S) with invalid track slots removed as keys.

    ``base`` is the (Q0, Q0 + memory_len) mask over the queries before the track
    slots (DN groups + detection queries) or None.  Track slots are visible to
    every query; matching queries (detection + tracks) never see DN queries,
    which ``base`` already encodes for its rows.
    """
    B, K = slots_valid.shape
    Q0 = num_queries_before
    Q = Q0 + K
    S = Q + memory_len
    device = slots_valid.device
    mask = torch.zeros(Q, S, dtype=torch.bool, device=device)
    if base is not None:
        mask[:Q0, :Q0] = base[:Q0, :Q0]
        mask[:Q0, Q:] = base[:Q0, Q0:]
        # Track rows inherit the matching-query rule (cannot see DN queries).
        mask[Q0:, :Q0] = base[-1, :Q0]
    mask = mask.unsqueeze(0).repeat(B, 1, 1)
    mask[:, :, Q0:Q] |= ~slots_valid[:, None, :]
    return mask.repeat_interleave(num_heads, dim=0)
