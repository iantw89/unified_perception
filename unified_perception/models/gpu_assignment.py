"""Track-query training assignment without host round trips (``model.gpu_assignment``).

The reference implementation (:meth:`TrackManager.train_step` + :func:`supervision.match_many`)
copies ids/masks to the host every frame, decides slot roles in Python and solves the Hungarian
problem with SciPy: two device->host synchronizations per frame, each of which drains the GPU
queue while Python works.  This module computes exactly the same rules with batched tensor
operations so the per-frame loop never waits for the device:

* **matching cost**: one (B, Q, P) tensor for all lanes (:func:`batch_match_cost`); people a
  detection query may not take (claimed by a track, unmatchable) get a constant row, which leaves
  the optimal assignment of the others unchanged;
* **assignment**: a batched GPU linear-assignment solver (:func:`solve_assignment`, the Triton
  kernel of ``torch-linear-assignment`` launched without validation syncs; SciPy on CPU);
* **slot rules** (:func:`train_step`): track claims, stale/background slots, births, random drops
  and the false-track injection as masks, cumulative sums and scatters.

Output assignments are tensors -- ``target`` (B, Qm) person index or -1 and ``excluded`` (B, Qm)
(ignored / invalid track slots) -- consumed by the dense losses in ``models/losses.py``.
Random draws come from the device generator, so a run is not bitwise identical to the reference
path; with ``track_drop_prob = track_fp_prob = track_ref_noise = 0`` the two agree exactly
whenever the optimal assignment is unique (tests/test_gpu_assignment.py).
"""

from __future__ import annotations

import torch

from ..constants import ROOT_JOINTS, ROOT_SLICE

FALSE_TRACK_ID = -2


# --------------------------------------------------------------------------- #
# Cost and solver                                                              #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batch_match_cost(sup, t: int, logits: torch.Tensor, preds: torch.Tensor, costs: dict,
                     scale_px: float = 64.0) -> torch.Tensor:
    """(B, Q, P) matching cost of every query against every person of frame ``t``.

    Mirrors :func:`supervision.match_cost` + :func:`matcher.hungarian_cost` per lane: annotated
    joints only, root cost for people without joints or on unsupervised frames, the interpolated
    root for match-only people and reprojection cost for 2-D-only people.  Entries of people that
    are not matchable are meaningless (callers mask them).
    """
    B, Q = logits.shape[:2]
    dtype = preds.dtype
    prob = logits.reshape(B, Q).sigmoid()
    alpha, gamma = costs["alpha"], costs["gamma"]
    neg = (1 - alpha) * (prob ** gamma) * (-(1 - prob).clamp(min=1e-8).log())
    pos = alpha * ((1 - prob) ** gamma) * (-(prob).clamp(min=1e-8).log())
    cls_cost = (pos - neg)[:, :, None]                                          # (B, Q, 1)

    joint_valid = sup.keypoint_valid[:, t]                                      # (B, P, J)
    joint_count = joint_valid.sum(-1)
    has_joints = joint_count > 0
    gts = torch.where(joint_valid[..., None], sup.gt[:, t], 0.0).to(dtype)    # NaN-free
    per_joint = (preds[:, :, None] - gts[:, None]).abs().mean(-1)               # (B, Q, P, J)
    kpt_cost = torch.where(joint_valid[:, None], per_joint, 0.0).sum(-1)
    kpt_cost = kpt_cost / joint_count.clamp(min=1).to(kpt_cost.dtype)[:, None]

    match_only = sup.match_only[:, t]                                           # (B, P)
    roots = sup.root[:, t]
    roots_valid = sup.root_valid[:, t]
    if sup.match_root is not None:
        roots = torch.where(match_only[..., None], sup.match_root[:, t].to(roots.dtype), roots)
        roots_valid = roots_valid | match_only
    roots = torch.where(roots_valid[..., None], roots, 0.0).to(dtype)
    pred_root = preds[:, :, ROOT_SLICE].mean(-2)                         # (B, Q, 3)
    root_cost = (pred_root[:, :, None] - roots[:, None]).abs().mean(-1)         # (B, Q, P)
    root_cost = torch.where(roots_valid[:, None], root_cost, 0.0)

    root_only = ~sup.frame_valid[:, t].bool()                                   # (B,)
    use_root = root_only[:, None] | ~has_joints                                 # (B, P)
    kpt_component = costs["kpt"] * kpt_cost * (has_joints & ~root_only[:, None])[:, None].to(dtype)
    root_weight = torch.where(use_root, costs["root"], costs["root_labeled"]).to(dtype)
    cost = costs["class"] * cls_cost + kpt_component + root_weight[:, None] * root_cost

    if sup.kp2d is not None:
        extra_mask = ~sup.located[:, t] & ~match_only                           # (B, P)
        cost = torch.where(extra_mask[:, None],
                           costs["class"] * cls_cost + costs["kpt"] * _reprojection_cost(sup, t, preds, scale_px),
                           cost)
    return cost.float()


def _reprojection_cost(sup, t: int, preds: torch.Tensor, scale_px: float) -> torch.Tensor:
    """(B, Q, P) mean reprojection error / scale over each person's 2-D joints (all lanes)."""
    w2i = sup.world2img[:, t].float()                                           # (B, N, 4, 4)
    mm = preds.float() * sup.range_size + sup.range_min                         # (B, Q, J, 3)
    homog = torch.cat([mm, torch.ones_like(mm[..., :1])], dim=-1)
    cam = torch.einsum("bnij,bqkj->bnqki", w2i, homog)                          # (B, N, Q, J, 4)
    depth = cam[..., 2]
    uv = cam[..., :2] / depth.clamp(min=1e-3)[..., None]
    valid = sup.kp2d_valid[:, t]                                                # (B, N, P, J)
    target = torch.where(valid[..., None], sup.kp2d[:, t], 0.0)[..., :2].float()
    err = (uv[:, :, :, None] - target[:, :, None]).norm(dim=-1)                 # (B, N, Q, P, J)
    err = torch.where(depth[:, :, :, None] > 0, err, torch.full_like(err, 10 * scale_px))
    mask = valid[:, :, None].float()
    return ((err * mask).sum(dim=(1, 4)) / mask.sum(dim=(1, 4)).clamp(min=1) / scale_px).to(preds.dtype)


_BIG_COST = 1.0e6
_TRITON_OK: bool | None = None


def _triton_solver():
    global _TRITON_OK
    if _TRITON_OK is False:
        return None
    try:
        from torch_linear_assignment import _triton
        _TRITON_OK = True
        return _triton
    except Exception:  # pragma: no cover - optional dependency
        _TRITON_OK = False
        return None


@torch.no_grad()
def solve_assignment(cost: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Batched Hungarian: ``cost`` (B, Q, P), ``available`` (B, P) people that may be matched.

    Returns (B, Q) long: the person each query is matched to, or -1.  Requires Q >= P (more
    queries than people).  Unavailable people get a constant cost row, so they absorb leftover
    queries without changing the optimum of the available ones.
    """
    B, Q, P = cost.shape
    out = torch.full((B, Q + 1), -1, dtype=torch.long, device=cost.device)
    if B == 0 or P == 0:
        return out[:, :Q]
    if P > Q:
        raise ValueError(f"{P} people exceed {Q} detection queries")
    rows = torch.where(available[:, :, None], cost.transpose(1, 2), 0.0).float()               # (B, P, Q)
    # The solver runs without validation (no sync): a NaN/inf row would come back unassigned (-1).
    # Non-finite costs (a diverged forward) become a large finite cost instead; the step's loss is
    # non-finite too and is skipped on the device (engine/step.py), callers count these entries.
    rows = torch.nan_to_num(rows, nan=_BIG_COST, posinf=_BIG_COST, neginf=-_BIG_COST).contiguous()
    solver = _triton_solver() if rows.is_cuda else None
    if solver is not None:
        col = solver.batch_linear_assignment(rows, validation="off")             # (B, P)
    else:
        from scipy.optimize import linear_sum_assignment
        host = rows.cpu().numpy()
        col = torch.empty(B, P, dtype=torch.long)
        for b in range(B):
            r, c = linear_sum_assignment(host[b])
            col[b, torch.as_tensor(r)] = torch.as_tensor(c)
        col = col.to(cost.device)
    people = torch.arange(P, device=cost.device).expand(B, P)
    ok = available & (col >= 0) & (col < Q)          # never scatter an out-of-range column
    index = torch.where(ok, col, Q)                                              # Q = discard column
    out.scatter_(1, index, torch.where(ok, people, -1))
    return out[:, :Q]


# --------------------------------------------------------------------------- #
# Slot rules                                                                   #
# --------------------------------------------------------------------------- #
def _gather_rows(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """x (B, S, ...) gathered at index (B, M) -> (B, M, ...)."""
    shape = index.shape + x.shape[2:]
    return torch.gather(x, 1, index.reshape(*index.shape, *([1] * (x.dim() - 2))).expand(shape))


def _scatter_rows(dest: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    """dest (B, S+1, ...)[b, index[b, m]] = src[b, m] (index S = discard)."""
    shape = index.shape + dest.shape[2:]
    dest.scatter_(1, index.reshape(*index.shape, *([1] * (dest.dim() - 2))).expand(shape), src.to(dest.dtype))


def _root(kpts: torch.Tensor) -> torch.Tensor:
    return kpts[..., ROOT_SLICE, :].mean(dim=-2)


@torch.no_grad()
def train_step(manager, state, num_det: int, cls, kpts, feat, targets: dict, t: int,
               codes=None, ema=None, app=None):
    """Device-only equivalent of :meth:`TrackManager.train_step`.

    Returns ({"target": (B, Qm) long, "excluded": (B, Qm) bool}, new_state).
    """
    from .supervision import Supervision
    B, Qm = cls.shape[:2]
    K = Qm - num_det
    device = cls.device
    sup = targets.get("_supervision")
    if sup is None:
        cache = manager.__dict__.setdefault("_range_on", {})
        if device not in cache:
            cache[device] = manager.position_range.to(device)   # once per device
        sup = targets["_supervision"] = Supervision(targets, cache[device])
    valid = targets["valid"][:, t].bool()                                       # (B, P)
    ids = targets["instance_ids"][:, t].long()
    exists = (targets["exists"][:, t].bool() | valid) & (ids >= 0)
    P = valid.shape[1]
    slot_valid, slot_ids = state.valid, state.ids.long()                        # (B, K)

    # Track claims: a valid slot whose identity is present is assigned that person directly.
    same = slot_valid[:, :, None] & (slot_ids[:, :, None] == ids[:, None, :]) & (ids >= 0)[:, None, :]
    claim = same & valid[:, None, :]                                            # (B, K, P)
    tracked = claim.any(-1)
    person_index = torch.arange(P, device=device)
    track_g = (claim.long() * (person_index + 1)).argmax(-1)                    # last match wins
    stale = slot_valid & ~tracked & (same & exists[:, None, :]).any(-1)         # unobserved: ignored
    claimed = (claim & (person_index == track_g[..., None])[..., :] & tracked[..., None]).any(1)

    # Detection queries: Hungarian against the present, matchable people no track claimed.
    free = valid & ~claimed & sup.matchable[:, t]
    cost = batch_match_cost(sup, t, cls[:, :num_det], kpts[:, :num_det], manager.costs)
    det_target = solve_assignment(cost, free)                                   # (B, num_det)
    bad = (~torch.isfinite(cost) & free[:, None, :]).sum()
    manager.nonfinite_cost = bad if getattr(manager, "nonfinite_cost", None) is None else manager.nonfinite_cost + bad

    target = torch.cat([det_target, torch.where(tracked, track_g, -1)], dim=1)
    excluded = torch.cat([torch.zeros_like(det_target, dtype=torch.bool), ~slot_valid | stale], dim=1)
    assignment = {"target": target, "excluded": excluded}

    # Next frame: carried slots (k order), then births (query order), drops, cut at K slots.
    matched = det_target >= 0
    birth_id = torch.gather(ids, 1, det_target.clamp(min=0))
    birth = matched & (birth_id >= 0)
    carry = tracked | stale
    keep = torch.cat([carry, birth], dim=1)                                     # (B, K + num_det)
    if manager.drop_prob > 0:
        keep = keep & (torch.rand(keep.shape, device=device) >= manager.drop_prob)
    pos = keep.long().cumsum(1) - 1
    placed = keep & (pos < K)
    dest = torch.where(placed, pos, K)                                          # K = discard
    n_placed = placed.sum(1)

    new = manager._new_state(state, K, feat)
    reid = bool(manager.reid_dims)
    has_app = bool(new.app_ema.shape[-1])
    is_stale = torch.cat([stale, torch.zeros_like(birth)], dim=1)               # frozen candidates
    slot_q = torch.arange(num_det, Qm, device=device).expand(B, K)
    source_q = torch.cat([slot_q, torch.arange(num_det, device=device).expand(B, num_det)], dim=1)

    embed = torch.where(is_stale[..., None],
                        torch.cat([state.embed, state.embed.new_zeros(B, num_det, state.embed.shape[-1])], 1),
                        _gather_rows(feat, source_q).to(state.embed.dtype))
    ref = _root(_gather_rows(kpts, source_q)).float()
    if manager.ref_noise > 0:
        ref = (ref + torch.randn_like(ref) * manager.ref_noise).clamp(0.0, 1.0)
    ref = torch.where(is_stale[..., None], torch.cat([state.ref, state.ref.new_zeros(B, num_det, 3)], 1).float(),
                      ref)
    cand_ids = torch.cat([slot_ids, birth_id], dim=1)
    inactive = torch.cat([torch.where(stale, state.inactive + 1, 0),
                          torch.zeros(B, num_det, dtype=state.inactive.dtype, device=device)], dim=1)

    def write(name, values):
        buf = torch.zeros((B, K + 1) + getattr(new, name).shape[2:], dtype=getattr(new, name).dtype,
                          device=device)
        if name == "ids":
            buf.fill_(-1)
        _scatter_rows(buf, dest, values)
        getattr(new, name).copy_(buf[:, :K])

    write("embed", embed)
    write("ref", ref)
    write("valid", placed)
    write("ids", cand_ids)
    write("inactive", inactive)
    if reid:
        slot_code = torch.where(stale[..., None], state.id_ema, ema.float())
        write("id_ema", torch.cat([slot_code, codes.float()[:, :num_det]], dim=1))
    if has_app:
        if app is not None:
            tracked_ok = app[1] & tracked
            slot_ok = torch.where(stale, state.app_valid, tracked_ok)
            slot_emb = torch.where(stale[..., None], state.app_ema, app[0].to(state.app_ema.dtype))
        else:
            slot_ok = stale & state.app_valid
            slot_emb = state.app_ema
        slot_emb = torch.where(slot_ok[..., None], slot_emb, 0.0)
        write("app_valid", torch.cat([slot_ok, torch.zeros_like(birth)], dim=1))
        write("app_ema", torch.cat([slot_emb, new.app_ema.new_zeros(B, num_det, new.app_ema.shape[-1])], 1))

    # Optional false track: the highest-scoring unmatched detection query, which must learn to die.
    if manager.fp_prob > 0:
        event = (n_placed < K) & (torch.rand(B, device=device) < manager.fp_prob) & (matched.sum(1) < num_det)
        scores = cls[:, :num_det, 0].float().masked_fill(matched, float("-inf"))
        fq = scores.argmax(1)                                                   # (B,)
        slot = torch.where(event, n_placed, K)[:, None]                         # (B, 1)
        fq1 = fq[:, None]
        f_ref = _root(_gather_rows(kpts, fq1)).float()
        if manager.ref_noise > 0:
            f_ref = (f_ref + torch.randn_like(f_ref) * manager.ref_noise).clamp(0.0, 1.0)
        values = {"embed": _gather_rows(feat, fq1), "ref": f_ref, "valid": event[:, None],
                  "ids": torch.full_like(fq1, FALSE_TRACK_ID), "inactive": torch.zeros_like(fq1)}
        if reid:
            values["id_ema"] = _gather_rows(codes.float(), fq1)
        if has_app:
            values["app_valid"] = torch.zeros_like(event[:, None])
            values["app_ema"] = new.app_ema.new_zeros(B, 1, new.app_ema.shape[-1])
        for name, value in values.items():
            current = getattr(new, name)
            buf = torch.cat([current, current[:, :1]], dim=1).clone()
            _scatter_rows(buf, slot, value)
            current.copy_(buf[:, :K])
    return assignment, new
