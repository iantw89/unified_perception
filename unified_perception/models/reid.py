"""Re-identification embeddings for track queries.

Identity code per query = fusion of

* a learned projection of the decoder's query embedding.  The query must
  regress the person's 3-D keypoints every frame, so stable body proportions
  (height, limb lengths) and motion are implicit in it; a per-track EMA of the
  codes averages out per-frame detection noise.  Raw measurements are never
  used directly;
* an appearance code: backbone features sampled at the person's projected
  keypoints in every valid view, visibility-averaged across views and joints
  (sampled without gradient into the backbone; only the projection learns).

Training: supervised contrastive (InfoNCE) loss per batch lane between the
current window's identity codes and a detached bank of the lane's earlier
codes -- recent ones plus the first ones ever seen per identity, so positives
span long gaps within a streamed piece.

Inference (see ``TrackManager.infer_step``): ended tracks enter a per-lane
gallery with their EMA code; a newly born track is provisional for
``reid_wait_frames`` active frames, then takes over the best gallery identity
whose cosine similarity is >= ``reid_threshold``.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; multiplies the gradient by ``scale`` on the way back."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0 or not x.requires_grad:
        return x
    return x.detach() if scale == 0.0 else _ScaleGrad.apply(x, scale)


class IdentityHead(nn.Module):
    """Fuses the query projection with optional backbone and pretrained-network appearance codes."""

    def __init__(self, embed_dims: int, reid_dims: int, appearance: bool, external_dims: int = 0,
                 query_grad_scale: float = 1.0):
        super().__init__()
        # The decoder query is the only input shared with the pose / detection losses (appearance
        # and external codes are sampled without gradient): scaling the gradient here bounds how
        # much the contrastive loss steers the shared decoder, while the identity head itself
        # still learns at the full loss weight.
        self.query_grad_scale = float(query_grad_scale)
        self.query_proj = _mlp(embed_dims, embed_dims, reid_dims)
        self.appearance, self.external_dims = appearance, external_dims
        parts = 1
        if appearance:
            self.app_proj = _mlp(embed_dims, embed_dims, reid_dims)
            parts += 1
        if external_dims:
            self.ext_proj = _mlp(external_dims, embed_dims, reid_dims)
            # Used where no crop embedding exists yet (detections, brand-new tracks).
            self.ext_missing = nn.Parameter(torch.zeros(reid_dims))
            parts += 1
        if parts > 1:
            self.fuse = _mlp(parts * reid_dims, reid_dims, reid_dims)

    def forward(self, query: torch.Tensor, appearance: torch.Tensor | None = None,
                external: torch.Tensor | None = None, external_valid: torch.Tensor | None = None) -> torch.Tensor:
        parts = [self.query_proj(scale_grad(query, self.query_grad_scale))]
        if self.appearance:
            parts.append(self.app_proj(appearance))
        if self.external_dims:
            ext = self.ext_proj(external.to(query.dtype))
            missing = self.ext_missing.to(ext.dtype).expand_as(ext)
            parts.append(torch.where(external_valid[..., None], ext, missing))
        code = self.fuse(torch.cat(parts, dim=-1)) if len(parts) > 1 else parts[0]
        return F.normalize(code.float(), dim=-1)


@torch.no_grad()
def sample_appearance(
    tokens: torch.Tensor,          # (B, N*sumL, C) per-view multi-level tokens of one frame
    spatial_shapes: list,          # [(H_l, W_l), ...]
    keypoints_mm: torch.Tensor,    # (B, Q, J, 3) world coordinates
    world2img: torch.Tensor,       # (B, N, 4, 4)
    image_hw: tuple[int, int],
    camera_valid: torch.Tensor | None = None,  # (B, N)
) -> torch.Tensor:
    """Visibility-weighted mean of features at the projected keypoints, (B, Q, C)."""
    B, Q, J, _ = keypoints_mm.shape
    N = world2img.shape[1]
    C = tokens.shape[-1]
    H, W = image_hw
    points = torch.cat([keypoints_mm, keypoints_mm.new_ones(B, Q, J, 1)], dim=-1).float()
    cam = torch.einsum("bnij,bqkj->bnqki", world2img.float(), points)        # (B, N, Q, J, 4)
    depth = cam[..., 2]
    uv = cam[..., :2] / depth.clamp(min=1e-3)[..., None]
    valid = (depth > 1e-3) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    if camera_valid is not None:
        valid &= camera_valid[:, :, None, None]
    grid = torch.stack([uv[..., 0] / W * 2 - 1, uv[..., 1] / H * 2 - 1], dim=-1)
    grid = grid.reshape(B * N, Q * J, 1, 2).to(tokens.dtype)
    per_view = tokens.reshape(B, N, -1, C)
    total, start = 0, 0
    for h, w in spatial_shapes:
        level = per_view[:, :, start:start + h * w].reshape(B * N, h, w, C).permute(0, 3, 1, 2)
        start += h * w
        total = total + F.grid_sample(level, grid, align_corners=False)      # (B*N, C, Q*J, 1)
    feats = (total / len(spatial_shapes)).reshape(B, N, C, Q, J).permute(0, 1, 3, 4, 2)
    weight = valid.to(feats.dtype)[..., None]
    return (feats * weight).sum(dim=(1, 3)) / weight.sum(dim=(1, 3)).clamp(min=1.0)


class ReIDLoss:
    """Supervised contrastive loss against a per-lane bank of earlier codes."""

    def __init__(self, temperature: float, recent: int, anchors: int, stride: int):
        self.temperature, self.recent, self.anchors, self.stride = temperature, recent, anchors, stride
        self.banks: dict[int, dict[int, dict]] = {}
        self.clock: dict[int, int] = {}

    def reset(self):
        self.banks, self.clock = {}, {}

    def _bank(self, lane: int):
        entries = [(e, i) for i, slot in self.banks.get(lane, {}).items()
                   for e in list(slot["anchors"]) + list(slot["recent"])]
        if not entries:
            return None, None
        codes, ids = zip(*entries)
        return torch.stack(codes), torch.tensor(ids)

    def __call__(self, samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                 stream_reset: torch.Tensor | None, window_len: int):
        """``samples[b]`` = (codes (n, D) with grad, ids (n,), frame index (n,)) for lane b."""
        total, count = None, 0
        for lane, (codes, ids, frames) in enumerate(samples):
            if stream_reset is None or bool(stream_reset[lane]) or lane not in self.banks:
                self.banks[lane], self.clock[lane] = {}, 0
            if len(codes) == 0:
                self.clock[lane] += window_len
                continue
            bank_codes, bank_ids = self._bank(lane)
            keys = codes if bank_codes is None else torch.cat([codes, bank_codes.to(codes)], dim=0)
            key_ids = ids.cpu() if bank_ids is None else torch.cat([ids.cpu(), bank_ids])
            key_frames = torch.cat([frames.cpu(), torch.full((len(keys) - len(codes),), -1)])
            sim = codes @ keys.T / self.temperature                         # (n, n + m)
            n = len(codes)
            self_mask = torch.zeros_like(sim, dtype=torch.bool)
            self_mask[torch.arange(n), torch.arange(n)] = True
            same = (ids.cpu()[:, None] == key_ids[None, :]).to(sim.device)
            # Same identity in the same frame is the same observation (frame + EMA code).
            same_frame = (frames.cpu()[:, None] == key_frames[None, :]).to(sim.device)
            positive = same & ~self_mask & ~same_frame
            usable = positive.any(1) & (~same).any(1)
            if usable.any():
                logits = sim.masked_fill(self_mask | (same & same_frame), float("-inf"))
                log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
                per_anchor = -(log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp(min=1))
                loss = per_anchor[usable].sum()
                total = loss if total is None else total + loss
                count += int(usable.sum())
            self._insert(lane, codes.detach(), ids.cpu(), frames.cpu())
            self.clock[lane] += window_len
        if total is None:
            return None, 0
        return total / count, count

    def _insert(self, lane: int, codes, ids, frames):
        bank = self.banks[lane]
        base = self.clock[lane]
        for code, identity, frame in zip(codes, ids.tolist(), frames.tolist()):
            slot = bank.setdefault(identity, {"anchors": [], "recent": deque(maxlen=self.recent), "last": None})
            now = base + frame
            if slot["last"] is not None and now - slot["last"] < self.stride:
                continue
            slot["last"] = now
            (slot["anchors"] if len(slot["anchors"]) < self.anchors else slot["recent"]).append(code)


class DeviceReIDLoss:
    """:class:`ReIDLoss` with the bank held in device tensors (``model.gpu_assignment``).

    Same loss and bank rules -- per identity the first ``anchors`` codes ever inserted plus the
    ``recent`` latest, at most one insertion per ``stride`` frames, entries taken in the reference
    order (each frame's query codes in frame order, then the slots' EMA codes) -- with every step a
    batched tensor operation, so nothing waits for the device.  Identities map to bank rows
    through a per-lane table of ``rows`` entries; identities beyond it are not banked (the gym
    golden set has 17 people, other sources at most ``max_people``).
    """

    def __init__(self, temperature: float, recent: int, anchors: int, stride: int, rows: int = 64):
        self.temperature, self.recent, self.anchors, self.stride, self.rows = (
            temperature, recent, anchors, stride, rows)
        self.state = None

    def reset(self):
        self.state = None

    _NEVER = -(1 << 40)

    def _ensure(self, B: int, D: int, device):
        s = self.state
        if s is not None and s["codes"].shape[0] == B and s["codes"].shape[-1] == D and s["codes"].device == device:
            return s
        S, E = self.rows, self.anchors + self.recent
        self.state = s = {                                             # row S is a write sink
            "codes": torch.zeros(B, S + 1, E, D, device=device),
            "valid": torch.zeros(B, S + 1, E, dtype=torch.bool, device=device),
            "ids": torch.full((B, S + 1), -1, dtype=torch.long, device=device),
            "n_anchor": torch.zeros(B, S + 1, dtype=torch.long, device=device),
            "ptr": torch.zeros(B, S + 1, dtype=torch.long, device=device),
            "last": torch.full((B, S + 1), self._NEVER, dtype=torch.long, device=device),
            "clock": torch.zeros(B, dtype=torch.long, device=device),
        }
        return s

    def _clear(self, s, lanes: torch.Tensor):
        row = lanes.view(-1, 1)
        s["valid"].masked_fill_(lanes.view(-1, 1, 1), False)
        s["ids"].masked_fill_(row, -1)
        s["n_anchor"].masked_fill_(row, 0)
        s["ptr"].masked_fill_(row, 0)
        s["last"].masked_fill_(row, self._NEVER)
        s["clock"].masked_fill_(lanes, 0)

    def __call__(self, codes: torch.Tensor, labels: torch.Tensor, frames: torch.Tensor, mask: torch.Tensor,
                 groups: list[torch.Tensor], stream_reset: torch.Tensor | None, window_len: int):
        """codes (B, n, D) with grad; labels / frames (B, n) long; mask (B, n) bool.

        ``groups``: (n,) bool masks of entry blocks inserted one after the other (each block has
        at most one entry per identity).  Returns (loss, number of usable anchors), both tensors.
        """
        B, n, D = codes.shape
        device = codes.device
        s = self._ensure(B, D, device)
        self._clear(s, torch.ones(B, dtype=torch.bool, device=device) if stream_reset is None
                    else stream_reset.to(device=device, dtype=torch.bool))

        S, E = self.rows, self.anchors + self.recent
        bank = s["codes"][:, :S].reshape(B, S * E, D)
        bank_valid = s["valid"][:, :S].reshape(B, S * E)
        bank_ids = s["ids"][:, :S, None].expand(B, S, E).reshape(B, S * E)
        keys = torch.cat([codes, bank.to(codes.dtype)], dim=1)                      # (B, n + SE, D)
        key_valid = torch.cat([mask, bank_valid], dim=1)
        key_ids = torch.cat([labels, bank_ids], dim=1)
        key_frames = torch.cat([frames, torch.full_like(bank_ids, -1)], dim=1)
        sim = codes @ keys.transpose(1, 2) / self.temperature                     # (B, n, n + SE)
        eye = torch.zeros(n, n + S * E, dtype=torch.bool, device=device)
        eye[:, :n] = torch.eye(n, dtype=torch.bool, device=device)
        same = (labels[:, :, None] == key_ids[:, None, :]) & key_valid[:, None, :]
        same_frame = frames[:, :, None] == key_frames[:, None, :]
        positive = same & ~eye & ~same_frame
        usable = mask & positive.any(-1) & (key_valid[:, None, :] & ~same).any(-1)
        logits = sim.masked_fill(eye | (same & same_frame) | ~key_valid[:, None, :], float("-inf"))
        logits = torch.where(usable[..., None], logits, 0.0)       # unusable rows: finite, unused
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        per_anchor = -(log_prob.masked_fill(~positive, 0).sum(-1) / positive.sum(-1).clamp(min=1))
        count = usable.sum()
        loss = torch.where(usable, per_anchor, 0.0).sum() / count.clamp(min=1)

        with torch.no_grad():
            for block in groups:
                self._insert(s, codes.detach().float(), labels, frames, mask & block)
            s["clock"] += window_len
        return loss, count

    def _insert(self, s, codes, labels, frames, mask):
        B, n, D = codes.shape
        S, A, R = self.rows, self.anchors, self.recent
        table = s["ids"][:, :S]                                                     # (B, S)
        hit = (table[:, None, :] == labels[:, :, None]) & (table[:, None, :] >= 0)  # (B, n, S)
        found = hit.any(-1)
        row = hit.long().argmax(-1)
        new = mask & ~found & (labels >= 0)
        free = table < 0
        free_rows = torch.sort((~free).long(), dim=1, stable=True).indices          # free rows first
        rank = new.long().cumsum(1) - 1
        alloc = torch.gather(free_rows, 1, rank.clamp(min=0, max=S - 1))
        alloc = torch.where(rank < free.sum(1, keepdim=True), alloc, S)
        row = torch.where(mask, torch.where(found, row, alloc), S)
        s["ids"].scatter_(1, row, torch.where(row < S, labels, -1))
        now = s["clock"][:, None] + frames
        last = torch.gather(s["last"], 1, row)
        insert = mask & (row < S) & (now - last >= self.stride)
        row = torch.where(insert, row, S)
        n_anchor = torch.gather(s["n_anchor"], 1, row)
        ptr = torch.gather(s["ptr"], 1, row)
        to_anchor = n_anchor < A
        slot = torch.where(to_anchor, n_anchor, A + ptr)
        flat = row * (A + R) + slot
        s["codes"].view(B, -1, D).scatter_(1, flat[..., None].expand(B, n, D), codes)
        s["valid"].view(B, -1).scatter_(1, flat, insert)
        s["n_anchor"].scatter_(1, row, torch.where(to_anchor, n_anchor + 1, n_anchor))
        s["ptr"].scatter_(1, row, torch.where(to_anchor, ptr, (ptr + 1) % max(R, 1)))
        s["last"].scatter_(1, row, torch.where(insert, now, last))
        s["ids"][:, S] = -1                                                         # keep the sink empty
        s["valid"][:, S] = False
        s["n_anchor"][:, S] = 0
        s["ptr"][:, S] = 0
        s["last"][:, S] = self._NEVER


# --------------------------------------------------------------------------- #
# External pretrained person re-ID network on per-view crops                   #
# --------------------------------------------------------------------------- #
def person_crop_boxes(keypoints_mm: torch.Tensor, world2img: torch.Tensor, image_hw: tuple[int, int],
                      camera_valid: torch.Tensor | None = None, min_joints: int = 3,
                      aspect: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-view person boxes (B, Q, N, 4) xyxy pixels and weights (B, Q, N), 0 = unusable.

    Boxes enclose the projected body joints (no head joint, so the top is padded
    more), padded and grown to the re-ID crop aspect (height / width).  The
    weight is the number of joints that project inside the image, so views that
    see more of the person count more.
    """
    B, Q, J, _ = keypoints_mm.shape
    H, W = image_hw
    points = torch.cat([keypoints_mm, keypoints_mm.new_ones(B, Q, J, 1)], dim=-1).float()
    cam = torch.einsum("bnij,bqkj->bqnki", world2img.float(), points)        # (B, Q, N, J, 4)
    depth = cam[..., 2]
    uv = cam[..., :2] / depth.clamp(min=1e-3)[..., None]
    inside = (depth > 1e-3) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    if camera_valid is not None:
        inside &= camera_valid[:, None, :, None]
    big = torch.finfo(uv.dtype).max
    x0 = torch.where(inside, uv[..., 0], big).amin(-1)
    y0 = torch.where(inside, uv[..., 1], big).amin(-1)
    x1 = torch.where(inside, uv[..., 0], -big).amax(-1)
    y1 = torch.where(inside, uv[..., 1], -big).amax(-1)
    h, w = (y1 - y0).clamp(min=1.0), (x1 - x0).clamp(min=1.0)
    y0, y1 = y0 - 0.30 * h, y1 + 0.08 * h          # head above the shoulders, feet below the ankles
    x0, x1 = x0 - 0.20 * w, x1 + 0.20 * w
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = y1 - y0, x1 - x0
    h = torch.maximum(h, w * aspect)
    w = h / aspect
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    count = inside.sum(-1)
    weight = torch.where(count >= min_joints, count.float(), torch.zeros_like(count, dtype=torch.float))
    # Views with no joint inside have infinite extents; give them a finite placeholder box so a
    # fixed crop budget (ExternalReID.max_crops) that samples them still gets finite pixels.
    placeholder = torch.tensor([0.0, 0.0, 1.0, 1.0], device=boxes.device, dtype=boxes.dtype)
    boxes = torch.where((weight > 0)[..., None] & torch.isfinite(boxes).all(-1, keepdim=True), boxes, placeholder)
    return boxes, weight


class ExternalReID(nn.Module):
    """Frozen TorchScript re-ID network (tools/export_fastreid.py) applied to per-view crops."""

    def __init__(self, path: str, crop_hw: tuple[int, int] = (384, 128), chunk: int = 256,
                 max_crops: int = 0):
        super().__init__()
        self.net = torch.jit.load(path, map_location="cpu").eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.crop_hw, self.chunk = tuple(crop_hw), chunk
        # > 0: a fixed number of crops per call (the highest-weight usable views), so the call has
        # static shapes and never synchronizes; 0: exactly the usable views (one nonzero sync).
        self.max_crops = max_crops
        self.device = torch.device("cpu")
        with torch.no_grad():
            self.dims = int(self.net(torch.zeros(1, 3, *self.crop_hw)).shape[-1])

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()  # BatchNorm statistics stay frozen
        return self

    @torch.no_grad()
    def forward(self, images_255: torch.Tensor, keypoints_mm: torch.Tensor, world2img: torch.Tensor,
                camera_valid: torch.Tensor | None = None,
                person_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Visibility-weighted mean embedding per person (B, Q, E) and availability (B, Q).

        images_255: (B, N, 3, H, W) RGB in [0, 255]; keypoints_mm: (B, Q, J, 3).
        """
        from torchvision.ops import roi_align
        if self.device != images_255.device:
            # fp16 on GPU: embeddings agree with fp32 to cosine >= 0.9996, ~40% faster.
            self.net.to(images_255.device, torch.float16 if images_255.is_cuda else torch.float32)
            self.device = images_255.device
        dtype = torch.float16 if images_255.is_cuda else torch.float32
        B, N, _, H, W = images_255.shape
        Q = keypoints_mm.shape[1]
        boxes, weight = person_crop_boxes(keypoints_mm, world2img, (H, W), camera_valid,
                                          aspect=self.crop_hw[0] / self.crop_hw[1])
        if person_valid is not None:
            weight = weight * person_valid[..., None]
        out = images_255.new_zeros(B, Q, self.dims, dtype=torch.float32)
        usable = weight > 0
        if self.max_crops > 0:
            top_w, top = torch.topk(weight.reshape(-1), min(self.max_crops, weight.numel()))
            b, rem = top // (Q * N), top % (Q * N)
            q, n = rem // N, rem % N
            keep = top_w > 0
        else:
            if not usable.any():
                return out, usable.any(-1)
            b, q, n = torch.nonzero(usable, as_tuple=True)
            keep = None
        rois = torch.cat([(b * N + n).float()[:, None], boxes[b, q, n]], dim=1)
        crops = roi_align(images_255.reshape(B * N, 3, H, W).float(), rois, self.crop_hw,
                          spatial_scale=1.0, sampling_ratio=2, aligned=True)
        # Fixed precision: the surrounding training autocast (bf16) must not recast the frozen net.
        with torch.autocast(device_type=images_255.device.type, enabled=False):
            embeds = torch.cat([self.net(crops[i:i + self.chunk].to(dtype))
                                for i in range(0, len(crops), self.chunk)])
        w = weight[b, q, n][:, None]
        # Dropped / unused crops (w = 0) contribute nothing, even if their embedding is not finite.
        out.index_put_((b, q), torch.where(w > 0, embeds.float() * w, 0.0), accumulate=True)
        if keep is not None:   # views beyond the budget do not count towards the normalization
            kept = torch.zeros_like(weight).index_put_((b, q, n), keep.to(weight.dtype))
            weight = weight * kept
            usable = weight > 0
        total = weight.sum(-1)
        out = F.normalize(out / total.clamp(min=1e-6)[..., None], dim=-1)
        return out, usable.any(-1)
