"""StreamPETR / Detr3D sparse head for multi-person 3D keypoints.

This implements the Detr3D-style cross-attention from StreamPETR / RepDETR3D:
rather than adding dense 3D position embeddings to all image tokens (PETR),
the decoder projects each query's 3D reference point into the 2D image plane
of every camera and bilinearly samples features there.  No
``CameraPositionEncoder3D`` is needed; the projection geometry is handled
inside ``Detr3DCrossAttn``.

Domain-specific adaptations vs the original StreamPETR:
1. **No ego-motion** -- static capture rig, memory stored in world frame.
2. **3D keypoints** instead of 3D boxes (each query regresses J joints).
3. **Single class** -- one sigmoid logit per query.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..constants import ROOT_JOINTS
from .misc import (
    MLN,
    inverse_sigmoid,
    memory_refresh,
    pos2posemb1d,
    pos2posemb3d,
    topk_gather,
)
from .reid import IdentityHead, sample_appearance
from .tracking import TrackManager, TrackState, track_key_mask
from .transformer import TransformerDecoder


class SparseHead(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_joints: int,
        num_query: int,
        num_propagated: int,
        memory_len: int,
        topk_proposals: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_cams: int,
        num_levels: int,
        num_pts: int,
        position_range: List[float],
        num_groups: int = 1,
        with_dn: bool = True,
        scalar: int = 10,
        noise_scale: float = 0.2,
        split: float = 0.75,
        time_step: float = 1.0,
        use_mmcv_msda: bool = False,
        max_tracks: int = 0,
        track_manager: Optional[TrackManager] = None,
        reid_dims: int = 0,
        reid_appearance: bool = True,
        reid_ema: float = 0.9,
        external_reid: Optional[nn.Module] = None,
        external_stride: int = 4,
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
    ):
        super().__init__()
        if (max_tracks > 0) != (track_manager is not None):
            raise ValueError("track queries need both max_tracks > 0 and a TrackManager")
        if max_tracks > 0:
            # Identity-carrying track slots replace the top-k propagated queries.
            num_propagated = 0
        self.max_tracks     = max_tracks
        self.track_manager  = track_manager
        self.num_heads      = num_heads
        self.embed_dims     = embed_dims
        self.num_joints     = num_joints
        self.num_query      = num_query
        self.num_propagated = num_propagated
        self.memory_len     = memory_len
        self.topk_proposals = topk_proposals
        self.num_layers     = num_layers
        self.with_dn        = with_dn
        self.scalar         = scalar
        self.dn_noise_scale = noise_scale
        self.dn_split       = split
        self.time_step      = float(time_step)

        pc_range = torch.tensor(position_range, dtype=torch.float32)
        self.register_buffer("pc_range", pc_range)

        self.decoder = TransformerDecoder(
            dim=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_cams=num_cams,
            num_levels=num_levels,
            num_pts=num_pts,
            dropout=dropout,
            pc_range=pc_range,
            num_groups=num_groups,
            use_mmcv_msda=use_mmcv_msda,
        )

        # Query positional embedding from a 3D reference point.
        self.query_embedding = nn.Sequential(
            nn.Linear(embed_dims * 3 // 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        # Time embedding for the streaming memory age.
        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )
        # Spatial alignment: inject per-view calibration into the image tokens.
        # 14 = fx, fy, plus the 3x4 world->camera extrinsic.
        self.spatial_alignment = MLN(14, embed_dims)

        cls_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
        )
        reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_joints * 3),
        )
        self.cls_branches = nn.ModuleList([cls_branch for _ in range(num_layers)])
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_layers)])
        self._shared_branches = (all(m is cls_branch for m in self.cls_branches)
                                 and all(m is reg_branch for m in self.reg_branches))

        # Learnable query anchors (normalised [0, 1]) and propagated pseudo anchors.
        self.reference_points = nn.Embedding(num_query, 3)
        nn.init.uniform_(self.reference_points.weight, 0, 1)
        if num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(num_propagated, 3)
            nn.init.uniform_(self.pseudo_reference_points.weight, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        if reid_dims > 0 and max_tracks == 0:
            raise ValueError("re-identification requires track queries")
        self.reid_dims, self.reid_ema = reid_dims, float(reid_ema)
        self.reid_appearance = reid_appearance
        # Frozen and loaded from its own file: kept out of the module tree so it is never
        # saved in checkpoints nor seen by the optimizer (moved to the input device on use).
        self.__dict__["external_reid"] = external_reid if reid_dims > 0 else None
        self.external_stride = max(1, int(external_stride))
        self.external_dims = self.external_reid.dims if self.external_reid is not None else 0
        # Undo the training normalization for the crop network (it expects RGB 0..255).
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean or [0.0, 0.0, 0.0]).view(1, 1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std or [255.0, 255.0, 255.0]).view(1, 1, 3, 1, 1),
                             persistent=False)
        self._frames_seen = 0
        if reid_dims > 0:
            self.identity_head = IdentityHead(embed_dims, reid_dims, reid_appearance, self.external_dims)
        if max_tracks > 0:
            # Marks track slots apart from detection queries in self-attention.
            self.track_embedding = nn.Parameter(torch.zeros(embed_dims))
            nn.init.trunc_normal_(self.track_embedding, std=0.02)

        self.track_output: Optional[dict] = None
        self.reset_memory()

    # ------------------------------------------------------------------ #
    # Streaming memory                                                    #
    # ------------------------------------------------------------------ #
    def reset_memory(self):
        self.memory_embedding: Optional[torch.Tensor] = None
        self.memory_reference_point: Optional[torch.Tensor] = None
        self.memory_timestamp: Optional[torch.Tensor] = None
        self.track_state: Optional[TrackState] = None
        self._frames_seen = 0

    def get_stream_state(self) -> dict:
        """Snapshot of the cross-call temporal state (e.g. around an evaluation pass)."""
        def clone(x):
            return None if x is None else x.clone()
        return {"memory_embedding": clone(self.memory_embedding),
                "memory_reference_point": clone(self.memory_reference_point),
                "memory_timestamp": clone(self.memory_timestamp),
                "track_state": None if self.track_state is None else self.track_state.clone()}

    def set_stream_state(self, state: dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def pre_update_memory(self, B, prev_exists, device, dtype):
        x = prev_exists.view(B)
        if self.memory_embedding is None:
            self.memory_embedding = torch.zeros(B, self.memory_len, self.embed_dims, device=device, dtype=dtype)
            self.memory_reference_point = torch.zeros(B, self.memory_len, 3, device=device, dtype=dtype)
            self.memory_timestamp = torch.zeros(B, self.memory_len, 1, device=device, dtype=dtype)
        else:
            self.memory_timestamp = self.memory_timestamp + self.time_step
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, : self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, : self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, : self.memory_len], x)

        # First-frame anchors come from the (non-learnable) pseudo reference points.
        if self.num_propagated > 0:
            pseudo = self.pseudo_reference_points.weight.to(device=device, dtype=dtype)
            self.memory_reference_point[:, : self.num_propagated] = (
                self.memory_reference_point[:, : self.num_propagated]
                + (1 - x).view(B, 1, 1) * pseudo
            )

    def post_update_memory(self, cls, kpts, dec_feat):
        """Prepend the current frame''s top-k confident queries to the memory."""
        a, b = ROOT_JOINTS
        last_kpts = kpts[-1]                                  # (B, Qm, J, 3)
        rec_ref = 0.5 * (last_kpts[:, :, a] + last_kpts[:, :, b])  # (B, Qm, 3)
        rec_memory = dec_feat[-1]                             # (B, Qm, C)
        rec_score = cls[-1].sigmoid()                         # (B, Qm, 1)

        k = min(self.topk_proposals, rec_score.shape[1])
        _, topk = torch.topk(rec_score, k, dim=1)            # (B, k, 1)
        rec_timestamp = topk_gather(torch.zeros_like(rec_score), topk)
        rec_ref = topk_gather(rec_ref, topk).detach()
        rec_memory = topk_gather(rec_memory, topk).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_reference_point = torch.cat([rec_ref, self.memory_reference_point], dim=1)
        self.memory_timestamp = self.memory_timestamp - self.time_step

    def integrate_temporal_queries(self, query_pos, tgt, reference_points):
        """Mix streaming memory queries into the current query set."""
        C = self.embed_dims
        temp_ref = self.memory_reference_point                       # (B, mem, 3)
        dtype = temp_ref.dtype
        temp_pos = self.query_embedding(pos2posemb3d(temp_ref, C // 2).to(dtype))
        temp_memory = self.memory_embedding

        query_pos = query_pos + self.time_embedding(
            pos2posemb1d(torch.zeros_like(reference_points[..., :1]), C).to(dtype)
        )
        temp_pos = temp_pos + self.time_embedding(pos2posemb1d(self.memory_timestamp, C).to(dtype))

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, : self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, : self.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points, temp_ref[:, : self.num_propagated]], dim=1
            )
            temp_memory = temp_memory[:, self.num_propagated :]
            temp_pos = temp_pos[:, self.num_propagated :]

        return tgt, query_pos, reference_points, temp_memory, temp_pos

    @torch.no_grad()
    def _slot_appearance(self, images, t, kpts_norm, world2img, camera_valid, state):
        """Crop re-id embeddings for live track slots every ``external_stride`` frames, as an
        updated per-slot EMA: returns (app (B, K, E), app_valid (B, K))."""
        if images is None:
            raise ValueError("crop re-identification needs the input images")
        if self._frames_seen % self.external_stride:
            return state.app_ema.clone(), state.app_valid.clone()
        kpts = kpts_norm[:, self.num_query:]
        kpts_mm = kpts * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
        frame = images[:, t].float()
        if images.dtype != torch.uint8:  # already normalized: back to RGB 0..255
            frame = frame * self.pixel_std + self.pixel_mean
        now, ok = self.external_reid(frame, kpts_mm, world2img, camera_valid, person_valid=state.valid)
        ok &= state.valid
        m = self.reid_ema
        blended = torch.nn.functional.normalize(m * state.app_ema + (1 - m) * now, dim=-1)
        update = torch.where(state.app_valid[..., None], blended, now)
        app = torch.where(ok[..., None], update, state.app_ema)
        return app, state.app_valid | ok

    def _identity_codes(self, query_feat, kpts_norm, tokens, spatial_shapes, world2img, image_hw,
                        camera_valid, state, app=None):
        """Per-query identity codes (B, Qm, D) and the slots' updated EMA codes (B, K, D)."""
        appearance = None
        if self.reid_appearance:
            kpts_mm = kpts_norm * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
            appearance = sample_appearance(tokens.detach(), spatial_shapes, kpts_mm, world2img,
                                           image_hw, camera_valid).to(query_feat.dtype)
        external = external_valid = None
        if self.external_dims:
            B, Qm = query_feat.shape[:2]
            external = query_feat.new_zeros(B, Qm, self.external_dims, dtype=torch.float32)
            external_valid = torch.zeros(B, Qm, dtype=torch.bool, device=query_feat.device)
            external[:, self.num_query:], external_valid[:, self.num_query:] = app
        codes = self.identity_head(query_feat, appearance, external, external_valid)
        slot_codes = codes[:, self.num_query:]
        m = self.reid_ema
        blended = torch.nn.functional.normalize(m * state.id_ema + (1 - m) * slot_codes, dim=-1)
        ema = torch.where(state.valid[..., None], blended, slot_codes)
        return codes, ema

    # ------------------------------------------------------------------ #
    # Query denoising (DN)                                                #
    # ------------------------------------------------------------------ #
    def _dn_pad_size(self, gt_valid) -> Tuple[int, int]:
        """single_pad = max_people (static padded dimension
        gt_valid.shape[-1] == P == cfg.data.max_people: GT tensors are always
        padded to this fixed size
        """
        single_pad = gt_valid.shape[-1]  # P = max_people (static)
        return single_pad, single_pad * self.scalar

    def _dn_attn_mask(self, pad_size, single_pad, device):
        num_query = self.num_query
        scalar = self.scalar
        size = pad_size + num_query
        attn_mask = torch.zeros(size, size, dtype=torch.bool, device=device)
        # Matching queries cannot see the reconstruct (DN) queries.
        attn_mask[pad_size:, :pad_size] = True
        # DN groups cannot see each other.
        for i in range(scalar):
            s = single_pad * i
            e = single_pad * (i + 1)
            if i == 0:
                attn_mask[s:e, e:pad_size] = True
            if i == scalar - 1:
                attn_mask[s:e, :s] = True
            else:
                attn_mask[s:e, e:pad_size] = True
                attn_mask[s:e, :s] = True

        # Extend over the temporal keys (propagated + memory) appended in
        # self-attention.  Columns beyond (pad + num_query) are the temporal
        # memory and are left visible to everyone except the rule below.
        query_size = pad_size + num_query + self.num_propagated
        key_size = pad_size + num_query + self.memory_len
        temporal = torch.zeros(query_size, key_size, dtype=torch.bool, device=device)
        temporal[:size, :size] = attn_mask
        temporal[pad_size:, :pad_size] = True
        return temporal

    def _dn_attn_mask_cached(self, pad_size, single_pad, device):
        key = (pad_size, single_pad, self.num_query, self.num_propagated, self.memory_len, str(device))
        cache = self.__dict__.setdefault("_dn_mask_cache", {})
        if key not in cache:
            cache[key] = self._dn_attn_mask(pad_size, single_pad, device)
        return cache[key]

    def prepare_dn_window(self, gt_kpts, gt_valid, single_pad, pad_size, gt_keypoint_valid=None):
        """Denoising queries of a whole window at once (no per-lane compaction, no host syncs).

        Person ``g`` of a frame occupies slot ``g`` of every one of the ``scalar`` DN groups
        (``single_pad`` = P slots per group); slots of absent people are padding.

        Args:
            gt_kpts (B, T, P, J, 3) normalized; gt_valid (B, T, P); gt_keypoint_valid (B, T, P, J).
        Returns:
            dn_ref (B, T, pad, 3) noised anchors and a dict of (B, T, pad, ...) targets.
        """
        B, T, P, J = gt_kpts.shape[:4]
        S = self.scalar
        root1, root2 = ROOT_JOINTS
        valid = gt_valid.bool()[:, :, None].expand(B, T, S, P)
        roots = 0.5 * (gt_kpts[..., root1, :] + gt_kpts[..., root2, :])                # (B, T, P, 3)
        noised = roots[:, :, None].expand(B, T, S, P, 3)
        is_person = valid
        if self.dn_noise_scale > 0:
            rand = torch.rand(B, T, S, P, 3, device=gt_kpts.device, dtype=roots.dtype) * 2 - 1
            noised = (noised + rand * self.dn_noise_scale).clamp(0, 1)
            is_person = valid & ~(rand.norm(2, dim=-1) > self.dn_split)  # too noisy -> background
        dn_ref = torch.where(valid[..., None], noised, torch.zeros_like(noised)).reshape(B, T, S * P, 3)
        targets = torch.where(valid[..., None, None], gt_kpts[:, :, None].expand(B, T, S, P, J, 3),
                              torch.zeros((), dtype=gt_kpts.dtype, device=gt_kpts.device))
        joint_valid = (torch.ones(B, T, P, J, dtype=torch.bool, device=gt_kpts.device)
                       if gt_keypoint_valid is None else gt_keypoint_valid.bool())
        joint_valid = joint_valid[:, :, None].expand(B, T, S, P, J) & valid[..., None]
        info = {
            "target_kpts": targets.reshape(B, T, S * P, J, 3),
            "target_kpt_valid": joint_valid.reshape(B, T, S * P, J),
            "valid": valid.reshape(B, T, S * P),
            "is_person": is_person.reshape(B, T, S * P),
        }
        return dn_ref, info

    def prepare_for_dn(self, B, ref, gt_kpts, gt_valid, single_pad, pad_size, gt_keypoint_valid=None):
        """Build noised reconstruct anchors + targets for query denoising.

        Args:
            ref: (num_query, 3) learnable anchors in [0, 1].
            gt_kpts: (B, P, J, 3) normalised GT keypoints.
            gt_valid: (B, P) bool.
        Returns:
            padded_ref: (B, pad + num_query, 3)
            attn_mask:  (query_size, key_size) bool
            dn_info:    dict of targets for the DN loss
        """
        device = ref.device
        J = self.num_joints
        root1, root2 = ROOT_JOINTS

        per_kpts: List[torch.Tensor] = []
        per_joint_valid: List[torch.Tensor] = []
        known_num: List[int] = []
        for bi in range(B):
            v = gt_valid[bi].bool()
            k = gt_kpts[bi][v]
            per_kpts.append(k)
            if gt_keypoint_valid is None:
                per_joint_valid.append(torch.ones(k.shape[:2], dtype=torch.bool, device=device))
            else:
                per_joint_valid.append(gt_keypoint_valid[bi][v].bool())
            known_num.append(int(k.shape[0]))
        total = sum(known_num)

        padded_ref = torch.cat([ref.new_zeros(pad_size, 3), ref], dim=0)
        padded_ref = padded_ref.unsqueeze(0).repeat(B, 1, 1)
        target_kpts = gt_kpts.new_zeros(B, pad_size, J, 3)
        valid = torch.zeros(B, pad_size, dtype=torch.bool, device=device)
        is_person = torch.zeros(B, pad_size, dtype=torch.bool, device=device)
        target_kpt_valid = torch.zeros(B, pad_size, J, dtype=torch.bool, device=device)

        if total > 0:
            all_kpts = torch.cat(per_kpts, dim=0)                 # (total, J, 3)
            roots = 0.5 * (all_kpts[:, root1] + all_kpts[:, root2])       # (total, 3)
            batch_idx = torch.cat(
                [torch.full((known_num[i],), i, dtype=torch.long, device=device) for i in range(B)]
            )
            all_joint_valid = torch.cat(per_joint_valid, dim=0)
            known_kpts = all_kpts.repeat(self.scalar, 1, 1)
            known_joint_valid = all_joint_valid.repeat(self.scalar, 1)
            known_roots = roots.repeat(self.scalar, 1)
            known_bid = batch_idx.repeat(self.scalar)
            known_labels = torch.zeros(known_roots.shape[0], device=device)  # 0 = person

            noised = known_roots.clone()
            if self.dn_noise_scale > 0:
                rand = torch.rand_like(noised) * 2 - 1
                noised = (noised + rand * self.dn_noise_scale).clamp(0, 1)
                flip = rand.norm(2, dim=1) > self.dn_split
                known_labels[flip] = 1.0  # too-noisy -> background

            map_known = torch.cat([torch.arange(n, device=device) for n in known_num])
            map_known = torch.cat(
                [map_known + single_pad * i for i in range(self.scalar)]
            ).long()
            idx = (known_bid, map_known)
            padded_ref[idx] = noised
            target_kpts[idx] = known_kpts
            target_kpt_valid[idx] = known_joint_valid
            valid[idx] = True
            is_person[idx] = known_labels == 0

        attn_mask = self._dn_attn_mask(pad_size, single_pad, device)
        dn_info = {
            "target_kpts": target_kpts,
            "target_kpt_valid": target_kpt_valid,
            "valid": valid,
            "is_person": is_person,
        }
        return padded_ref, attn_mask, dn_info

    # ------------------------------------------------------------------ #
    # Heads                                                               #
    # ------------------------------------------------------------------ #
    def _apply_heads(self, dec, reference_points):
        """dec: (L, B, Q, C); reference_points: (B, Q, 3) -> cls, keypoints."""
        L, Bd, Q, _ = dec.shape
        J = self.num_joints
        ref_logit = inverse_sigmoid(reference_points).unsqueeze(2)  # (B, Q, 1, 3)
        if self._shared_branches:
            # Every layer uses the same branch modules: one call over all layers.
            cls = self.cls_branches[0](dec)                                   # (L, B, Q, 1)
            reg = self.reg_branches[0](dec).view(L, Bd, Q, J, 3)
            return cls, torch.sigmoid(reg + ref_logit)
        cls_out, kpt_out = [], []
        for lvl in range(L):
            feat = dec[lvl]
            cls_out.append(self.cls_branches[lvl](feat))           # (B, Q, 1)
            reg = self.reg_branches[lvl](feat).view(Bd, Q, J, 3)
            kpt_out.append(torch.sigmoid(reg + ref_logit))         # (B, Q, J, 3)
        return torch.stack(cls_out, 0), torch.stack(kpt_out, 0)

    # ------------------------------------------------------------------ #
    # Forward (whole temporal window)                                     #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        feats:        torch.Tensor,                      # (B, T, N*sum_L, C)
        mln_input:    torch.Tensor,                      # (B, T, N*sum_L, 14)
        world2img:    torch.Tensor,                      # (B, T, N, 4, 4)
        spatial_shapes: list,                            # [(Hp_l, Wp_l), ...]
        image_hw:     Tuple[int, int],                   # (H,  W)
        gt_kpts_norm: Optional[torch.Tensor] = None,     # (B, T, P, J, 3)
        gt_valid:     Optional[torch.Tensor] = None,     # (B, T, P)
        gt_keypoint_valid: Optional[torch.Tensor] = None, # (B, T, P, J)
        camera_valid: Optional[torch.Tensor] = None,     # (B, T, N)
        images: Optional[torch.Tensor] = None,           # (B, T, N, 3, H, W) normalized (crop re-id)
        track_targets: Optional[dict] = None,
        stream_reset: Optional[torch.Tensor] = None,     # (B,) bool
    ):
        """Run the streaming head over a temporal window.

        Args:
            feats:         (B, T, N*sum_L, C)  spatially-flattened per-camera tokens.
            mln_input:     (B, T, N*sum_L, 14) per-token camera calibration code.
            world2img:     (B, T, N, 4, 4) world→image projection matrices.
            spatial_shapes: [(Hp_l, Wp_l), ...]  per-level spatial dimensions.
            image_hw:      (H,  W)         original image dimensions (pixels).
            gt_kpts_norm:  (B, T, P, J, 3) normalised GT keypoints (DN only).
            gt_valid:      (B, T, P)       person validity mask (DN only).
            gt_keypoint_valid: (B,T,P,J)   per-joint DN supervision mask.

        Returns:
            cls_scores: (L, B, T, Qm, 1)
            keypoints:  (L, B, T, Qm, J, 3) normalised [0, 1]
            dn_meta:    dict for the DN loss, or None.
        """
        B, T, NsumL, C = feats.shape
        N    = world2img.shape[2]
        sumL = NsumL // N                                # total tokens per camera across all levels
        device = feats.device
        # Streaming: lanes with stream_reset False continue the previous call's state.
        continuing = (stream_reset is not None and self.memory_embedding is not None
                      and self.memory_embedding.shape[0] == B)
        if not continuing:
            self.reset_memory()
            first_prev = feats.new_zeros(B)
        else:
            first_prev = (~stream_reset.to(device=device, dtype=torch.bool)).to(feats.dtype)
        tracking = self.max_tracks > 0
        if tracking:
            if (self.track_state is None or self.track_state.valid.shape[0] != B
                    or self.track_state.valid.device != device):
                self.track_state = TrackState.empty(B, self.max_tracks, C, device, feats.dtype,
                                                    self.reid_dims, self.track_manager.gallery_size,
                                                    self.external_dims)
            else:
                self.track_state.reset_lanes(first_prev == 0)
            train_tracks = self.training and track_targets is not None
            slot_valid_frames, track_id_frames, assignments = [], [], []
            id_code_frames, id_ema_frames, aliases = [], [], []
            lane_reset = first_prev == 0

        use_dn = self.training and self.with_dn and gt_kpts_norm is not None
        single_pad = pad_size = 0
        if use_dn:
            single_pad, pad_size = self._dn_pad_size(gt_valid)
            if pad_size == 0:
                use_dn = False

        cls_frames, kpt_frames = [], []
        dn_cls, dn_kpts = [], []
        ref = torch.sigmoid(self.reference_points.weight)
        det_ref = ref.unsqueeze(0).expand(B, -1, -1)
        if use_dn:
            # All frames' denoising queries at once (they depend on ground truth only).
            dn_ref, dn_info = self.prepare_dn_window(gt_kpts_norm, gt_valid, single_pad, pad_size,
                                                     gt_keypoint_valid)
            dn_mask = self._dn_attn_mask_cached(pad_size, single_pad, device)

        # Per-frame views via unbind: its backward is one stack, whereas each feats[:, t] slice
        # allocates (and zero-fills) a full-window gradient in backward (8 x ~1-2 GB per step).
        feat_frames, mln_frames = feats.unbind(1), mln_input.unbind(1)
        for t in range(T):
            prev_exists = first_prev if t == 0 else feats.new_ones(B)
            self.pre_update_memory(B, prev_exists, device, feats.dtype)

            if use_dn:
                padded_ref = torch.cat([dn_ref[:, t].to(ref.dtype), det_ref], dim=1)
                attn_mask = dn_mask
            else:
                padded_ref = det_ref
                attn_mask = None

            query_pos = self.query_embedding(pos2posemb3d(padded_ref, C // 2).to(feats.dtype))
            tgt = torch.zeros_like(query_pos)
            tgt, query_pos, reference_points, temp_memory, temp_pos = self.integrate_temporal_queries(
                query_pos, tgt, padded_ref
            )
            if tracking:
                state = self.track_state
                slot_ref = torch.where(state.valid[..., None], state.ref, torch.full_like(state.ref, 0.5))
                track_pos = (
                    self.query_embedding(pos2posemb3d(slot_ref, C // 2).to(feats.dtype))
                    # Age = frames since the slot was last updated (0 while active),
                    # so stale slots in a gap are distinguishable, in training and inference.
                    + self.time_embedding(
                        pos2posemb1d(state.inactive[..., None].to(slot_ref.dtype), C).to(feats.dtype))
                    + self.track_embedding.to(feats.dtype)
                )
                tgt = torch.cat([tgt, state.embed.to(tgt.dtype)], dim=1)
                query_pos = torch.cat([query_pos, track_pos], dim=1)
                reference_points = torch.cat([reference_points, slot_ref.to(reference_points.dtype)], dim=1)
                attn_mask = track_key_mask(
                    attn_mask, pad_size + self.num_query, state.valid, temp_memory.shape[1], self.num_heads,
                )

            # Spatial alignment: inject camera calibration into per-token feats.
            # feats[:, t]: (B, N*sum_L, C); reshape to (B*N, sum_L, C) for the decoder.
            frame_feat = self.spatial_alignment(feat_frames[t], mln_frames[t])
            frame_feat_bn = frame_feat.view(B, N, sumL, C).reshape(B * N, sumL, C)

            w2i_t = world2img[:, t]                      # (B, N, 4, 4)

            dec = self.decoder(
                tgt,
                query_pos,
                frame_feat_bn,
                reference_points,
                w2i_t,
                spatial_shapes,
                image_hw,
                temp_memory=temp_memory,
                temp_pos=temp_pos,
                attn_mask=attn_mask,
                camera_valid=None if camera_valid is None else camera_valid[:, t],
            )

            cls, kpts = self._apply_heads(dec, reference_points)

            match_cls  = cls[:, :, pad_size:]
            match_kpts = kpts[:, :, pad_size:]
            if tracking:
                # Invalid slots carry no prediction: constant logits, excluded from losses/memory.
                invalid = torch.cat([state.valid.new_zeros(B, self.num_query), ~state.valid], dim=1)
                match_cls = match_cls.masked_fill(invalid[None, :, :, None], -20.0)
            self.post_update_memory(match_cls, match_kpts, dec[:, :, pad_size:])
            if tracking:
                slot_valid_frames.append(state.valid.clone())
                feat_last = dec[-1, :, pad_size:].detach()
                codes = ema = app = None
                if self.reid_dims:
                    cam_t = None if camera_valid is None else camera_valid[:, t]
                    if self.external_reid is not None:
                        app = self._slot_appearance(images, t, match_kpts[-1].detach(), w2i_t, cam_t, state)
                    codes, ema = self._identity_codes(
                        dec[-1, :, pad_size:], match_kpts[-1].detach(), feat_frames[t], spatial_shapes,
                        w2i_t, image_hw, cam_t, state, app,
                    )
                    id_code_frames.append(codes)
                    id_ema_frames.append(ema)
                    codes, ema = codes.detach(), ema.detach()
                if train_tracks:
                    assign, self.track_state = self.track_manager.train_step(
                        state, self.num_query, match_cls[-1].detach(), match_kpts[-1].detach(),
                        feat_last, track_targets, t, codes=codes, ema=ema, app=app,
                    )
                    assignments.append(assign)
                    ids = torch.cat([torch.full((B, self.num_query), -1, dtype=torch.long, device=device),
                                     torch.where(state.valid, state.ids, -1)], dim=1)
                else:
                    ids, self.track_state, new_aliases = self.track_manager.infer_step(
                        state, self.num_query, match_cls[-1].detach().float(),
                        match_kpts[-1].detach().float(), feat_last.float(), codes=codes, ema=ema, app=app,
                    )
                    aliases.extend(new_aliases)
                track_id_frames.append(ids)
                self._frames_seen += 1

            cls_frames.append(match_cls)
            kpt_frames.append(match_kpts)
            if use_dn:
                dn_cls.append(cls[:, :, :pad_size])
                dn_kpts.append(kpts[:, :, :pad_size])

        self.track_output = None
        if tracking:
            self.track_output = {
                "num_det": self.num_query,
                "slot_valid": torch.stack(slot_valid_frames, dim=1),     # (B, T, K)
                "track_ids": torch.stack(track_id_frames, dim=1),        # (B, T, Qm), -1 = none
                # [T][B] dicts (host path) or, with model.gpu_assignment, tensors
                # {"target", "excluded"} (B, T, Qm).
                "assignments": (None if not train_tracks else
                                {k: torch.stack([a[k] for a in assignments], dim=1) for k in assignments[0]}
                                if assignments and isinstance(assignments[0], dict) and "target" in assignments[0]
                                else assignments),
                "aliases": aliases,            # inference re-id takeovers: (lane, provisional, recovered)
                "lane_reset": lane_reset,      # (B,) lanes whose state was reset by this call
            }
            if self.reid_dims:
                self.track_output["id_codes"] = torch.stack(id_code_frames, dim=1)  # (B, T, Qm, D)
                self.track_output["id_ema"] = torch.stack(id_ema_frames, dim=1)     # (B, T, K, D)
                if train_tracks:
                    self.track_output["instance_ids"] = track_targets["instance_ids"]
        cls_scores = torch.stack(cls_frames, dim=2)   # (L, B, T, Qm, 1)
        keypoints  = torch.stack(kpt_frames, dim=2)   # (L, B, T, Qm, J, 3)

        dn_meta = None
        if use_dn:
            dn_meta = {
                "dn_cls":       torch.stack(dn_cls,    dim=2),  # (L, B, T, pad, 1)
                "dn_kpts":      torch.stack(dn_kpts,   dim=2),  # (L, B, T, pad, J, 3)
                **dn_info,  # target_kpts (B,T,pad,J,3), target_kpt_valid, valid, is_person (B,T,pad)
            }
        return cls_scores, keypoints, dn_meta