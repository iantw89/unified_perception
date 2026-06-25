"""StreamPETR sparse head, re-implemented for multi-person 3D keypoints.

This is a faithful port of StreamPETR''s ``SparseHead`` (the streaming sparse
query head) adapted to this codebase, with three deliberate modifications:

1. **No ego-motion compensation.**  The capture rig is static and the world
   frame is fixed, so the streaming memory bank stores reference points in a
   single world frame -- there is no ego-pose transform of the memory and no
   ego-pose positional encoding.
2. **3D keypoints instead of 3D boxes.**  Each query regresses ``J`` keypoints
   (anchored at its 3D reference point) in normalised ``[0, 1]`` volume
   coordinates rather than a 9/10-D bounding box.
3. **Single class (person).**  Classification is a single "is a person" logit
   (sigmoid), so there is no multi-class head.

Retained StreamPETR machinery
-----------------------------
* a streaming **memory bank** (content + reference point + timestamp) that
  propagates confident detections across frames;
* **temporal alignment** -- memory queries are mixed into the current query set
  and contribute extra self-attention keys, with a time embedding encoding
  memory age;
* **spatial alignment** -- a modulated LayerNorm injects per-view camera
  calibration into the image tokens;
* top-k **proposal propagation** into the memory bank;
* **query denoising (DN)** during training, here noising the per-person 3D
  reference (mid-hip) anchor.

The head consumes per-frame image tokens (already produced by the YOLO backbone
and 3D position encoder in the detector) and manages its own memory across the
frames of a temporal window.
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
        with_dn: bool = True,
        scalar: int = 10,
        noise_scale: float = 0.2,
        split: float = 0.75,
        time_step: float = 1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_joints = num_joints
        self.num_query = num_query
        self.num_propagated = num_propagated
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_layers = num_layers
        self.with_dn = with_dn
        self.scalar = scalar
        self.dn_noise_scale = noise_scale
        self.dn_split = split
        self.time_step = float(time_step)

        self.decoder = TransformerDecoder(
            dim=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
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

        # Learnable query anchors (normalised [0, 1]) and propagated pseudo anchors.
        self.reference_points = nn.Embedding(num_query, 3)
        nn.init.uniform_(self.reference_points.weight, 0, 1)
        if num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(num_propagated, 3)
            nn.init.uniform_(self.pseudo_reference_points.weight, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.reset_memory()

    # ------------------------------------------------------------------ #
    # Streaming memory                                                    #
    # ------------------------------------------------------------------ #
    def reset_memory(self):
        self.memory_embedding: Optional[torch.Tensor] = None
        self.memory_reference_point: Optional[torch.Tensor] = None
        self.memory_timestamp: Optional[torch.Tensor] = None

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

    def temporal_alignment(self, query_pos, tgt, reference_points):
        """Mix streaming memory queries into the current query set."""
        C = self.embed_dims
        temp_ref = self.memory_reference_point                       # (B, mem, 3)
        temp_pos = self.query_embedding(pos2posemb3d(temp_ref, C // 2))
        temp_memory = self.memory_embedding

        query_pos = query_pos + self.time_embedding(
            pos2posemb1d(torch.zeros_like(reference_points[..., :1]), C)
        )
        temp_pos = temp_pos + self.time_embedding(pos2posemb1d(self.memory_timestamp, C))

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, : self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, : self.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points, temp_ref[:, : self.num_propagated]], dim=1
            )
            temp_memory = temp_memory[:, self.num_propagated :]
            temp_pos = temp_pos[:, self.num_propagated :]

        return tgt, query_pos, reference_points, temp_memory, temp_pos

    # ------------------------------------------------------------------ #
    # Query denoising (DN)                                                #
    # ------------------------------------------------------------------ #
    def _dn_pad_size(self, gt_valid) -> Tuple[int, int]:
        """single_pad = max #people over the batch & time; pad = single_pad * scalar."""
        per_frame = gt_valid.reshape(-1, gt_valid.shape[-1]).sum(dim=-1)
        single_pad = int(per_frame.max().item()) if per_frame.numel() else 0
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

    def prepare_for_dn(self, B, ref, gt_kpts, gt_valid, single_pad, pad_size):
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
        known_num: List[int] = []
        for bi in range(B):
            v = gt_valid[bi].bool()
            k = gt_kpts[bi][v]
            per_kpts.append(k)
            known_num.append(int(k.shape[0]))
        total = sum(known_num)

        padded_ref = torch.cat([ref.new_zeros(pad_size, 3), ref], dim=0)
        padded_ref = padded_ref.unsqueeze(0).repeat(B, 1, 1)
        target_kpts = gt_kpts.new_zeros(B, pad_size, J, 3)
        valid = torch.zeros(B, pad_size, dtype=torch.bool, device=device)
        is_person = torch.zeros(B, pad_size, dtype=torch.bool, device=device)

        if total > 0:
            all_kpts = torch.cat(per_kpts, dim=0)                 # (total, J, 3)
            roots = 0.5 * (all_kpts[:, root1] + all_kpts[:, root2])       # (total, 3)
            batch_idx = torch.cat(
                [torch.full((known_num[i],), i, dtype=torch.long, device=device) for i in range(B)]
            )
            known_kpts = all_kpts.repeat(self.scalar, 1, 1)
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
            valid[idx] = True
            is_person[idx] = known_labels == 0

        attn_mask = self._dn_attn_mask(pad_size, single_pad, device)
        dn_info = {"target_kpts": target_kpts, "valid": valid, "is_person": is_person}
        return padded_ref, attn_mask, dn_info

    # ------------------------------------------------------------------ #
    # Heads                                                               #
    # ------------------------------------------------------------------ #
    def _apply_heads(self, dec, reference_points):
        """dec: (L, B, Q, C); reference_points: (B, Q, 3) -> cls, keypoints."""
        L, Bd, Q, _ = dec.shape
        J = self.num_joints
        ref_logit = inverse_sigmoid(reference_points).unsqueeze(2)  # (B, Q, 1, 3)
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
    def forward(self, feats, pos, mln_input, gt_kpts_norm=None, gt_valid=None):
        """Run the streaming head over a temporal window.

        Args:
            feats: (B, T, M, C) per-frame image tokens (pre-processed by detector).
            pos:   (B, T, M, C) per-frame 3D position embeddings.
            mln_input: (B, T, M, 14) per-token camera calibration code.

            gt_kpts_norm: (B, T, P, J, 3) normalised GT keypoints (DN only).
            gt_valid: (B, T, P) bool (DN only).
        Returns:
            cls_scores: (L, B, T, Qm, 1)
            keypoints:  (L, B, T, Qm, J, 3) normalised [0, 1]
            dn_meta:    dict for the DN loss, or None.
        """
        B, T, M, C = feats.shape
        device = feats.device
        self.reset_memory()

        use_dn = self.training and self.with_dn and gt_kpts_norm is not None
        single_pad = pad_size = 0
        if use_dn:
            single_pad, pad_size = self._dn_pad_size(gt_valid)
            if pad_size == 0:
                use_dn = False

        cls_frames, kpt_frames = [], []
        dn_cls, dn_kpts, dn_tgt, dn_valid, dn_person = [], [], [], [], []

        for t in range(T):
            prev_exists = feats.new_zeros(B) if t == 0 else feats.new_ones(B)
            self.pre_update_memory(B, prev_exists, device, feats.dtype)

            ref = self.reference_points.weight
            if use_dn:
                padded_ref, attn_mask, dn_info = self.prepare_for_dn(
                    B, ref, gt_kpts_norm[:, t], gt_valid[:, t], single_pad, pad_size
                )
            else:
                padded_ref = ref.unsqueeze(0).repeat(B, 1, 1)
                attn_mask, dn_info = None, None

            query_pos = self.query_embedding(pos2posemb3d(padded_ref, C // 2))
            tgt = torch.zeros_like(query_pos)
            tgt, query_pos, reference_points, temp_memory, temp_pos = self.temporal_alignment(
                query_pos, tgt, padded_ref
            )

            frame_feat = self.spatial_alignment(feats[:, t], mln_input[:, t])
            frame_pos = pos[:, t]

            dec = self.decoder(
                tgt,
                query_pos,
                frame_feat,
                frame_pos,
                temp_memory=temp_memory,
                temp_pos=temp_pos,
                attn_mask=attn_mask,
            )

            cls, kpts = self._apply_heads(dec, reference_points)

            match_cls = cls[:, :, pad_size:]
            match_kpts = kpts[:, :, pad_size:]
            self.post_update_memory(match_cls, match_kpts, dec[:, :, pad_size:])

            cls_frames.append(match_cls)
            kpt_frames.append(match_kpts)
            if use_dn:
                dn_cls.append(cls[:, :, :pad_size])
                dn_kpts.append(kpts[:, :, :pad_size])
                dn_tgt.append(dn_info["target_kpts"])
                dn_valid.append(dn_info["valid"])
                dn_person.append(dn_info["is_person"])

        cls_scores = torch.stack(cls_frames, dim=2)   # (L, B, T, Qm, 1)
        keypoints = torch.stack(kpt_frames, dim=2)    # (L, B, T, Qm, J, 3)

        dn_meta = None
        if use_dn:
            dn_meta = {
                "dn_cls": torch.stack(dn_cls, dim=2),       # (L, B, T, pad, 1)
                "dn_kpts": torch.stack(dn_kpts, dim=2),     # (L, B, T, pad, J, 3)
                "target_kpts": torch.stack(dn_tgt, dim=1),  # (B, T, pad, J, 3)
                "valid": torch.stack(dn_valid, dim=1),      # (B, T, pad)
                "is_person": torch.stack(dn_person, dim=1), # (B, T, pad)
            }
        return cls_scores, keypoints, dn_meta