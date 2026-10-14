"""Top-level model: YOLO image features + Detr3D-style cross-attention head.

Mirrors the StreamPETR / RepDETR3D data flow in a static world frame,
predicting per-person 3D keypoints instead of 3D boxes:

    images (B,T,N,3,H,W) --YOLO→ per-view tokens
                          + per-view calibration code  (spatial alignment)
    world2img matrices   -------> passed to SparseHead for 3D projection
    person queries       -SparseHead→ per-person classification + keypoints

Unlike the previous PETR-style version there is no ``CameraPositionEncoder3D``.
The 3D geometry is handled inside ``Detr3DCrossAttn`` by projecting each
query's 3D reference point into the image plane of every camera and sampling
features there.  This matches the approach used in RepDETR3D / StreamPETR.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config
from .sparse_head import SparseHead
from .yolo_backbone import YOLOPoseBackbone


class UnifiedPerceptionModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.num_joints = m.num_joints
        self.register_buffer(
            "position_range", torch.tensor(m.position_range, dtype=torch.float32)
        )

        self.backbone = YOLOPoseBackbone(
            model_name=m.yolo_model_name,
            feature_indices=m.yolo_feature_indices,
            trainable=m.yolo_trainable,
        )
        # Probe per-level channel counts and build 1×1 linear projections
        # to unify all scales to embed_dims.
        with torch.no_grad():
            _probe, _ = self.backbone(torch.zeros(1, 3, 256, 256))
        feat_channels = [int(f.shape[1]) for f in _probe]
        self.embed_dims: int = m.embed_dims
        self.num_levels: int = len(feat_channels)
        self.feat_projs = nn.ModuleList(
            [
                nn.Linear(c, self.embed_dims) if c != self.embed_dims else nn.Identity()
                for c in feat_channels
            ]
        )

        # Per-camera learnable embedding (helps disambiguate views).
        self.cam_embed = nn.Parameter(torch.zeros(cfg.data.num_cameras, self.embed_dims))
        nn.init.trunc_normal_(self.cam_embed, std=0.02)

        self.head = SparseHead(
            embed_dims=self.embed_dims,
            num_joints=m.num_joints,
            num_query=m.num_queries,
            num_propagated=m.num_propagated,
            memory_len=m.memory_len,
            topk_proposals=m.memory_topk,
            num_layers=m.dec_layers,
            num_heads=m.dec_heads,
            ffn_dim=m.dec_ffn_dim,
            dropout=m.dropout,
            num_cams=cfg.data.num_cameras,
            num_levels=self.num_levels,
            num_pts=m.dec_num_pts,
            position_range=m.position_range,
            num_groups=m.dec_groups,
            with_dn=m.with_dn,
            scalar=m.dn_scalar,
            noise_scale=m.dn_noise_scale,
            split=m.dn_split,
            use_mmcv_msda=m.use_mmcv_msda,
        )

    # ------------------------------------------------------------------ #
    def _encode_frame(
        self,
        imgs_t:            torch.Tensor,           # (B, N, 3, H, W)
        world2img_t:       torch.Tensor,           # (B, N, 4, 4)
        intrinsics_t:      Optional[torch.Tensor], # (B, N, 4, 4) or None
        return_detections: bool = False,
    ):
        """Run the backbone and build per-token calibration codes.

        Backbone is called once for all B×N images and returns one feature map
        per tapped layer.  All levels are projected to ``self.embed_dims`` with
        a per-level 1×1 linear layer and then concatenated along the spatial
        dimension before returning.

        Returns:
            tokens        (B, N*sum_L, C)  image tokens with camera embedding.
            mln           (B, N*sum_L, 14) per-token camera calibration code.
            spatial_shapes list[(Hp_l, Wp_l)] one entry per feature level.
            detections    list of B*N dicts — only when return_detections=True, None otherwise.
        """
        B, N, _, H, W = imgs_t.shape
        C = self.embed_dims

        # Run backbone once for all B*N images.
        feats_list, dets_flat = self.backbone(
            imgs_t.reshape(B * N, 3, H, W), return_detections=return_detections
        )
        # feats_list: [Tensor(B*N, C_l, Hp_l, Wp_l) for each level l]

        all_tokens: list[torch.Tensor] = []
        spatial_shapes: list[tuple[int, int]] = []

        for feat_l, proj_l in zip(feats_list, self.feat_projs):
            _, _, Hp_l, Wp_l = feat_l.shape
            # (B*N, C_l, Hp, Wp) → (B*N, Hp*Wp, C_l) → project → (B*N, Hp*Wp, C)
            toks_l = feat_l.flatten(2).transpose(1, 2)
            toks_l = proj_l(toks_l)                                   # (B*N, L_l, C)
            # Reshape to (B, N, L_l, C) and add per-camera embedding
            toks_l = toks_l.view(B, N, Hp_l * Wp_l, C)
            toks_l = toks_l + self.cam_embed[:N].view(1, N, 1, C)
            all_tokens.append(toks_l)
            spatial_shapes.append((Hp_l, Wp_l))

        # Concatenate levels along spatial dim: (B, N, sum_L, C)
        tokens = torch.cat(all_tokens, dim=2)
        sum_L  = tokens.shape[2]
        tokens = tokens.reshape(B, N * sum_L, C)

        # Per-token camera calibration code for spatial alignment (MLN).
        if intrinsics_t is not None:
            K    = intrinsics_t.float()
            extr = torch.matmul(torch.linalg.inv(K), world2img_t.float())
            # extr = [R | t] where t is in mm.  fx/fy are already /1000.
            # Normalize translation column by 1000 as well so all 14 elements
            # are O(1-5); raw mm values (±3000) would push the MLN reduce-layer
            # activations to O(1000) and cause gradient explosions on beta/gamma.
            _e = extr[..., :3, :].clone()
            _e[..., 3] = _e[..., 3] / 1e3   # mm → 1000-mm units  (O(1-5))
            fx   = (K[..., 0, 0] / 1e3).unsqueeze(-1)
            fy   = (K[..., 1, 1] / 1e3).unsqueeze(-1)
            mln  = torch.cat([fx, fy, _e.reshape(B, N, 12)], dim=-1)  # (B, N, 14)
            mln  = mln.unsqueeze(2).expand(B, N, sum_L, 14).reshape(B, N * sum_L, 14).to(tokens.dtype)
        else:
            mln = tokens.new_zeros(B, N * sum_L, 14)

        return tokens, mln, spatial_shapes, dets_flat

    def forward(
        self,
        imgs:               torch.Tensor,           # (B, T, N, 3, H, W)
        world2img:          torch.Tensor,           # (B, T, N, 4, 4)
        intrinsics:         Optional[torch.Tensor] = None,  # (B, T, N, 4, 4)
        gt_keypoints_norm:  Optional[torch.Tensor] = None,
        gt_valid:           Optional[torch.Tensor] = None,
        return_2d:          bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict], Optional[list]]:
        """Forward pass over a temporal window.

        Returns ``(cls_scores, kpts_norm, dn_meta, dets_2d)``

        ``dets_2d`` is a flat list of ``T*B*N`` detection dicts (one per image
        in time-major, batch-major, camera-major order).  Each dict contains:
            boxes  (K, 4)    float32  xyxy pixel coords
            kpts   (K, J, 2) float32  xy pixel coords
            scores (K,)      float32  objectness score
        For ``B=1`` the index ``t * N + c`` gives frame t, camera c.

        ``gt_keypoints_norm``/``gt_valid`` are used only during training (DN).
        """
        B, T, N, _, H, W = imgs.shape
        feats, mlns = [], []
        spatial_shapes = None
        dets_2d: list = []
        for t in range(T):
            k_t = None if intrinsics is None else intrinsics[:, t]
            ft, mt, spatial_shapes, dets_t = self._encode_frame(
                imgs[:, t], world2img[:, t], k_t, return_detections=return_2d
            )
            if return_2d:
                dets_2d.extend(dets_t)   # B*N dicts per timestep → T*B*N total
            feats.append(ft)
            mlns.append(mt)
        feats = torch.stack(feats, dim=1)   # (B, T, N*sum_L, C)
        mlns  = torch.stack(mlns,  dim=1)   # (B, T, N*sum_L, 14)

        cls_scores, kpts_norm, dn_meta = self.head(
            feats, mlns, world2img, spatial_shapes, (H, W),
            gt_keypoints_norm, gt_valid,
        )
        return cls_scores, kpts_norm, dn_meta, dets_2d if return_2d else None

    # ------------------------------------------------------------------ #
    def denormalize_keypoints(self, kpts_norm):
        pr = self.position_range
        return kpts_norm * (pr[3:6] - pr[:3]) + pr[:3]

    def normalize_keypoints(self, kpts_mm):
        pr = self.position_range
        return (kpts_mm - pr[:3]) / (pr[3:6] - pr[:3])