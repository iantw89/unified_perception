"""Top-level model: YOLO image features + 3D position encoding + StreamPETR head.

Mirrors the StreamPETR data flow but in a static world frame and predicting
per-person 3D keypoints instead of 3D boxes:

    images (B,T,N,3,H,W) --YOLO--> per-view tokens
                          + 3D position embedding (from calibration)
                          + per-view calibration code (for spatial alignment)
    person queries --SparseHead--> per-person classification + keypoints

All of the streaming/temporal logic (memory bank, temporal alignment, top-k
proposal propagation, query denoising) lives inside :class:`SparseHead`; the
detector only produces the per-frame image tokens and geometry that the head
consumes.  Each forward call processes a self-contained temporal window.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config
from .position_encoding import CameraPositionEncoder3D
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
            feature_index=m.yolo_feature_index,
            trainable=m.yolo_trainable,
        )
        # embed_dims is the backbone''s native feature-map channel count.
        with torch.no_grad():
            _feat, _ = self.backbone(torch.zeros(1, 3, 256, 256))
        self.embed_dims: int = int(_feat.shape[1])

        self.pos_encoder = CameraPositionEncoder3D(
            embed_dims=self.embed_dims,
            position_range=m.position_range,
            depth_num=m.depth_num,
            depth_start_mm=m.depth_start_mm,
            depth_max_mm=m.depth_max_mm,
            lid=m.lid_depth,
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
            with_dn=m.with_dn,
            scalar=m.dn_scalar,
            noise_scale=m.dn_noise_scale,
            split=m.dn_split,
        )

    # ------------------------------------------------------------------ #
    def _encode_frame(self, imgs_t, world2img_t, intrinsics_t):
        """Encode one frame''s views into tokens, 3D position embeddings and a
        per-token camera-calibration code for spatial alignment.

        imgs_t: (B, N, 3, H, W); world2img_t/intrinsics_t: (B, N, 4, 4).
        Returns tokens (B, N*L, C), pos (B, N*L, C), mln (B, N*L, 14).
        """
        B, N, _, H, W = imgs_t.shape
        feat, _pose_preds = self.backbone(imgs_t.reshape(B * N, 3, H, W))  # (B*N, C, Hp, Wp)
        _, C, Hp, Wp = feat.shape
        L = Hp * Wp
        tokens = feat.flatten(2).transpose(1, 2).view(B, N, L, C)
        tokens = tokens + self.cam_embed[:N].view(1, N, 1, C)

        img2world = torch.linalg.inv(world2img_t.float())            # (B, N, 4, 4)
        pos = self.pos_encoder(img2world, (Hp, Wp), (H, W))          # (B, N, L, C)

        if intrinsics_t is not None:
            K = intrinsics_t.float()
            extr = torch.matmul(torch.linalg.inv(K), world2img_t.float())  # (B, N, 4, 4)
            fx = (K[..., 0, 0] / 1e3).unsqueeze(-1)                   # (B, N, 1)
            fy = (K[..., 1, 1] / 1e3).unsqueeze(-1)                   # (B, N, 1)
            mln = torch.cat([fx, fy, extr[..., :3, :].reshape(B, N, 12)], dim=-1)  # (B, N, 14)
            mln = mln.unsqueeze(2).expand(B, N, L, 14).reshape(B, N * L, 14).to(tokens.dtype)
        else:
            mln = tokens.new_zeros(B, N * L, 14)

        tokens = tokens.reshape(B, N * L, C)
        pos = pos.reshape(B, N * L, C)
        return tokens, pos, mln

    def forward(
        self,
        imgs,
        world2img,
        intrinsics: Optional[torch.Tensor] = None,
        gt_keypoints_norm: Optional[torch.Tensor] = None,
        gt_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """imgs: (B,T,N,3,H,W); world2img/intrinsics: (B,T,N,4,4).

        Returns cls_scores (L,B,T,Q,1), keypoints_norm (L,B,T,Q,J,3), dn_meta.
        ``gt_keypoints_norm``/``gt_valid`` are only used for query denoising
        during training and may be omitted at inference time.
        """
        T = imgs.shape[1]
        feats, poss, mlns = [], [], []
        for t in range(T):
            k_t = None if intrinsics is None else intrinsics[:, t]
            ft, pt, mt = self._encode_frame(imgs[:, t], world2img[:, t], k_t)
            feats.append(ft)
            poss.append(pt)
            mlns.append(mt)
        feats = torch.stack(feats, dim=1)   # (B, T, M, C)
        poss = torch.stack(poss, dim=1)
        mlns = torch.stack(mlns, dim=1)

        return self.head(feats, poss, mlns, gt_keypoints_norm, gt_valid)

    # ------------------------------------------------------------------ #
    def denormalize_keypoints(self, kpts_norm):
        """Map normalised [0,1] keypoints to world millimetres."""
        pr = self.position_range
        return kpts_norm * (pr[3:6] - pr[:3]) + pr[:3]

    def normalize_keypoints(self, kpts_mm):
        pr = self.position_range
        return (kpts_mm - pr[:3]) / (pr[3:6] - pr[:3])