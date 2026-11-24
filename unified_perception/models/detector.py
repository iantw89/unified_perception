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
from .backbones import BACKBONES, resolve_backbone, resolve_repo_path
from .pose2d_aux import Pose2DHead, aux2d_loss
from .vit_backbone import DinoV3SimpleFPNBackbone, fpn_levels_for
from .reid import ExternalReID
from .sparse_head import SparseHead
from .tracking import TrackManager
from .yolo_backbone import YOLOPoseBackbone



def _host_index(mask_host: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
    """Flat long index of the True entries of a CPU mask, on ``device`` without a sync."""
    if mask_host is None:
        return None
    index = mask_host.reshape(-1).nonzero().squeeze(1)
    if torch.device(device).type == "cuda":
        return index.pin_memory().to(device, non_blocking=True)
    return index.to(device)

def _active_rows(image_hw_host: Optional[torch.Tensor], mask_host: Optional[torch.Tensor],
                 T: int) -> Optional[torch.Tensor]:
    """(B, N, 2) per-camera valid image sizes -> one (h, w) per backbone image, in the (B, T, N)
    flattening order of the window's active views (``mask_host`` (B, T, N); all views if None)."""
    if image_hw_host is None:
        return None
    rows = image_hw_host[:, None].expand(-1, T, -1, -1).reshape(-1, 2)
    return rows if mask_host is None else rows[mask_host.reshape(-1).bool()]


class UnifiedPerceptionModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.num_joints = m.num_joints
        self.register_buffer(
            "position_range", torch.tensor(m.position_range, dtype=torch.float32)
        )
        # uint8 inputs are normalized one frame at a time (saves the full float32 window).
        self.register_buffer("input_mean", torch.tensor(cfg.data.pixel_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("input_std", torch.tensor(cfg.data.pixel_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
                             persistent=False)

        model_name, feature_indices = m.yolo_model_name, m.yolo_feature_indices
        if m.backbone:
            model_name, feature_indices = resolve_backbone(m.backbone)
        if m.backbone and BACKBONES[m.backbone].kind == "vit":
            self.backbone = DinoV3SimpleFPNBackbone(
                model_name, arch=BACKBONES[m.backbone].arch, out_channels=m.embed_dims,
                trainable_blocks=m.vit_trainable_blocks, compile_trunk=m.vit_compile,
                fpn_levels=m.vit_fpn_levels or fpn_levels_for(cfg.data.image_height, cfg.data.image_width))
        else:
            self.backbone = YOLOPoseBackbone(
                model_name=model_name,
                feature_indices=feature_indices,
                trainable=m.yolo_trainable,
            )
        # Native-aspect canvases: the backbone runs only each image's valid region (ViT backbones).
        self.crop_to_valid = bool(getattr(self.backbone, "accepts_valid_hw", False))
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
        # Auxiliary 2-D pose heatmaps on the finest (stride-8) level (experimental; training only).
        self.aux2d = (Pose2DHead(feat_channels[0], m.num_joints, m.aux2d_hidden) if m.aux2d_weight > 0 else None)
        self.aux2d_output: Optional[dict] = None
        self._aux_feat: Optional[torch.Tensor] = None

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
            max_tracks=m.max_tracks if m.track_queries else 0,
            track_manager=TrackManager(m, m.position_range) if m.track_queries else None,
            reid_dims=m.reid_dims if m.track_queries else 0,
            reid_appearance=m.reid_appearance,
            reid_ema=m.reid_ema,
            reid_query_grad_scale=getattr(m, "reid_query_grad_scale", 1.0),
            external_reid=(ExternalReID(resolve_repo_path(m.reid_external_model),
                                        max_crops=getattr(m, "reid_external_max_crops", 0))
                           if m.track_queries and m.reid_dims and m.reid_external_model else None),
            external_stride=m.reid_external_stride,
            pixel_mean=cfg.data.pixel_mean,
            pixel_std=cfg.data.pixel_std,
        )

    # ------------------------------------------------------------------ #
    def _run_backbone(self, flat_images: torch.Tensor, active: Optional[torch.Tensor],
                      return_detections: bool = False, crop_rows: Optional[torch.Tensor] = None):
        """Backbone features of (M, 3, H, W) images, zero for inactive (padded) views."""
        # Exclude padding from the backbone, including trainable BatchNorm statistics.
        # uint8 images are normalized chunk by chunk inside the backbone call (a float32
        # copy of the whole window would cost ~1 GB per batch).
        # ``active`` is a boolean mask or (preferred) the long index of the active views; an index
        # built on the host avoids the device synchronization of boolean-mask indexing.
        index = None
        if active is not None:
            index = active.nonzero().squeeze(1) if active.dtype == torch.bool else active
        images = flat_images if index is None else flat_images.index_select(0, index)
        mean, std = self.input_mean.view(1, 3, 1, 1), self.input_std.view(1, 3, 1, 1)
        if getattr(self.backbone, "normalizes_uint8", False):
            crop = {"valid_hw": crop_rows} if crop_rows is not None and getattr(self, "crop_to_valid", False) else {}
            feats_list, dets = self.backbone(images, return_detections=return_detections,
                                             input_mean=mean, input_std=std, **crop)
        else:  # other backbones (e.g. test stubs) take normalized float images
            if images.dtype == torch.uint8:
                images = (images.float() - mean) / std
            feats_list, dets = self.backbone(images, return_detections=return_detections)
        if index is not None:
            padded_feats = []
            for feat in feats_list:
                padded = feat.new_zeros(flat_images.shape[0], *feat.shape[1:])
                padded_feats.append(padded.index_copy(0, index, feat))
            feats_list = padded_feats
        return feats_list, dets

    def _encode_frame(
        self,
        imgs_t:            torch.Tensor,           # (B, N, 3, H, W) uint8 or normalized float
        world2img_t:       torch.Tensor,           # (B, N, 4, 4)
        intrinsics_t:      Optional[torch.Tensor], # (B, N, 4, 4) or None
        return_detections: bool = False,
        camera_valid: Optional[torch.Tensor] = None,
        active_index: Optional[torch.Tensor] = None,  # long index of active views (host-built)
        crop_rows: Optional[torch.Tensor] = None,     # (active views, 2) host valid (h, w) sizes
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

        active = None if camera_valid is None else camera_valid.reshape(-1)
        feats_list, dets_flat = self._run_backbone(
            imgs_t.reshape(B * N, 3, H, W), active if active_index is None else active_index, return_detections, crop_rows)
        if self.aux2d is not None and self.training:
            self._aux_feat = feats_list[0]          # (B*N, C, H/8, W/8), zero for inactive views
        if active is not None:
            if return_detections and dets_flat is not None:   # (ViT backbones report none)
                detections = iter(dets_flat)
                template = dets_flat[0]
                dets_flat = [
                    next(detections) if valid else {key: value[:0] for key, value in template.items()}
                    for valid in active.tolist()
                ]
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
            extr = torch.matmul(torch.linalg.inv_ex(K).inverse, world2img_t.float())  # _ex: no error-check sync
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
        gt_keypoint_valid:  Optional[torch.Tensor] = None,
        camera_valid:       Optional[torch.Tensor] = None,  # (B, T, N), True for real views
        track_targets:      Optional[dict] = None,
        stream_reset:       Optional[torch.Tensor] = None,  # (B,) bool
        camera_valid_host:  Optional[torch.Tensor] = None,  # CPU copy of camera_valid (no device sync)
        image_hw_host:     Optional[torch.Tensor] = None,  # (B, N, 2) CPU valid image sizes (data/collate.py)
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
        ``camera_valid`` excludes padded views from the backbone and attention.
        Masked views have empty 2-D detections, preserving flat-list indexing.

        With ``model.track_queries``, ``track_targets`` (normalized GT plus
        ``instance_ids``/``exists``) drives identity assignment in training and
        ``self.head.track_output`` holds per-query track ids / assignments.
        ``stream_reset`` (B,) continues the temporal state of lanes that are
        False from the previous call (streaming); None resets every lane.
        """
        B, T, N, _, H, W = imgs.shape
        if N > self.cam_embed.shape[0]:
            raise ValueError("Input camera count exceeds configured num_cameras")
        if camera_valid is not None:
            if camera_valid.shape != (B, T, N) or camera_valid.dtype != torch.bool:
                raise ValueError("camera_valid must be boolean with shape (B,T,N)")
            host_mask = camera_valid_host if camera_valid_host is not None else (
                None if camera_valid.is_cuda else camera_valid)
            if host_mask is not None:
                if not host_mask.any(dim=-1).all():
                    raise ValueError("Every sample frame must contain at least one valid camera")
            else:   # device mask without a host copy: checked asynchronously (no sync)
                torch._assert_async(camera_valid.any(dim=-1).all(),
                                    "Every sample frame must contain at least one valid camera")
            identity = torch.eye(4, device=world2img.device, dtype=world2img.dtype)
            world2img = torch.where(camera_valid[..., None, None], world2img, identity)
            if intrinsics is not None:
                intrinsics = torch.where(camera_valid[..., None, None], intrinsics, identity)
        dets_2d: list = []
        if return_2d:
            # Detections are reported time-major: encode frame by frame.
            feats, mlns = [], []
            spatial_shapes = None
            for t in range(T):
                k_t = None if intrinsics is None else intrinsics[:, t]
                ft, mt, spatial_shapes, dets_t = self._encode_frame(
                    imgs[:, t], world2img[:, t], k_t, return_detections=True,
                    camera_valid=None if camera_valid is None else camera_valid[:, t],
                )
                dets_2d.extend(dets_t or [])   # (ViT backbones report none) B*N dicts per timestep → T*B*N total
                feats.append(ft)
                mlns.append(mt)
            feats = torch.stack(feats, dim=1)   # (B, T, N*sum_L, C)
            mlns  = torch.stack(mlns,  dim=1)   # (B, T, N*sum_L, 14)
        else:
            # All frames of the window in one pass (frames are independent here): one
            # backbone sweep in large chunks instead of T small ones (kernel-launch bound).
            ft, mt, spatial_shapes, _ = self._encode_frame(
                imgs.reshape(B * T, N, *imgs.shape[3:]), world2img.reshape(B * T, N, 4, 4),
                None if intrinsics is None else intrinsics.reshape(B * T, N, 4, 4),
                camera_valid=None if camera_valid is None else camera_valid.reshape(B * T, N),
                active_index=_host_index(camera_valid_host, imgs.device),
                # (rows must follow the host-indexed view selection; none with a device-only mask)
                crop_rows=(_active_rows(image_hw_host, camera_valid_host, T)
                           if camera_valid is None or camera_valid_host is not None else None),
            )
            feats = ft.view(B, T, *ft.shape[1:])   # (B, T, N*sum_L, C)
            mlns = mt.view(B, T, *mt.shape[1:])    # (B, T, N*sum_L, 14)

        cls_scores, kpts_norm, dn_meta = self.head(
            feats, mlns, world2img, spatial_shapes, (H, W),
            gt_keypoints_norm, gt_valid, gt_keypoint_valid,
            camera_valid=camera_valid,
            images=imgs if self.head.external_reid is not None else None,
            track_targets=track_targets,
            stream_reset=stream_reset,
        )
        self.aux2d_output = None
        if self.aux2d is not None and self.training and track_targets is not None and self._aux_feat is not None:
            loss, num_pos = aux2d_loss(
                self.aux2d, self._aux_feat, 8, self.cfg.model.aux2d_sigma_cells,
                gt_mm=self.denormalize_keypoints(track_targets["gt"]),
                keypoint_valid=track_targets["keypoint_valid"], frame_valid=track_targets["frame_valid"],
                world2img=world2img, camera_valid=camera_valid,
                image_hw=None if image_hw_host is None else image_hw_host.to(imgs.device, non_blocking=True),
                image_size=(H, W), sigma_mm=track_targets.get("sigma_mm"),
                kp2d=track_targets.get("kp2d"), kp2d_valid=track_targets.get("kp2d_valid"))
            self.aux2d_output = {"loss": loss, "num_pos": num_pos}
        self._aux_feat = None
        return cls_scores, kpts_norm, dn_meta, dets_2d if return_2d else None

    # ------------------------------------------------------------------ #
    def denormalize_keypoints(self, kpts_norm):
        pr = self.position_range
        return kpts_norm * (pr[3:6] - pr[:3]) + pr[:3]

    def normalize_keypoints(self, kpts_mm):
        pr = self.position_range
        return (kpts_mm - pr[:3]) / (pr[3:6] - pr[:3])