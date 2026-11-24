"""Detr3D-style transformer decoder (no mmcv / custom CUDA ops).

Architecture matches StreamPETR / RepDETR3D's ``Detr3DTransformerDecoder`` +
``DeformableFeatureAggregationCuda``, adapted for our static world frame and
4-camera rig.

Key difference from the previous PETR-style implementation
-----------------------------------------------------------
PETR adds a dense 3D position embedding to **every** image token and then runs
full cross-attention (O(Q × N × L) quadratic).

Detr3D instead **projects** each query's 3D reference point into the 2D image
plane of every camera, samples image features bilinearly at those locations
(O(Q × N × P)), and uses learned attention weights to aggregate across cameras
and sampling points.  No 3D position encoding on image tokens is needed.

``Detr3DCrossAttn`` is a pure-PyTorch reimplementation of
``DeformableFeatureAggregationCuda`` (StreamPETR) using ``F.grid_sample``
instead of the mmcv CUDA kernel so that no mmdet/mmcv dependency is required.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction as MMCVMSDAfn,
    )
    MMCV_MSDA_AVAILABLE = True
except ImportError:
    MMCVMSDAfn = None
    MMCV_MSDA_AVAILABLE = False

# DCNv4 "Flash Deformable Attention" (OpenGVLab/DCNv4, DCNv4_op): fp16, softmax fused in the kernel.
# Optional; selected per model by ``model.msda_impl: flash`` (see Detr3DCrossAttn._forward_flash).
try:
    from DCNv4.functions.flash_deform_attn_func import FlashDeformAttnFunction as FlashMSDAfn
    FLASH_MSDA_AVAILABLE = True
except Exception:   # not installed, or an unsupported GPU (the module raises at import)
    FlashMSDAfn = None
    FLASH_MSDA_AVAILABLE = False

_SHAPE_CACHE: dict = {}


def msda_inputs(feats: torch.Tensor, spatial_shapes: list, groups: int) -> tuple:
    """MMCV MSDA inputs of one frame: float32 value (B*N, sum_L, G, C/G) and cached
    (S, 2) shapes / (S,) level starts (no per-call host-to-device copy, which syncs)."""
    key = (tuple(tuple(int(v) for v in hw) for hw in spatial_shapes), feats.device)
    cached = _SHAPE_CACHE.get(key)
    if cached is None:
        shapes = torch.tensor([list(hw) for hw in key[0]], dtype=torch.long, device=feats.device)
        starts = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
        cached = _SHAPE_CACHE[key] = (shapes, starts)
    BN, L, C = feats.shape
    return (feats.reshape(BN, L, groups, C // groups).float(),) + cached


class Detr3DCrossAttn(nn.Module):
    """3D-projection feature-sampling cross-attention (multi-scale).

    For each query with a 3D reference point:
    1. Predict ``num_pts`` 3D key points as offsets from the reference.
    2. Project every key point into every camera's image plane.
    3. Bilinearly sample the feature map at each scale level at those locations.
    4. Aggregate via attention weights conditioned on query content + camera
       geometry, using a global softmax over all N×S×P sampling locations
       (N cameras, S scales, P sampling points per camera).

    This is a pure-PyTorch equivalent of StreamPETR's
    ``DeformableFeatureAggregationCuda`` (single group, multi-scale).
    """

    def __init__(
        self,
        embed_dims:    int,
        num_cams:      int,
        num_levels:    int,
        num_pts:       int = 4,
        num_groups:    int = 1,
        dropout:       float = 0.1,
        use_mmcv_msda: bool = False,
    ):
        super().__init__()
        assert embed_dims % num_groups == 0, (
            f"embed_dims ({embed_dims}) must be divisible by num_groups ({num_groups})"
        )
        self.embed_dims    = embed_dims
        self.num_groups    = num_groups
        self.group_dims    = embed_dims // num_groups
        self.num_cams      = num_cams
        self.num_levels    = num_levels
        self.num_pts       = num_pts
        self.use_mmcv_msda = use_mmcv_msda
        self.flash_msda    = False   # set by the detector from model.msda_impl

        # Predict num_pts 3D offsets from query content; init near-zero.
        self.learnable_fc = nn.Linear(embed_dims, num_pts * 3)
        nn.init.zeros_(self.learnable_fc.weight)
        nn.init.uniform_(self.learnable_fc.bias, -0.01, 0.01)

        # Camera geometry embedding: first 3 rows of world2img (K@[R|t]).
        self.cam_embed = nn.Sequential(
            nn.Linear(12, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.LayerNorm(embed_dims),
        )

        # Predict num_groups * num_levels * num_pts weights per camera per query,
        # matching StreamPETR's DeformableFeatureAggregationCuda.  Softmax is taken
        # over all N * S * P locations per group (global cross-camera competition).
        self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * num_pts)
        nn.init.zeros_(self.weights_fc.weight)
        nn.init.zeros_(self.weights_fc.bias)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query:            torch.Tensor,  # (B, Q, C)
        query_pos:        torch.Tensor,  # (B, Q, C)
        feats:            torch.Tensor,  # (B*N, sum_L, C)  spatially-aligned
        reference_points: torch.Tensor,  # (B, Q, 3)    normalised [0, 1]
        pc_range:         torch.Tensor,  # (6,)          world-space bounds mm
        world2img:        torch.Tensor,  # (B, N, 4, 4)
        spatial_shapes:   list,          # [(Hp_0, Wp_0), (Hp_1, Wp_1), ...]
        image_hw:         Tuple[int, int],  # (H, W) original image pixels
        camera_valid:     Optional[torch.Tensor] = None,  # (B, N)
        msda:             Optional[tuple] = None,  # (value f32, shapes, level_start) per frame
    ) -> torch.Tensor:                   # (B, Q, C)
        B, Q, C = query.shape
        N       = world2img.shape[1]
        H,  W   = image_hw
        P       = self.num_pts
        S       = len(spatial_shapes)

        # ---- 3D key points -------------------------------------------------
        ref_mm  = reference_points * (pc_range[3:6] - pc_range[:3]) + pc_range[:3]
        offsets = self.learnable_fc(query).reshape(B, Q, P, 3)
        key_pts = ref_mm.unsqueeze(2) + offsets              # (B, Q, P, 3)

        # ---- Project to 2-D per camera -------------------------------------
        ones    = key_pts.new_ones(*key_pts.shape[:-1], 1)
        pts_h   = torch.cat([key_pts, ones], dim=-1)         # (B, Q, P, 4)
        pts_cam = torch.einsum("bnij,bqpj->bnqpi", world2img, pts_h)  # (B, N, Q, P, 4)

        depth   = pts_cam[..., 2].clamp(min=1e-5)
        u_n     = pts_cam[..., 0] / depth / W                # col in [0,1]
        v_n     = pts_cam[..., 1] / depth / H                # row in [0,1]

        valid   = (
            (pts_cam[..., 2] > 0)
            & (u_n > 0) & (u_n < 1)
            & (v_n > 0) & (v_n < 1)
        )  # (B, N, Q, P)
        if camera_valid is not None:
            valid = valid & camera_valid[:, :, None, None]

        # ---- Attention weights (B, Q, G, N*S*P) — StreamPETR DFA style ----
        G       = self.num_groups
        extr    = world2img[..., :3, :].flatten(-2)          # (B, N, 12)
        cam_emb = self.cam_embed(extr)                       # (B, N, C)
        feat_pos = (query + query_pos).unsqueeze(2) + cam_emb.unsqueeze(1)  # (B, Q, N, C)

        # weights_fc outputs G*S*P per camera → (B, Q, N, G*S*P).
        # Reshape to (B, Q, N*S*P, G): N outermost, G innermost — matching StreamPETR.
        # Softmax over N*S*P (global across cameras) independently per group.
        weights = self.weights_fc(feat_pos).reshape(B, Q, N * S * P, G)  # (B, Q, N*S*P, G)

        # Validity expanded across levels and groups (same 2D projection per level/group)
        # valid: (B, N, Q, P) → (B, N, S, Q, P) → (B, Q, N, S, P) → (B, Q, N*S*P, 1)
        valid_flat = (
            valid.unsqueeze(2)
                 .expand(B, N, S, Q, P)
                 .permute(0, 3, 1, 2, 4)
                 .reshape(B, Q, N * S * P, 1)
        )
        weights = weights.masked_fill(~valid_flat, float("-inf"))
        # No visible sampling points: avoid an all--inf softmax and NaN gradients.
        any_valid = valid_flat.any(dim=-2, keepdim=True)                 # (B, Q, 1, 1)
        weights = torch.where(any_valid, weights, 0.0)
        if self.flash_msda and FLASH_MSDA_AVAILABLE and feats.is_cuda:
            output = self._forward_flash(feats, u_n, v_n, weights, any_valid, spatial_shapes,
                                         B, N, Q, C, S, P, query.dtype)
            output = self.output_proj(output)
            return self.drop(output) + query
        weights = weights.softmax(dim=-2).masked_fill(~valid_flat, 0.0)
        weights = weights.permute(0, 1, 3, 2)                # (B, Q, G, N*S*P)

        # ---- Feature sampling (dispatch PyTorch or MMCV CUDA) --------------
        _use_mmcv = self.use_mmcv_msda and MMCV_MSDA_AVAILABLE and feats.is_cuda
        if _use_mmcv:
            output = self._forward_mmcv(
                feats, u_n, v_n,
                weights,
                spatial_shapes, B, N, Q, C, S, P, query.dtype, msda,
            )
        else:
            output = self._forward_pytorch(
                feats, u_n, v_n, weights, spatial_shapes, B, N, Q, C, S, P,
            )

        output = self.output_proj(output)
        return self.drop(output) + query

    # ---------------------------------------------------------------------- #
    def _forward_pytorch(
        self,
        feats:          torch.Tensor,   # (B*N, sum_L, C)
        u_n:            torch.Tensor,   # (B, N, Q, P)  in [0, 1]
        v_n:            torch.Tensor,   # (B, N, Q, P)  in [0, 1]
        weights:        torch.Tensor,   # (B, Q, G, N*S*P) per-group global-softmaxed
        spatial_shapes: list,
        B: int, N: int, Q: int, C: int, S: int, P: int,
    ) -> torch.Tensor:                  # (B, Q, C)
        """Pure-PyTorch F.grid_sample path (default, no extra dependencies)."""
        grid_x  = u_n * 2.0 - 1.0
        grid_y  = v_n * 2.0 - 1.0
        grid_bn = torch.stack([grid_x, grid_y], dim=-1).reshape(B * N, Q, P, 2).to(feats.dtype)

        level_start = 0
        sampled_levels: list[torch.Tensor] = []
        for Hp_l, Wp_l in spatial_shapes:
            L_l      = Hp_l * Wp_l
            feats_l  = feats[:, level_start : level_start + L_l, :]
            feats_2d = feats_l.view(B * N, Hp_l, Wp_l, C).permute(0, 3, 1, 2)
            samp_l   = F.grid_sample(
                feats_2d, grid_bn,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )                                                 # (B*N, C, Q, P)
            samp_l = samp_l.permute(0, 2, 3, 1).view(B, N, Q, P, C)
            sampled_levels.append(samp_l)
            level_start += L_l

        # (B, N, S, Q, P, C) → (B, Q, N, S, P, C) → (B, Q, N*S*P, C)
        sampled = (
            torch.stack(sampled_levels, dim=2)
                 .permute(0, 3, 1, 2, 4, 5)
                 .reshape(B, Q, N * S * P, C)
        )
        # Multi-group aggregation matching StreamPETR: split C into G groups,
        # apply per-group weights independently.
        # sampled (B, Q, N*S*P, C) → (B, Q, N*S*P, G, group_dims) → (B, Q, G, N*S*P, group_dims)
        G = self.num_groups
        sampled_g = sampled.view(B, Q, N * S * P, G, C // G).permute(0, 1, 3, 2, 4)
        # weights (B, Q, G, N*S*P) × sampled_g (B, Q, G, N*S*P, group_dims) → (B, Q, G, group_dims)
        return (weights.unsqueeze(-1) * sampled_g).sum(dim=-2).reshape(B, Q, C)

    def _forward_mmcv(
        self,
        feats:          torch.Tensor,   # (B*N, sum_L, C)
        u_n:            torch.Tensor,   # (B, N, Q, P)  in [0, 1]
        v_n:            torch.Tensor,   # (B, N, Q, P)  in [0, 1]
        weights:        torch.Tensor,   # (B, Q, G, N*S*P) per-group global-softmaxed
        spatial_shapes: list,
        B: int, N: int, Q: int, C: int, S: int, P: int,
        out_dtype: torch.dtype,
        msda: Optional[tuple] = None,
    ) -> torch.Tensor:                  # (B, Q, C)
        """MMCV CUDA MSDA kernel path.

        Reshapes our multi-camera tensors into the MMCV expected format
        (bs=B*N, num_heads=G, num_levels=S, num_points=P) then sums per-camera
        weighted outputs to recover the same result as _forward_pytorch.
        The global softmax over N*S*P has already been applied per group;
        summing per-camera MMCV outputs is mathematically equivalent to the
        group-weighted-sum done by the PyTorch path.
        """
        G          = self.num_groups
        group_dims = C // G
        device     = feats.device

        if msda is None:
            msda = msda_inputs(feats, spatial_shapes, G)
        # value: (B*N, sum_L, G, group_dims) float32 — shared by every layer of the frame
        # (one conversion, and autograd saves it once); shapes / level starts are cached.
        value, shapes_t, level_start_idx = msda

        # sampling_locations: (B*N, Q, G, S, P, 2) in [0, 1]
        # The same 2D location is used at every pyramid level and group.
        locs = torch.stack([u_n, v_n], dim=-1)              # (B, N, Q, P, 2)
        locs = (
            locs.reshape(B * N, Q, P, 2)                    # (B*N, Q, P, 2)
                [:, :, None, None, :, :]                    # (B*N, Q, 1, 1, P, 2)
                .expand(B * N, Q, G, S, P, 2)
                .contiguous()
                .float()
        )                                                    # (B*N, Q, G, S, P, 2)

        # attention_weights: (B*N, Q, G, S, P)
        # weights (B, Q, G, N*S*P) → view (B, Q, G, N, S, P)
        # → permute (B, N, Q, G, S, P) → reshape (B*N, Q, G, S, P)
        attn_w = (
            weights.view(B, Q, G, N, S, P)
            .permute(0, 3, 1, 2, 4, 5)                      # (B, N, Q, G, S, P)
            .contiguous()
            .view(B * N, Q, G, S, P)
            .float()
        )

        # im2col_step must divide B*N; fall back to 1 if it does not
        im2col_step = min(64, B * N)
        if (B * N) % im2col_step != 0:
            im2col_step = 1

        # Output: (B*N, Q, G*group_dims) = (B*N, Q, C)
        out = MMCVMSDAfn.apply(
            value, shapes_t, level_start_idx, locs, attn_w, im2col_step,
        )

        # Sum across cameras: (B*N, Q, C) → (B, N, Q, C) → (B, Q, C)
        return out.to(out_dtype).view(B, N, Q, C).sum(dim=1)

    def _forward_flash(
        self,
        feats:          torch.Tensor,   # (B*N, sum_L, C)
        u_n:            torch.Tensor,   # (B, N, Q, P)  in [0, 1] where valid
        v_n:            torch.Tensor,   # (B, N, Q, P)
        logits:         torch.Tensor,   # (B, Q, N*S*P, G) pre-softmax, -inf at invalid points
        any_valid:      torch.Tensor,   # (B, Q, 1, 1) the query sees at least one valid point
        spatial_shapes: list,
        B: int, N: int, Q: int, C: int, S: int, P: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:                  # (B, Q, C)
        """DCNv4 FlashDeformAttn path: one kernel call per frame, cameras as extra levels.

        Each camera's S pyramid levels become levels ``n * S + s`` of one (B, N*sum_L, G, C/G)
        value, so the kernel's fused softmax over levels x points is exactly the global softmax
        over N*S*P of the other paths, and the per-camera sum disappears.  Invalid points keep
        their -inf logit (weight 0); a query with no valid point gets uniform logits and every
        location moved off the image, so it samples zeros like the masked paths.  fp16 inside.
        """
        G = self.num_groups
        value = feats.reshape(B, N * feats.shape[1], G, C // G).half()
        key = (tuple(tuple(int(v) for v in hw) for hw in spatial_shapes), N, feats.device, "flash")
        cached = _SHAPE_CACHE.get(key)
        if cached is None:
            shapes = torch.tensor([list(hw) for hw in key[0]] * N, dtype=torch.long, device=feats.device)
            starts = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
            cached = _SHAPE_CACHE[key] = (shapes, starts)
        shapes_t, level_start = cached
        # Locations (B, Q, G, N, S, P, 2), the same per level and group; clamped so points behind a
        # camera (huge u, v) stay finite in fp16 (they are out of range either way).
        locs = torch.stack([u_n, v_n], dim=-1).clamp(-1.0, 2.0)          # (B, N, Q, P, 2)
        locs = torch.where(any_valid.view(B, 1, Q, 1, 1), locs, -1.0)
        locs = (locs.permute(0, 2, 1, 3, 4)[:, :, None, :, None]          # (B, Q, 1, N, 1, P, 2)
                    .expand(B, Q, G, N, S, P, 2).reshape(B, Q, G, N * S * P * 2))
        attn = logits.permute(0, 1, 3, 2)                                  # (B, Q, G, N*S*P)
        sampling_loc_attn = torch.cat([locs.half(), attn.half()], dim=-1).contiguous()
        out = FlashMSDAfn.apply(value, shapes_t, level_start, sampling_loc_attn, B, P)
        return out.to(out_dtype).reshape(B, Q, C)

class DecoderLayer(nn.Module):
    """Single Detr3D temporal decoder layer (post-norm, matching StreamPETR).

    Operation order:
        self-attn (current queries + streaming memory)  → norm
        Detr3D cross-attn (3D projection → sample)      → norm
        FFN                                              → norm
    """

    def __init__(
        self,
        dim:           int,
        num_heads:     int,
        ffn_dim:       int,
        num_cams:      int,
        num_levels:    int,
        num_pts:       int,
        dropout:       float,
        pc_range:      torch.Tensor,
        num_groups:    int = 1,
        use_mmcv_msda: bool = False,
    ):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = Detr3DCrossAttn(
            dim, num_cams=num_cams, num_levels=num_levels, num_pts=num_pts,
            num_groups=num_groups, dropout=dropout, use_mmcv_msda=use_mmcv_msda,
        )
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.norm3  = nn.LayerNorm(dim)
        self.ffn    = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pc_range", pc_range)

    def forward(
        self,
        query:            torch.Tensor,           # (B, Q, C)
        query_pos:        torch.Tensor,           # (B, Q, C)
        feats:            torch.Tensor,           # (B*N, sum_L, C)
        reference_points: torch.Tensor,           # (B, Q, 3)
        world2img:        torch.Tensor,           # (B, N, 4, 4)
        spatial_shapes:   list,                   # [(Hp_l, Wp_l), ...]
        image_hw:         Tuple[int, int],
        temp_memory:      Optional[torch.Tensor] = None,  # (B, M, C)
        temp_pos:         Optional[torch.Tensor] = None,  # (B, M, C)
        attn_mask:        Optional[torch.Tensor] = None,
        camera_valid:     Optional[torch.Tensor] = None,
        msda:             Optional[tuple] = None,
    ) -> torch.Tensor:
        def _wp(x, p): return x if p is None else x + p

        # Temporal self-attention
        q_sa = _wp(query, query_pos)
        if temp_memory is not None:
            k_sa = torch.cat([q_sa, _wp(temp_memory, temp_pos)], dim=1)
            v_sa = torch.cat([query, temp_memory], dim=1)
        else:
            k_sa, v_sa = q_sa, query
        q2    = self.self_attn(q_sa, k_sa, v_sa, attn_mask=attn_mask, need_weights=False)[0]
        query = self.norm1(query + self.dropout(q2))

        # Detr3D cross-attention
        q2    = self.cross_attn(
            query, query_pos, feats, reference_points,
            self.pc_range, world2img, spatial_shapes, image_hw,
            camera_valid=camera_valid, msda=msda,
        )
        query = self.norm2(query + self.dropout(q2))

        # FFN
        query = self.norm3(query + self.dropout(self.ffn(query)))
        return query


# ---------------------------------------------------------------------------
# Stacked decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """Stack of ``DecoderLayer``s; returns all intermediate outputs for aux losses."""

    def __init__(
        self,
        dim:           int,
        num_layers:    int,
        num_heads:     int,
        ffn_dim:       int,
        num_cams:      int,
        num_levels:    int,
        num_pts:       int,
        dropout:       float,
        pc_range:      torch.Tensor,
        num_groups:    int = 1,
        use_mmcv_msda: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim=dim, num_heads=num_heads, ffn_dim=ffn_dim,
                    num_cams=num_cams, num_levels=num_levels, num_pts=num_pts,
                    dropout=dropout, pc_range=pc_range,
                    num_groups=num_groups, use_mmcv_msda=use_mmcv_msda,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query:            torch.Tensor,           # (B, Q, C)
        query_pos:        torch.Tensor,           # (B, Q, C)
        feats:            torch.Tensor,           # (B*N, sum_L, C)
        reference_points: torch.Tensor,           # (B, Q, 3)
        world2img:        torch.Tensor,           # (B, N, 4, 4)
        spatial_shapes:   list,                   # [(Hp_l, Wp_l), ...]
        image_hw:         Tuple[int, int],
        temp_memory:      Optional[torch.Tensor] = None,
        temp_pos:         Optional[torch.Tensor] = None,
        attn_mask:        Optional[torch.Tensor] = None,
        camera_valid:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                            # (L, B, Q, C)
        intermediate = []
        cross = self.layers[0].cross_attn
        msda = None
        if cross.use_mmcv_msda and MMCV_MSDA_AVAILABLE and feats.is_cuda:
            msda = msda_inputs(feats, spatial_shapes, cross.num_groups)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            # Additive float mask built once per frame instead of in every layer's attention.
            mask_dtype = torch.get_autocast_dtype(query.device.type) if torch.is_autocast_enabled(query.device.type) else query.dtype
            attn_mask = torch.zeros(attn_mask.shape, dtype=mask_dtype, device=query.device).masked_fill_(
                attn_mask, float("-inf"))
        for layer in self.layers:
            query = layer(
                query, query_pos, feats, reference_points,
                world2img, spatial_shapes, image_hw,
                temp_memory=temp_memory, temp_pos=temp_pos,
                attn_mask=attn_mask,
                camera_valid=camera_valid,
                msda=msda,
            )
            intermediate.append(query)
        return torch.stack(intermediate, dim=0)