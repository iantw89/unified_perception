"""3D position encoding (PETR-style), adapted for arbitrary camera calibration.

Core idea (from PETR / StreamPETR): each image feature cell corresponds to a
ray in 3D.  We sample ``depth_num`` points along that ray, un-project them to
world coordinates using the inverse projection matrix (img2world), normalise by
the capture-volume ``position_range``, and pass the resulting per-cell 3D point
cloud through an MLP to obtain a position embedding that is added to the image
tokens.  This injects 3D, calibration-aware geometry into the 2D features.

Unlike StreamPETR (which works in the LiDAR/ego frame), the human-pose world
frame is static, so no ego-motion handling is needed here -- the calibration
alone defines each ray.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .misc import MLP, inverse_sigmoid


class CameraPositionEncoder3D(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        position_range: List[float],
        depth_num: int = 64,
        depth_start_mm: float = 500.0,
        depth_max_mm: float = 8000.0,
        lid: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.depth_num = depth_num
        self.register_buffer(
            "position_range", torch.tensor(position_range, dtype=torch.float32)
        )

        if lid:
            index = torch.arange(depth_num, dtype=torch.float32)
            index_1 = index + 1
            bin_size = (depth_max_mm - depth_start_mm) / (depth_num * (1 + depth_num))
            coords_d = depth_start_mm + bin_size * index * index_1
        else:
            index = torch.arange(depth_num, dtype=torch.float32)
            bin_size = (depth_max_mm - depth_start_mm) / depth_num
            coords_d = depth_start_mm + bin_size * index
        self.register_buffer("coords_d", coords_d)  # (D,)

        self.position_encoder = MLP(
            in_dim=depth_num * 3,
            hidden_dim=embed_dims * 4,
            out_dim=embed_dims,
            num_layers=2,
        )

    def forward(
        self,
        img2world: torch.Tensor,   # (B, N, 4, 4)
        feat_hw: Tuple[int, int],  # (Hp, Wp)
        image_hw: Tuple[int, int], # (H, W) the resolution intrinsics refer to
    ) -> torch.Tensor:
        """Return position embeddings (B, N, Hp*Wp, embed_dims)."""
        B, N = img2world.shape[:2]
        Hp, Wp = feat_hw
        H, W = image_hw
        D = self.depth_num
        device = img2world.device
        dtype = img2world.dtype

        # Pixel-centre coordinates for each feature cell, in image resolution.
        ys = (torch.arange(Hp, device=device, dtype=dtype) + 0.5) / Hp * H
        xs = (torch.arange(Wp, device=device, dtype=dtype) + 0.5) / Wp * W
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (Hp, Wp)
        uv = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (L, 2), L=Hp*Wp
        L = uv.shape[0]

        coords_d = self.coords_d.to(dtype)  # (D,)
        # Build homogeneous [u*d, v*d, d, 1] for every (cell, depth).
        uv_exp = uv.view(L, 1, 2).expand(L, D, 2)
        d_exp = coords_d.view(1, D, 1).expand(L, D, 1)
        coords = torch.cat([uv_exp * d_exp, d_exp, torch.ones(L, D, 1, device=device, dtype=dtype)], dim=-1)
        # (L, D, 4)

        # Un-project per camera: world = img2world @ coords.
        coords = coords.view(1, 1, L, D, 4, 1)
        m = img2world.view(B, N, 1, 1, 4, 4)
        world = torch.matmul(m, coords).squeeze(-1)[..., :3]  # (B, N, L, D, 3)

        pr = self.position_range
        world = (world - pr[:3]) / (pr[3:6] - pr[:3])  # normalise to [0,1]
        feat = inverse_sigmoid(world.clamp(0, 1)).reshape(B, N, L, D * 3)
        pos_embed = self.position_encoder(feat)  # (B, N, L, embed_dims)
        return pos_embed