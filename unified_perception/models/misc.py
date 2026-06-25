"""Small shared model utilities (ported/adapted from StreamPETR)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def inverse_sigmoid(x, eps: float = 1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pos2posemb3d(pos, num_pos_feats: int = 128, temperature: int = 10000):
    """Sinusoidal embedding of normalised 3D positions (last dim 3).

    Returns a tensor whose last dim is ``3 * num_pos_feats``.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x, pos_z), dim=-1)


def pos2posemb1d(pos, num_pos_feats: int = 256, temperature: int = 10000):
    """Sinusoidal embedding of a scalar (last dim 1).

    Returns a tensor whose last dim is ``num_pos_feats``.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return pos_x



class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU activations."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers: list = []
        for i in range(num_layers):
            i_dim = in_dim if i == 0 else hidden_dim
            o_dim = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(i_dim, o_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLN(nn.Module):
    """Modulated LayerNorm (from StreamPETR).

    Conditions a feature via a latent code ``c``: the code predicts an affine
    (gamma, beta) that modulates a (non-affine) LayerNorm of the feature.  Used
    here for spatial alignment -- injecting per-view camera calibration into the
    image tokens.

    Args:
        c_dim: dimension of the conditioning code ``c``.
        f_dim: feature dimension being modulated.
    """

    def __init__(self, c_dim: int, f_dim: int = 256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.reduce = nn.Sequential(nn.Linear(c_dim, f_dim), nn.ReLU())
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        return self.gamma(c) * x + self.beta(c)


def memory_refresh(memory, prev_exist):
    """Zero out memory for batch elements whose sequence just (re)started.

    ``prev_exist`` is a (B,) tensor of 0/1; broadcast over the remaining dims.
    """
    view_shape = [1 for _ in range(memory.dim())]
    prev_exist = prev_exist.view(-1, *view_shape[1:])
    return memory * prev_exist


def topk_gather(feat, topk_indexes):
    """Gather along dim=1 using indices shaped (B, k, ...)."""
    if topk_indexes is None:
        return feat
    feat_shape = feat.shape
    topk_shape = topk_indexes.shape
    view_shape = [1 for _ in range(len(feat_shape))]
    view_shape[:2] = topk_shape[:2]
    topk_indexes = topk_indexes.view(*view_shape)
    return torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))