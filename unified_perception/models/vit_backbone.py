"""DINOv3 ViT (self-contained; loads Meta's official checkpoints) + ViTDet SimpleFPN.

The ViT re-implements DINOv3 inference as in ``facebookresearch/dinov3`` / Hugging Face
``DINOv3ViTModel`` (verified against the latter in tests/test_vit_backbone.py): patch embedding,
cls + 4 storage (register) tokens, 2-D RoPE on patch tokens only (patch-centre coordinates in
[-1, 1], periods ``base ** (4 i / head_dim)``, base 100), pre-norm blocks with LayerScale, no
key bias, final LayerNorm.  Official ``dinov3_vit*16_pretrain_*.pth`` state dicts load directly.
At 640 x 640 the stride-16 grid is exactly 40 x 40, so projected 3-D points need no correction.

Frozen blocks run as a cached low-precision copy with autocast off -- identical numerics in
training and evaluation whether or not the caller is under autocast -- in chunks, without
autograd.  The SimpleFPN (Li et al., "Exploring Plain Vision Transformer Backbones for Object
Detection") turns the single stride-16 map into strides 8 / 16 / 32 (optionally + 64; the YOLO
P3-P6 levels it replaces) with ``out_channels`` each, and is trained.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: (dim, depth, heads, storage tokens) of the official LVD-1689M ViTs with a plain MLP.
VIT_ARCHS = {"vits16": (384, 12, 6, 4), "vitb16": (768, 12, 12, 4), "vitl16": (1024, 24, 16, 4)}


def rope_tables(h: int, w: int, head_dim: int, base: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin), each (h*w, head_dim): DINOv3 2-D RoPE for an h x w patch grid (inference: no
    coordinate augmentation)."""
    ys = (torch.arange(0.5, h, device=device, dtype=torch.float32) / h) * 2 - 1
    xs = (torch.arange(0.5, w, device=device, dtype=torch.float32) / w) * 2 - 1
    coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1).flatten(0, 1)        # (hw, 2) y, x
    inv_freq = 1.0 / base ** torch.arange(0, 1, 4 / head_dim, device=device, dtype=torch.float32)
    angles = (2 * math.pi * coords[:, :, None] * inv_freq).flatten(1).tile(2)            # (hw, head_dim)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)          # key third of the bias is zero (no key bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        P = cos.shape[0]                              # patch tokens are the last P
        qp, kp = q[..., N - P:, :], k[..., N - P:, :]
        q = torch.cat([q[..., :N - P, :], qp * cos + _rotate_half(qp) * sin], dim=-2)
        k = torch.cat([k[..., :N - P, :], kp * cos + _rotate_half(kp) * sin], dim=-2)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = _Attention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = _Mlp(dim, 4 * dim)
        self.ls2 = _LayerScale(dim)

    def forward(self, x, cos, sin):
        x = x + self.ls1(self.attn(self.norm1(x), cos, sin))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, patch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)


class DinoV3ViT(nn.Module):
    """DINOv3 ViT returning the final-norm patch-token grid (B, dim, H/16, W/16)."""

    def __init__(self, arch: str = "vitb16", patch: int = 16, rope_base: float = 100.0, eps: float = 1e-5):
        super().__init__()
        dim, depth, heads, storage = VIT_ARCHS[arch]
        self.patch, self.dim, self.heads, self.num_prefix = patch, dim, heads, 1 + storage
        self.rope_base = rope_base
        self.patch_embed = _PatchEmbed(patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, storage, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))      # unused at inference; keeps keys
        self.blocks = nn.ModuleList(_Block(dim, heads, eps) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=eps)

    def load_official(self, path: str) -> None:
        """Load an official ``dinov3_*_pretrain_*.pth``: drops the RoPE period and qkv bias-mask
        buffers and zeroes the key bias (the released models have none; HF conversion likewise)."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.endswith(("bias_mask", "rope_embed.periods"))}
        for k in [k for k in state if k.endswith("attn.qkv.bias")]:
            bias = state[k].clone()
            bias[self.dim:2 * self.dim] = 0
            state[k] = bias
        self.load_state_dict(state, strict=True)

    def rope(self, h: int, w: int, device, dtype):
        return rope_tables(h, w, self.dim // self.heads, self.rope_base, device, dtype)

    def tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.patch_embed.proj(x)
        B, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        prefix = torch.cat([self.cls_token, self.storage_tokens], 1).to(x.dtype).expand(B, -1, -1)
        return torch.cat([prefix, x], 1), h, w

    def grid(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.norm(x)[:, self.num_prefix:]
        return x.transpose(1, 2).reshape(x.shape[0], self.dim, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h, w = self.tokens(x)
        cos, sin = self.rope(h, w, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.grid(x, h, w)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), x.shape[1:2], self.weight, self.bias,
                            self.eps).permute(0, 3, 1, 2)


def _conv_norm(cin: int, cout: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), LayerNorm2d(cout))


class SimpleFPN(nn.Module):
    """ViTDet simple feature pyramid: scales 2, 1, 1/2 of the last ViT map (deconvolution /
    identity / max-pool, each followed by 1x1 + 3x3 conv with LayerNorm), plus, with
    ``num_levels=4``, a 1/4 level max-pooled from the 1/2 output (detectron2 LastLevelMaxPool).
    At 640 px the 1/4 level would be 10 x 10 (stride 64): too coarse to help, so 3 by default."""

    def __init__(self, in_dim: int, out_channels: int, num_levels: int = 3):
        super().__init__()
        if num_levels not in (3, 4):
            raise ValueError("SimpleFPN num_levels must be 3 (strides 8/16/32) or 4 (+64)")
        self.num_levels = num_levels
        self.up = nn.ConvTranspose2d(in_dim, in_dim // 2, 2, 2)
        dims = (in_dim // 2, in_dim, in_dim)
        self.lateral = nn.ModuleList(_conv_norm(d, out_channels, 1) for d in dims)
        self.output = nn.ModuleList(_conv_norm(out_channels, out_channels, 3) for _ in dims)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        inputs = (self.up(x), x, F.max_pool2d(x, 2, 2))
        levels = [out(lat(f)) for f, lat, out in zip(inputs, self.lateral, self.output)]
        if self.num_levels == 4:
            levels.append(F.max_pool2d(levels[-1], 1, 2))
        return levels


def fpn_levels_for(image_height: int, image_width: int) -> int:
    """SimpleFPN levels for an input size: strides 8/16/32 below 1024 px, + stride 64 from 1024 px
    (at 1280 px the stride-64 map is 20 x 20 and covers people that fill much of the view)."""
    return 4 if max(image_height, image_width) >= 1024 else 3


def _frozen_forward(trunk: DinoV3ViT, apply_norm: bool):
    """Embedding + frozen blocks (+ final norm and grid when the whole ViT is frozen)."""
    def run(x, cos, sin):
        x, h, w = trunk.tokens(x)
        for block in trunk.blocks:
            x = block(x, cos, sin)
        return trunk.grid(x, h, w) if apply_norm else x
    return run


class DinoV3SimpleFPNBackbone(nn.Module):
    """Drop-in replacement for YOLOPoseBackbone: ``forward(images) -> (levels, None)``."""

    #: images per frozen-ViT call (activations of one chunk alive at a time).
    chunk_images: int = 32
    #: forward(..., input_mean=, input_std=) normalizes uint8 input itself.
    normalizes_uint8: bool = True

    def __init__(self, weights: str | None, arch: str = "vitb16", out_channels: int = 256,
                 trainable_blocks: int = 0, frozen_dtype: torch.dtype = torch.bfloat16,
                 compile_trunk: bool = False, fpn_levels: int = 3):
        super().__init__()
        self.vit = DinoV3ViT(arch)
        if weights:
            self.vit.load_official(weights)
        depth = len(self.vit.blocks)
        if not 0 <= trainable_blocks <= depth:
            raise ValueError(f"trainable_blocks must be in [0, {depth}]")
        self.trainable_blocks, self.frozen_dtype = trainable_blocks, frozen_dtype
        self.num_frozen = depth - trainable_blocks
        self.vit.requires_grad_(False)
        if trainable_blocks:
            self.vit.blocks[self.num_frozen:].requires_grad_(True)
            self.vit.norm.requires_grad_(True)
        self.fpn = SimpleFPN(self.vit.dim, out_channels, fpn_levels)
        self.register_buffer("vit_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("vit_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.compile_trunk = compile_trunk
        # Frozen part as a low-precision copy kept outside the module tree (checkpoints keep the
        # fp32 keys); rebuilt on device or weight changes.
        self.__dict__["_trunk"] = None
        self._register_load_state_dict_pre_hook(self._drop_trunk)

    def _drop_trunk(self, *args, **kwargs):
        self.__dict__["_trunk"] = None

    def _apply(self, fn, *args, **kwargs):
        self.__dict__["_trunk"] = None
        return super()._apply(fn, *args, **kwargs)

    def _frozen_trunk(self):
        if self.__dict__["_trunk"] is None:
            trunk = copy.deepcopy(self.vit).to(self.frozen_dtype).eval().requires_grad_(False)
            trunk.blocks = trunk.blocks[:self.num_frozen]
            fn = _frozen_forward(trunk, apply_norm=self.trainable_blocks == 0)
            compiled = None
            if self.compile_trunk:
                # Compute-bound (large GEMMs): compile fuses norms/RoPE/residuals (~8% on a 4090);
                # dynamic batch so the variable last chunk does not recompile.  CUDA graphs bring
                # nothing here (~150 launches per chunk).  CUDA inputs only (CPU inductor needs a
                # C++ toolchain; e.g. the detector's construction-time channel probe runs on CPU).
                compiled = torch.compile(fn, dynamic=True, mode="max-autotune-no-cudagraphs")
            self.__dict__["_trunk"] = (trunk, fn, compiled)
        return self.__dict__["_trunk"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self

    def _vit_input(self, images: torch.Tensor, input_mean, input_std) -> torch.Tensor:
        if images.dtype == torch.uint8:
            if input_mean is None:
                raise ValueError("uint8 images need input_mean / input_std")
            images = (images.float() - input_mean) / input_std            # -> [0, 1] per data config
        images = (images.float() - self.vit_mean) / self.vit_std
        H, W = images.shape[-2:]
        p = self.vit.patch
        size = (max(p, round(H / p) * p), max(p, round(W / p) * p))
        if size != (H, W):          # grid must cover the image exactly (640 needs nothing)
            images = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
        return images.to(self.frozen_dtype)

    def _vit_features(self, images: torch.Tensor, input_mean, input_std) -> torch.Tensor:
        trunk, eager, compiled = self._frozen_trunk()
        fn = compiled if compiled is not None and images.is_cuda else eager
        outs, rope = [], None
        with torch.no_grad(), torch.autocast(images.device.type, enabled=False):
            for i in range(0, images.shape[0], self.chunk_images):
                x = self._vit_input(images[i:i + self.chunk_images], input_mean, input_std)
                h, w = x.shape[-2] // self.vit.patch, x.shape[-1] // self.vit.patch
                if rope is None:
                    rope = trunk.rope(h, w, x.device, x.dtype)
                outs.append(fn(x, *rope))
        if self.trainable_blocks == 0:
            return torch.cat(outs, 0)
        amp = torch.is_autocast_enabled(images.device.type)
        dtype = torch.get_autocast_dtype(images.device.type) if amp else torch.float32
        x = torch.cat(outs, 0).to(dtype)
        cos, sin = self.vit.rope(h, w, x.device, dtype)
        for block in self.vit.blocks[self.num_frozen:]:
            x = block(x, cos, sin)
        return self.vit.grid(x, h, w)

    def forward(self, x: torch.Tensor, return_detections: bool = False, conf_thresh: float = 0.25,
                input_mean: torch.Tensor | None = None, input_std: torch.Tensor | None = None):
        feats = self._vit_features(x, input_mean, input_std)
        amp = torch.is_autocast_enabled(feats.device.type)
        feats = feats.to(torch.get_autocast_dtype(feats.device.type) if amp else torch.float32)
        return self.fpn(feats), None
