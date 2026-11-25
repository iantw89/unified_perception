"""Frozen DINOv3 trunk alone: images/s and achieved TFLOP/s on the GPU, to compare with the
hardware tensor-pipe activity (nvidia-smi dmon --gpm-metrics 5) sampled while it runs.

    python -m unified_perception.tools.bench_trunk CONFIG [--images 576] [--height 480] [--seconds 30]

FLOPs are counted analytically per block (2 per multiply-add): qkv 3d^2, projection d^2,
MLP 8d^2 per token, plus attention 2 * 2 * n^2 * d per image (QK^T and AV).
"""

from __future__ import annotations

import argparse
import time

import torch

from ..config import load_config
from ..models import UnifiedPerceptionModel

H100_BF16_DENSE_TFLOPS = 989.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--images", type=int, default=576, help="images per call (batch 12 x 8 frames x 6 views)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    model = UnifiedPerceptionModel(cfg).cuda().eval()
    bb = model.backbone
    imgs = torch.randint(0, 255, (args.images, 3, args.height, args.width), dtype=torch.uint8, device="cuda")
    mean, std = model.input_mean.view(1, 3, 1, 1), model.input_std.view(1, 3, 1, 1)
    for _ in range(3):
        bb._frozen_features(imgs, mean, std)
    torch.cuda.synchronize()
    calls, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        bb._frozen_features(imgs, mean, std)
        calls += 1
        if calls % 4 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    vit = bb.vit
    d, depth, p = vit.dim, bb.num_frozen, vit.patch
    n = (args.height // p) * (args.width // p) + vit.num_prefix
    flops_image = depth * (2 * 12 * d * d * n + 4 * n * n * d)
    tflops = flops_image * args.images * calls / wall / 1e12
    print(f"[trunk] {args.images} x {args.height}x{args.width}, {calls} calls in {wall:.1f} s: "
          f"{args.images * calls / wall:.0f} images/s, {tflops:.0f} TFLOP/s "
          f"({tflops / H100_BF16_DENSE_TFLOPS:.0%} of H100 bf16 dense), fp8={bb.fp8}")


if __name__ == "__main__":
    main()
