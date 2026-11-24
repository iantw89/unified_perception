"""Parity and speed of the deformable-attention kernels in Detr3DCrossAttn (model.msda_impl).

    python -m unified_perception.tools.bench_msda [--lanes 12] [--iters 50]

One decoder cross-attention layer at the training shapes (7 cameras, 88 queries, 256 channels in
8 groups, 13 points, strides 8/16/32 of a 480 x 640 canvas), identical inputs and parameters for
every implementation: output and gradient differences against ``mmcv`` (fp32), then forward +
backward time.  ``flash`` needs DCNv4_op (OpenGVLab/DCNv4) importable.
"""

from __future__ import annotations

import argparse
import copy

import torch

from ..models import transformer as T


def build(lanes: int, cams: int, queries: int, dims: int, groups: int, points: int, shapes, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    layer = T.Detr3DCrossAttn(dims, cams, len(shapes), num_pts=points, num_groups=groups, dropout=0.0,
                              use_mmcv_msda=True)
    with torch.no_grad():   # non-trivial offsets and weights (the module initializes them to ~0)
        layer.learnable_fc.weight.normal_(0, 2.0, generator=g)
        layer.weights_fc.weight.normal_(0, 0.05, generator=g)
    tokens = sum(h * w for h, w in shapes)
    pc_range = torch.tensor([-8500.0, -14000.0, -200.0, 8500.0, 14000.0, 3000.0])
    # Cameras on a ring around the volume, looking at its centre (some points fall outside views).
    w2i = []
    for n in range(cams):
        a = torch.tensor(2 * torch.pi * n / cams)
        eye = torch.tensor([9000 * torch.cos(a), 9000 * torch.sin(a), 2500.0])
        f = -eye / eye.norm()
        r = torch.linalg.cross(f, torch.tensor([0.0, 0.0, 1.0]))
        r = r / r.norm()
        u = torch.linalg.cross(r, f)
        R = torch.stack([r, -u, f])
        K = torch.tensor([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
        m = torch.eye(4)
        m[:3, :4] = K @ torch.cat([R, (-R @ eye)[:, None]], 1)
        w2i.append(m)
    inputs = {
        "query": torch.randn(lanes, queries, dims, generator=g),
        "query_pos": torch.randn(lanes, queries, dims, generator=g),
        "feats": torch.randn(lanes * cams, tokens, dims, generator=g),
        "reference_points": 0.4 + 0.2 * torch.rand(lanes, queries, 3, generator=g),
        "pc_range": pc_range,
        "world2img": torch.stack(w2i)[None].expand(lanes, cams, 4, 4).contiguous(),
    }
    return layer, inputs


def run(layer, inputs, shapes, impl: str, backward: bool = True):
    layer.flash_msda = impl == "flash"
    x = {k: (v.detach().clone().requires_grad_(k in ("query", "feats")) if v.is_floating_point() else v)
         for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x["query"], x["query_pos"], x["feats"], x["reference_points"], x["pc_range"],
                    x["world2img"], shapes, (480, 640))
    if backward:
        out.float().square().mean().backward()
    return out, x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lanes", type=int, default=12)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    shapes = [(60, 80), (30, 40), (15, 20)]
    impls = ["mmcv"] + (["flash"] if T.FLASH_MSDA_AVAILABLE else [])
    if not T.FLASH_MSDA_AVAILABLE:
        print("flash: DCNv4 FlashDeformAttn not importable; mmcv only")
    layer, inputs = build(args.lanes, 7, 88, 256, 8, 13, shapes)
    layer = layer.cuda()
    inputs = {k: v.cuda() for k, v in inputs.items()}

    ref = None
    for impl in impls:
        torch.manual_seed(0)
        model = copy.deepcopy(layer)
        out, x = run(model, inputs, shapes, impl)
        grads = {"query": x["query"].grad, "feats": x["feats"].grad,
                 "weights_fc": model.weights_fc.weight.grad, "learnable_fc": model.learnable_fc.weight.grad}
        if ref is None:
            ref = (out.detach().float(), {k: v.float() for k, v in grads.items()})
        else:
            def rel(a, b):
                return float((a.float() - b).norm() / b.norm().clamp(min=1e-12))
            print(f"{impl} vs mmcv: output rel err {rel(out.detach(), ref[0]):.2e}  "
                  + "  ".join(f"grad[{k}] {rel(v, ref[1][k]):.2e}" for k, v in grads.items()))
        for _ in range(5):
            run(model, inputs, shapes, impl)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for label, backward in (("fwd", False), ("fwd+bwd", True)):
            start.record()
            for _ in range(args.iters):
                if backward:
                    run(model, inputs, shapes, impl)
                else:
                    with torch.no_grad():
                        run(model, inputs, shapes, impl, backward=False)
            end.record()
            torch.cuda.synchronize()
            print(f"{impl:6s} {label:8s} {start.elapsed_time(end) / args.iters:7.2f} ms per layer call")


if __name__ == "__main__":
    main()
