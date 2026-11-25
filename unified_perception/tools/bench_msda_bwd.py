"""Triton deformable-attention backward variants vs mmcv: parity and kernel time at training shapes.

    python -m unified_perception.tools.bench_msda_bwd [--images 84] [--queries 88] [--iters 50]

Variants: scalar atomics (the default), vector atomics (sm_90+), vector atomics + heads-fastest grid.
Shapes default to the cloud run (84 images per decoder call, 88 queries, 8 heads x 32 channels,
3 levels at 640 px, 13 points per level).
"""

from __future__ import annotations

import argparse

import torch

from ..models import msda_triton


def inputs(images: int, queries: int, device):
    G, D, P = 8, 32, 13
    shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], device=device)
    starts = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
    S = int((shapes[:, 0] * shapes[:, 1]).sum())
    L = shapes.shape[0]
    g = torch.Generator(device=device).manual_seed(0)
    value = torch.randn(images, S, G, D, device=device, generator=g).to(torch.bfloat16)
    # Sampling points cluster around people (real decoder): centres per query, small spread.
    centre = torch.rand(images, queries, 1, 1, 1, 2, device=device, generator=g) * 0.8 + 0.1
    loc = (centre + 0.03 * torch.randn(images, queries, G, L, P, 2, device=device, generator=g)).clamp(0, 1)
    attn = torch.rand(images, queries, G, L, P, device=device, generator=g).softmax(-1)
    return value, shapes, starts, loc, attn


def run(value, shapes, starts, loc, attn, grad, impl):
    v, l_, a = (t.detach().clone().requires_grad_(True) for t in (value, loc, attn))
    if impl == "mmcv":
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
        out = MultiScaleDeformableAttnFunction.apply(v.float(), shapes, starts, l_, a, v.shape[0])
    else:
        out = msda_triton.MSDATritonFunction.apply(v, shapes, starts, l_, a)
    out.backward(grad.to(out.dtype))
    return out.float(), v.grad.float(), l_.grad.float(), a.grad.float()


def timed(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[len(times) // 2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=int, default=84)
    ap.add_argument("--queries", type=int, default=88)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args(argv)
    device = torch.device("cuda")
    value, shapes, starts, loc, attn = inputs(args.images, args.queries, device)
    grad = torch.randn(args.images, args.queries, 8 * 32, device=device)
    ref = run(value, shapes, starts, loc, attn, grad, "mmcv")
    names = ("out", "grad_value", "grad_loc", "grad_attn")
    print(f"[msda] {torch.cuda.get_device_name()}  images={args.images} queries={args.queries}")
    t_mmcv = timed(lambda: run(value, shapes, starts, loc, attn, grad, "mmcv"), args.iters)
    print(f"[msda] mmcv                         fwd+bwd {t_mmcv:7.3f} ms")
    for vec, gfast in ((False, False), (True, False), (True, True)):
        msda_triton.BWD_VECTOR_ATOMICS, msda_triton.BWD_HEADS_FASTEST = vec, gfast
        got = run(value, shapes, starts, loc, attn, grad, "triton")
        errs = " ".join(f"{n} {((g - r).abs().max() / r.abs().max().clamp(min=1e-12)).item():.1e}"
                        for n, g, r in zip(names, got, ref))
        t = timed(lambda: run(value, shapes, starts, loc, attn, grad, "triton"), args.iters)
        print(f"[msda] triton vec={vec!s:5} heads_fastest={gfast!s:5} fwd+bwd {t:7.3f} ms   rel.max err: {errs}")


if __name__ == "__main__":
    main()
