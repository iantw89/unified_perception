"""Triton multi-scale deformable attention matches mmcv's kernel (outputs and all gradients)."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_triton_msda_matches_mmcv():
    mmcv_ops = pytest.importorskip("mmcv.ops.multi_scale_deform_attn")
    from unified_perception.models.msda_triton import MSDATritonFunction
    torch.manual_seed(0)
    Bv, Q, G, D, L, P = 6, 37, 4, 16, 3, 5
    shapes = torch.tensor([[12, 16], [6, 8], [3, 4]], device="cuda")
    starts = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
    S = int((shapes[:, 0] * shapes[:, 1]).sum())
    v0 = torch.randn(Bv, S, G, D, device="cuda")
    loc0 = torch.rand(Bv, Q, G, L, P, 2, device="cuda") * 1.4 - 0.2      # some points off the map
    w0 = torch.rand(Bv, Q, G, L, P, device="cuda")

    def run(fn, *extra):
        v, loc, w = (t.clone().requires_grad_() for t in (v0, loc0, w0))
        out = fn(v, shapes, starts, loc, w, *extra)
        (out * torch.linspace(-1, 1, out.shape[-1], device="cuda")).sum().backward()
        return out.detach(), v.grad, loc.grad, w.grad

    ref = run(mmcv_ops.MultiScaleDeformableAttnFunction.apply, Bv)
    got = run(MSDATritonFunction.apply)
    for name, a, b in zip(("out", "dvalue", "dloc", "dattn"), got, ref):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), name
