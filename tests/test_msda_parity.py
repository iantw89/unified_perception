"""Parity test: PyTorch F.grid_sample path vs MMCV CUDA MSDA kernel.

Verifies that ``Detr3DCrossAttn._forward_pytorch`` and
``Detr3DCrossAttn._forward_mmcv`` produce numerically equivalent outputs for
the same weights and inputs.

Usage
-----
Run standalone (prints pass/fail + error stats)::

    python tests/test_msda_parity.py

Or via pytest::

    pytest tests/test_msda_parity.py -v
"""
from __future__ import annotations

import math
import types

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mmcv_available() -> bool:
    try:
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction  # noqa
        return True
    except ImportError:
        return False


def _make_cameras(B: int, N: int, W: int, H: int, device: str) -> torch.Tensor:
    """Build (B, N, 4, 4) world2img matrices for N forward-facing cameras.

    Each camera is placed on a circle of radius 500 mm around the world origin,
    pointing in +Z.  The projection is:

        u_n = fx * (X - tx) / Z / W + 0.5
        v_n = fy * (Y - ty) / Z / H + 0.5

    where fx = W/2, fy = H/2, (tx, ty) is the camera XY position.
    Points near the world origin at Z ≈ 1000 mm project to (0.5, 0.5).
    """
    mats = []
    for cam in range(N):
        angle = 2 * math.pi * cam / N
        tx = math.cos(angle) * 500.0   # mm
        ty = math.sin(angle) * 500.0   # mm

        fx, fy = W / 2.0, H / 2.0
        cx, cy = W / 2.0, H / 2.0

        # world2img s.t. pts_cam = mat @ [X, Y, Z, 1]:
        #   pts_cam[0] = fx*(X - tx) + cx*Z
        #   pts_cam[2] = Z
        #   u_n = pts_cam[0] / Z / W = fx*(X-tx)/(Z*W) + cx/W
        mat = torch.zeros(4, 4, dtype=torch.float32, device=device)
        mat[0, 0] = fx          # fx * X
        mat[0, 2] = cx          # cx * Z
        mat[0, 3] = -fx * tx    # -fx * tx
        mat[1, 1] = fy
        mat[1, 2] = cy
        mat[1, 3] = -fy * ty
        mat[2, 2] = 1.0         # depth = Z
        mat[3, 3] = 1.0
        mats.append(mat)

    mat_n = torch.stack(mats, dim=0)                        # (N, 4, 4)
    return mat_n.unsqueeze(0).expand(B, -1, -1, -1).clone() # (B, N, 4, 4)


def _make_inputs(
    B: int = 2,
    N: int = 2,
    Q: int = 16,
    C: int = 64,
    W: int = 320,
    H: int = 320,
    spatial_shapes: list | None = None,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Return a dict of all inputs required by ``Detr3DCrossAttn.forward``."""
    torch.manual_seed(seed)
    if spatial_shapes is None:
        spatial_shapes = [(20, 20), (10, 10)]   # 2 feature pyramid levels

    S = len(spatial_shapes)
    sum_L = sum(h * w for h, w in spatial_shapes)

    query     = torch.randn(B, Q, C, device=device)
    query_pos = torch.randn(B, Q, C, device=device)
    feats     = torch.randn(B * N, sum_L, C, device=device)

    # Reference points: near world origin at Z ≈ 1000 mm → definitely valid
    # x_norm ∈ [0.45, 0.55] → X ∈ [-500, 500] mm  (pc_range x: -5000..5000)
    # y_norm ∈ [0.45, 0.55] → Y ∈ [-500, 500] mm
    # z_norm ∈ [0.55, 0.65] → Z ∈ [750, 1250] mm  (pc_range z: -2000..3000)
    ref_pts = (
        torch.rand(B, Q, 3, device=device) * 0.10
        + torch.tensor([0.45, 0.45, 0.55], device=device)
    )

    pc_range  = torch.tensor([-5000., -5000., -2000., 5000., 5000., 3000.], device=device)
    world2img = _make_cameras(B, N, W, H, device)
    image_hw  = (H, W)

    return dict(
        query=query, query_pos=query_pos, feats=feats,
        reference_points=ref_pts, pc_range=pc_range,
        world2img=world2img, spatial_shapes=spatial_shapes,
        image_hw=image_hw,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parity_full_layer(
    B: int = 2,
    N: int = 2,
    Q: int = 16,
    C: int = 64,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Full-layer parity: _forward_pytorch path vs _forward_msda path.

    Two Detr3DCrossAttn instances with identical weights receive the same
    input.  One uses ``_forward_pytorch`` (our custom multi-camera sampling),
    the other is through ``_forward_msda``.  Both must produce the same output.
    """
    from unified_perception.models.transformer import Detr3DCrossAttn

    spatial_shapes = [(20, 20), (10, 10)]
    S = len(spatial_shapes)
    P = 4

    # Both instances use the same weights; only the sampling path differs.
    attn_pt = Detr3DCrossAttn(
        embed_dims=C, num_cams=N, num_levels=S, num_pts=P,
        dropout=0.0, use_mmcv_msda=False,
    ).to(device).float().eval()

    attn_cuda = Detr3DCrossAttn(
        embed_dims=C, num_cams=N, num_levels=S, num_pts=P,
        dropout=0.0, use_mmcv_msda=True,
    ).to(device).float().eval()
    attn_cuda.load_state_dict(attn_pt.state_dict())

    inputs = _make_inputs(B=B, N=N, Q=Q, C=C, spatial_shapes=spatial_shapes, device=device)

    with torch.no_grad():
        out_pt  = attn_pt(**inputs)
        out_ref = attn_cuda(**inputs)

    max_err  = (out_pt - out_ref).abs().max().item()
    mean_err = (out_pt - out_ref).abs().mean().item()
    print(f"  [full layer]  max_abs={max_err:.2e}  mean_abs={mean_err:.2e}")
    torch.testing.assert_close(out_pt, out_ref, atol=atol, rtol=rtol,
                                msg="Full-layer outputs differ beyond tolerance")


def test_parity_sampling_kernel(
    B: int = 2,
    N: int = 2,
    Q: int = 16,
    C: int = 64,
    P: int = 4,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Kernel-level parity: _forward_pytorch vs _forward_msda.

    Constructs random feats / sampling locations / weights independently of
    the 3-D projection logic so any discrepancy is isolated to the sampling
    kernel and the multi-camera tensor adapter.
    """
    from unified_perception.models.transformer import Detr3DCrossAttn

    spatial_shapes = [(20, 20), (10, 10)]
    S = len(spatial_shapes)
    sum_L = sum(h * w for h, w in spatial_shapes)

    torch.manual_seed(0)
    feats = torch.randn(B * N, sum_L, C, device=device, dtype=torch.float32)

    # Sampling locations in (0.2, 0.8) — well inside the image
    u_n = torch.rand(B, N, Q, P, device=device) * 0.6 + 0.2
    v_n = torch.rand(B, N, Q, P, device=device) * 0.6 + 0.2

    # weights: (B, Q, G=1, N*S*P) with softmax over N*S*P per group
    w_raw = torch.rand(B, Q, N * S * P, device=device)
    weights = (w_raw / w_raw.sum(dim=-1, keepdim=True)).view(B, Q, 1, N * S * P)

    dummy = Detr3DCrossAttn(
        embed_dims=C, num_cams=N, num_levels=S, num_pts=P,
        dropout=0.0, use_mmcv_msda=False,
    ).to(device).float()

    out_pt  = dummy._forward_pytorch(feats, u_n, v_n, weights, spatial_shapes, B, N, Q, C, S, P)
    out_ref = dummy._forward_mmcv(feats, u_n, v_n, weights, spatial_shapes, B, N, Q, C, S, P, torch.float32)

    max_err  = (out_pt - out_ref).abs().max().item()
    mean_err = (out_pt - out_ref).abs().mean().item()
    print(f"  [kernel / ref]  max_abs={max_err:.2e}  mean_abs={mean_err:.2e}")
    torch.testing.assert_close(out_pt, out_ref, atol=atol, rtol=rtol,
                                msg="_forward_pytorch vs _forward_mmcv differ")


def test_fallback_when_no_mmcv() -> None:
    """The layer must not raise when use_mmcv_msda=True but MMCV is absent."""
    from unified_perception.models.transformer import Detr3DCrossAttn, MMCV_MSDA_AVAILABLE

    if MMCV_MSDA_AVAILABLE:
        print("  [fallback]  MMCV present — fallback not triggered (expected)")
        return

    device = "cpu"
    attn = Detr3DCrossAttn(
        embed_dims=32, num_cams=2, num_levels=2, num_pts=4,
        dropout=0.0, use_mmcv_msda=True,   # requested but unavailable
    ).to(device).float().eval()

    inputs = _make_inputs(B=1, N=2, Q=4, C=32, W=64, H=64,
                          spatial_shapes=[(8, 8), (4, 4)], device=device)
    with torch.no_grad():
        out = attn(**inputs)   # should silently fall back to PyTorch

    assert out.shape == (1, 4, 32), f"Unexpected shape {out.shape}"
    print("  [fallback]  correctly used PyTorch path when MMCV absent")


# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

def _time_kernel(
    fn,
    repeats: int = 200,
    warmup: int = 20,
) -> float:
    """Return mean wall-time in milliseconds for a CUDA kernel call."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats  # ms per call


def benchmark_execution_time(device: str = "cuda") -> None:
    """Compare _forward_pytorch vs _forward_mmcv for a range of shapes.

    Shapes chosen to span typical StreamPETR / Detr3D configurations:
      - Q:  small (100), medium (300), large (900) — dominant cost driver
      - C:  64 (ablation) and 256 (production)
      - N:  2 cameras (rig) and 6 cameras (surround-view)
      - S=4 FPN levels, P=4 sampling points, B=1

    Results are printed as a table with absolute timings and speedup ratio.
    """
    from unified_perception.models.transformer import (
        Detr3DCrossAttn, MMCV_MSDA_AVAILABLE,
    )

    if not MMCV_MSDA_AVAILABLE:
        print("  [benchmark]  MMCV CUDA ext not available — skipping")
        return

    # (B, N, Q, C, S, P)
    shapes = [
        (1,  2,  100, 64,  4, 4),
        (1,  2,  300, 64,  4, 4),
        (1,  2,  900, 64,  4, 4),
        (1,  6,  100, 64,  4, 4),
        (1,  6,  300, 64,  4, 4),
        (1,  6,  900, 64,  4, 4),
        (1,  2,  300, 256, 4, 4),
        (1,  6,  300, 256, 4, 4),
        (1,  6,  900, 256, 4, 4),
        (2,  6,  900, 256, 4, 4),
    ]

    # FPN sizes chosen so sum fits in memory and matches typical backbone output
    def make_spatial_shapes(S: int):
        base = 20
        return [(max(1, base >> s), max(1, base >> s)) for s in range(S)]

    col = "{:>6}"
    header = f"  {'B,N,Q,C,S,P':<22} | {col.format('PT(ms)')} | {col.format('MMCV(ms)')} | {col.format('Speedup')}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for (B, N, Q, C, S, P) in shapes:
        spatial_shapes = make_spatial_shapes(S)
        sum_L = sum(h * w for h, w in spatial_shapes)

        torch.manual_seed(0)
        feats       = torch.randn(B * N, sum_L, C, device=device, dtype=torch.float32)
        u_n         = torch.rand(B, N, Q, P, device=device) * 0.6 + 0.2
        v_n         = torch.rand(B, N, Q, P, device=device) * 0.6 + 0.2
        w_raw   = torch.rand(B, Q, N * S * P, device=device)
        weights = (w_raw / w_raw.sum(dim=-1, keepdim=True)).view(B, Q, 1, N * S * P)

        dummy = Detr3DCrossAttn(
            embed_dims=C, num_cams=N, num_levels=S, num_pts=P,
            dropout=0.0, use_mmcv_msda=False,
        ).to(device).float().eval()

        with torch.no_grad():
            t_pt   = _time_kernel(lambda: dummy._forward_pytorch(
                feats, u_n, v_n, weights, spatial_shapes, B, N, Q, C, S, P))
            t_mmcv = _time_kernel(lambda: dummy._forward_mmcv(
                feats, u_n, v_n, weights, spatial_shapes, B, N, Q, C, S, P, torch.float32))

        speedup = t_pt / t_mmcv
        shape_str = f"({B},{N},{Q:>3},{C:>3},{S},{P})"
        print(f"  {shape_str:<22} | {col.format(f'{t_pt:.3f}')} | {col.format(f'{t_mmcv:.3f}')} | {col.format(f'{speedup:.2f}x')}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    skip_cuda = not torch.cuda.is_available()

    print("=" * 60)
    print("MSDA parity test")
    print(f"  CUDA available : {not skip_cuda}")
    print(f"  MMCV available : {_mmcv_available()}")
    print("=" * 60)

    # --- Fallback (always runs, CPU-safe) ---
    print("\n[1] Fallback test (CPU, no-mmcv path)")
    test_fallback_when_no_mmcv()

    if skip_cuda:
        print("\nSkipping CUDA parity tests (no GPU)")
        return

    # --- Kernel-level parity (always runs; CUDA kernel test is optional inside) ---
    print("\n[2] Kernel-level parity (_forward_pytorch vs _forward_msda)")
    test_parity_sampling_kernel(device="cuda")
    print("    PASS")

    # --- Full-layer parity (always runs) ---
    print("\n[3] Full-layer parity (3D projection → sampling)")
    test_parity_full_layer(device="cuda")
    print("    PASS")

    # --- Timing benchmark ---
    # print("\n[4] Execution time benchmark (PyTorch grid_sample vs MMCV CUDA kernel)")
    # benchmark_execution_time(device="cuda")

    print("\nAll tests passed.\n")

if __name__ == "__main__":
    main()
