"""Multi-scale deformable attention in Triton (drop-in for mmcv's MultiScaleDeformableAttnFunction).

Same layout and sampling convention as mmcv / Deformable DETR:

    value               (Bv, S, G, D)   any float dtype (fp32 or bf16); S = sum of H_l * W_l
    spatial_shapes      (L, 2) long     (H_l, W_l)
    level_start_index   (L,) long
    sampling_locations  (Bv, Q, G, L, P, 2) in [0, 1], (x, y)
    attention_weights   (Bv, Q, G, L, P)
    -> output           (Bv, Q, G * D)   in value's dtype

One program per (image, block of queries, group) holds the D channels of its queries in registers
and loops over the L * P sampling points (unrolled: L and P are compile-time constants), gathering
the four bilinear corners as contiguous D-vectors and accumulating in fp32.  The backward recomputes
the sampling: gradients of locations and attention weights are reduced over D in registers (no
shared-memory reduction pass), and only the value gradient needs atomic adds (fp32 buffer).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _corner(value_ptr, base, hy, wx, H, W, g, G: tl.constexpr, D: tl.constexpr, d, qmask):
    ok = qmask & (hy >= 0) & (hy < H) & (wx >= 0) & (wx < W)
    row = base + hy * W + wx                                         # (BQ,) token index in image
    ptr = value_ptr + (row[:, None] * G + g) * D + d[None, :]
    v = tl.load(ptr, mask=ok[:, None], other=0.0).to(tl.float32)
    return v, ok, row


@triton.autotune(
    configs=[triton.Config({"BQ": bq}, num_warps=nw) for bq, nw in ((8, 4), (16, 4), (32, 4), (16, 8), (32, 8))],
    key=["Q", "G", "D", "L", "P"])
@triton.jit
def _msda_fwd(value_ptr, shapes_ptr, starts_ptr, loc_ptr, attn_ptr, out_ptr,
              S, Q, G: tl.constexpr, D: tl.constexpr, L: tl.constexpr, P: tl.constexpr, BQ: tl.constexpr):
    pid_q, g, bv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q = pid_q * BQ + tl.arange(0, BQ)
    qmask = q < Q
    d = tl.arange(0, D)
    acc = tl.zeros((BQ, D), dtype=tl.float32)
    qg = (bv * Q + q) * G + g                                        # (BQ,) flat (bv, q, g)
    for l in tl.static_range(L):
        H = tl.load(shapes_ptr + 2 * l)
        W = tl.load(shapes_ptr + 2 * l + 1)
        base = bv * S + tl.load(starts_ptr + l)
        for p in tl.static_range(P):
            lp = l * P + p
            x = tl.load(loc_ptr + (qg * (L * P) + lp) * 2, mask=qmask, other=-2.0)
            y = tl.load(loc_ptr + (qg * (L * P) + lp) * 2 + 1, mask=qmask, other=-2.0)
            a = tl.load(attn_ptr + qg * (L * P) + lp, mask=qmask, other=0.0)
            h_im = y * H - 0.5
            w_im = x * W - 0.5
            inside = qmask & (h_im > -1) & (w_im > -1) & (h_im < H) & (w_im < W)
            h0f = tl.floor(h_im)
            w0f = tl.floor(w_im)
            lh = h_im - h0f
            lw = w_im - w0f
            hh = 1 - lh
            hw = 1 - lw
            h0 = h0f.to(tl.int32)
            w0 = w0f.to(tl.int32)
            v1, _, _ = _corner(value_ptr, base, h0, w0, H, W, g, G, D, d, inside)
            v2, _, _ = _corner(value_ptr, base, h0, w0 + 1, H, W, g, G, D, d, inside)
            v3, _, _ = _corner(value_ptr, base, h0 + 1, w0, H, W, g, G, D, d, inside)
            v4, _, _ = _corner(value_ptr, base, h0 + 1, w0 + 1, H, W, g, G, D, d, inside)
            sampled = ((hh * hw)[:, None] * v1 + (hh * lw)[:, None] * v2
                       + (lh * hw)[:, None] * v3 + (lh * lw)[:, None] * v4)
            acc += a[:, None] * sampled
    out = out_ptr + ((bv * Q + q)[:, None] * G + g) * D + d[None, :]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=qmask[:, None])


# Tuned on a 4090 at the training shapes (84 images, 88 queries, 8 x 32, 3 x 13): BQ8/8 warps 4.1 ms,
# BQ32/2 warps 5.2 ms, the untuned BQ32/4 warps ~7.5 ms.  The value gradient is reset between trials.
@triton.autotune(
    configs=[triton.Config({"BQ": bq}, num_warps=nw) for bq, nw in ((8, 8), (32, 2), (8, 4), (16, 8))],
    key=["Q", "G", "D", "L", "P"], reset_to_zero=["gvalue_ptr"])
@triton.jit
def _msda_bwd(value_ptr, shapes_ptr, starts_ptr, loc_ptr, attn_ptr, gout_ptr,
              gvalue_ptr, gloc_ptr, gattn_ptr,
              S, Q, G: tl.constexpr, D: tl.constexpr, L: tl.constexpr, P: tl.constexpr, BQ: tl.constexpr):
    pid_q, g, bv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q = pid_q * BQ + tl.arange(0, BQ)
    qmask = q < Q
    d = tl.arange(0, D)
    qg = (bv * Q + q) * G + g
    go = tl.load(gout_ptr + ((bv * Q + q)[:, None] * G + g) * D + d[None, :],
                 mask=qmask[:, None], other=0.0).to(tl.float32)      # (BQ, D)
    for l in tl.static_range(L):
        H = tl.load(shapes_ptr + 2 * l)
        W = tl.load(shapes_ptr + 2 * l + 1)
        base = bv * S + tl.load(starts_ptr + l)
        for p in tl.static_range(P):
            lp = l * P + p
            x = tl.load(loc_ptr + (qg * (L * P) + lp) * 2, mask=qmask, other=-2.0)
            y = tl.load(loc_ptr + (qg * (L * P) + lp) * 2 + 1, mask=qmask, other=-2.0)
            a = tl.load(attn_ptr + qg * (L * P) + lp, mask=qmask, other=0.0)
            h_im = y * H - 0.5
            w_im = x * W - 0.5
            inside = qmask & (h_im > -1) & (w_im > -1) & (h_im < H) & (w_im < W)
            h0f = tl.floor(h_im)
            w0f = tl.floor(w_im)
            lh = h_im - h0f
            lw = w_im - w0f
            hh = 1 - lh
            hw = 1 - lw
            h0 = h0f.to(tl.int32)
            w0 = w0f.to(tl.int32)
            v1, ok1, r1 = _corner(value_ptr, base, h0, w0, H, W, g, G, D, d, inside)
            v2, ok2, r2 = _corner(value_ptr, base, h0, w0 + 1, H, W, g, G, D, d, inside)
            v3, ok3, r3 = _corner(value_ptr, base, h0 + 1, w0, H, W, g, G, D, d, inside)
            v4, ok4, r4 = _corner(value_ptr, base, h0 + 1, w0 + 1, H, W, g, G, D, d, inside)
            sampled = ((hh * hw)[:, None] * v1 + (hh * lw)[:, None] * v2
                       + (lh * hw)[:, None] * v3 + (lh * lw)[:, None] * v4)
            # attention weight: d out / d a = sampled
            tl.store(gattn_ptr + qg * (L * P) + lp, tl.sum(go * sampled, axis=1), mask=qmask)
            top = go * a[:, None]                                    # (BQ, D)
            # location: bilinear weights' derivatives (zero outside, like mmcv)
            dh = tl.sum(top * (-hw[:, None] * v1 - lw[:, None] * v2 + hw[:, None] * v3 + lw[:, None] * v4), axis=1)
            dw = tl.sum(top * (-hh[:, None] * v1 + hh[:, None] * v2 - lh[:, None] * v3 + lh[:, None] * v4), axis=1)
            dh = tl.where(inside, dh, 0.0)
            dw = tl.where(inside, dw, 0.0)
            tl.store(gloc_ptr + (qg * (L * P) + lp) * 2, dw * W, mask=qmask)
            tl.store(gloc_ptr + (qg * (L * P) + lp) * 2 + 1, dh * H, mask=qmask)
            # value: scatter the bilinear share of top to the four corners
            tl.atomic_add(gvalue_ptr + (r1[:, None] * G + g) * D + d[None, :], (hh * hw)[:, None] * top,
                          mask=ok1[:, None], sem="relaxed")
            tl.atomic_add(gvalue_ptr + (r2[:, None] * G + g) * D + d[None, :], (hh * lw)[:, None] * top,
                          mask=ok2[:, None], sem="relaxed")
            tl.atomic_add(gvalue_ptr + (r3[:, None] * G + g) * D + d[None, :], (lh * hw)[:, None] * top,
                          mask=ok3[:, None], sem="relaxed")
            tl.atomic_add(gvalue_ptr + (r4[:, None] * G + g) * D + d[None, :], (lh * lw)[:, None] * top,
                          mask=ok4[:, None], sem="relaxed")


class MSDATritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights,
                im2col_step=None):
        Bv, S, G, D = value.shape
        _, Q, _, L, P, _ = sampling_locations.shape
        assert D & (D - 1) == 0, "channels per group must be a power of two"
        value = value.contiguous()
        loc = sampling_locations.contiguous().float()
        attn = attention_weights.contiguous().float()
        shapes = spatial_shapes.to(torch.int32).contiguous()
        starts = level_start_index.to(torch.int32).contiguous()
        out = torch.empty(Bv, Q, G * D, device=value.device, dtype=value.dtype)
        grid = lambda meta: (triton.cdiv(Q, meta["BQ"]), G, Bv)
        _msda_fwd[grid](value, shapes, starts, loc, attn, out, S, Q, G=G, D=D, L=L, P=P)
        ctx.save_for_backward(value, shapes, starts, loc, attn)
        ctx.loc_dtype, ctx.attn_dtype = sampling_locations.dtype, attention_weights.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        value, shapes, starts, loc, attn = ctx.saved_tensors
        Bv, S, G, D = value.shape
        _, Q, _, L, P, _ = loc.shape
        gvalue = torch.zeros(value.shape, device=value.device, dtype=torch.float32)
        gloc = torch.empty_like(loc)
        gattn = torch.empty_like(attn)
        grid = lambda meta: (triton.cdiv(Q, meta["BQ"]), G, Bv)
        _msda_bwd[grid](value, shapes, starts, loc, attn, grad_out.contiguous(), gvalue, gloc, gattn,
                        S, Q, G=G, D=D, L=L, P=P)
        return (gvalue.to(value.dtype), None, None, gloc.to(ctx.loc_dtype), gattn.to(ctx.attn_dtype), None)


def msda_triton(value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
    return MSDATritonFunction.apply(value, spatial_shapes, level_start_index, sampling_locations,
                                    attention_weights)
