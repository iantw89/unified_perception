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
@triton.jit
def _red_add_v4(addr, v0, v1, v2, v3, mask):
    """One 16-byte vector reduction (``red.global.add.v4.f32``, sm_90+) per element of ``addr``:
    four consecutive fp32 values at a 16-byte aligned address; no return value."""
    return tl.inline_asm_elementwise(
        asm="{ .reg .pred %p; setp.ne.b32 %p, $6, 0; @%p red.global.add.v4.f32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        constraints="=r,l,f,f,f,f,r",
        args=[addr, v0, v1, v2, v3, mask.to(tl.int32)], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _quarters(x, BQ: tl.constexpr, D: tl.constexpr):
    """(BQ, D) -> four (BQ, D // 4) tensors holding channels 4c, 4c + 1, 4c + 2, 4c + 3."""
    x = tl.reshape(x, (BQ, D // 4, 2, 2))
    j0, j1 = tl.split(x)              # last axis: channel 4c + 2i + j
    c0, c2 = tl.split(j0)
    c1, c3 = tl.split(j1)
    return c0, c1, c2, c3


@triton.jit
def _scatter_corner(gvalue_ptr, r, ok, w, t0, t1, t2, t3, g, G: tl.constexpr, D: tl.constexpr,
                    BQ: tl.constexpr, VEC: tl.constexpr, top):
    if VEC:
        c4 = tl.arange(0, D // 4) * 4
        addr = (gvalue_ptr + (r[:, None] * G + g) * D + c4[None, :]).to(tl.int64, bitcast=True)
        wq = w[:, None]
        _red_add_v4(addr, wq * t0, wq * t1, wq * t2, wq * t3, ok[:, None] & (c4[None, :] >= 0))
    else:
        d = tl.arange(0, D)
        tl.atomic_add(gvalue_ptr + (r[:, None] * G + g) * D + d[None, :], w[:, None] * top,
                      mask=ok[:, None], sem="relaxed")


@triton.autotune(
    configs=[triton.Config({"BQ": bq}, num_warps=nw) for bq, nw in ((8, 8), (32, 2), (8, 4), (16, 8), (16, 4))],
    key=["Q", "G", "D", "L", "P", "VEC", "GFAST"], reset_to_zero=["gvalue_ptr"])
@triton.jit
def _msda_bwd(value_ptr, shapes_ptr, starts_ptr, loc_ptr, attn_ptr, gout_ptr,
              gvalue_ptr, gloc_ptr, gattn_ptr,
              S, Q, G: tl.constexpr, D: tl.constexpr, L: tl.constexpr, P: tl.constexpr, BQ: tl.constexpr,
              VEC: tl.constexpr, GFAST: tl.constexpr):
    if GFAST:       # heads on the fastest grid axis: the eight heads of a query tile run together
        g, pid_q, bv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    else:
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
            if VEC:
                t0, t1, t2, t3 = _quarters(top, BQ, D)
            else:
                t0 = t1 = t2 = t3 = top
            _scatter_corner(gvalue_ptr, r1, ok1, hh * hw, t0, t1, t2, t3, g, G, D, BQ, VEC, top)
            _scatter_corner(gvalue_ptr, r2, ok2, hh * lw, t0, t1, t2, t3, g, G, D, BQ, VEC, top)
            _scatter_corner(gvalue_ptr, r3, ok3, lh * hw, t0, t1, t2, t3, g, G, D, BQ, VEC, top)
            _scatter_corner(gvalue_ptr, r4, ok4, lh * lw, t0, t1, t2, t3, g, G, D, BQ, VEC, top)

#: Backward variants (module switches, set from model config): one 16-byte vector reduction per four
#: channels instead of four scalar atomics (Hopper and newer only; ignored elsewhere), and the
#: eight heads of a query tile scheduled next to each other.
BWD_VECTOR_ATOMICS = False
BWD_HEADS_FASTEST = False


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
        vec = (BWD_VECTOR_ATOMICS and value.device.type == "cuda"
               and torch.cuda.get_device_capability(value.device)[0] >= 9 and D % 4 == 0)
        gfast = BWD_HEADS_FASTEST
        grid = ((lambda meta: (G, triton.cdiv(Q, meta["BQ"]), Bv)) if gfast else
                (lambda meta: (triton.cdiv(Q, meta["BQ"]), G, Bv)))
        _msda_bwd[grid](value, shapes, starts, loc, attn, grad_out.contiguous(), gvalue, gloc, gattn,
                        S, Q, G=G, D=D, L=L, P=P, VEC=vec, GFAST=gfast)
        return (gvalue.to(value.dtype), None, None, gloc.to(ctx.loc_dtype), gattn.to(ctx.attn_dtype), None)


def msda_triton(value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
    return MSDATritonFunction.apply(value, spatial_shapes, level_start_index, sampling_locations,
                                    attention_weights)
