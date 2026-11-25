"""DINOv3 ViT port vs Hugging Face ``DINOv3ViTModel`` (an independent implementation) on the
official ViT-B/16 checkpoint, plus the SimpleFPN backbone's shapes and precision contract."""

from pathlib import Path

import pytest
import torch

from unified_perception.models.vit_backbone import DinoV3SimpleFPNBackbone, DinoV3ViT, SimpleFPN, fpn_levels_for

WEIGHTS = Path(__file__).resolve().parents[1] / "weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
needs_weights = pytest.mark.skipif(not WEIGHTS.is_file(), reason="DINOv3 ViT-B/16 weights not present")


def _hf_state(official: dict) -> dict:
    """Official DINOv3 keys -> HF DINOv3ViTModel keys (qkv split; no key bias)."""
    out = {"embeddings.cls_token": official["cls_token"],
           "embeddings.mask_token": official["mask_token"].view(1, 1, -1),
           "embeddings.register_tokens": official["storage_tokens"],
           "embeddings.patch_embeddings.weight": official["patch_embed.proj.weight"],
           "embeddings.patch_embeddings.bias": official["patch_embed.proj.bias"],
           "norm.weight": official["norm.weight"], "norm.bias": official["norm.bias"]}
    depth = 1 + max(int(k.split(".")[1]) for k in official if k.startswith("blocks."))
    for i in range(depth):
        o, h = f"blocks.{i}.", f"model.layer.{i}."
        q, k, v = official[o + "attn.qkv.weight"].chunk(3)
        qb, _, vb = official[o + "attn.qkv.bias"].chunk(3)
        out.update({h + "attention.q_proj.weight": q, h + "attention.q_proj.bias": qb,
                    h + "attention.k_proj.weight": k,
                    h + "attention.v_proj.weight": v, h + "attention.v_proj.bias": vb,
                    h + "attention.o_proj.weight": official[o + "attn.proj.weight"],
                    h + "attention.o_proj.bias": official[o + "attn.proj.bias"],
                    h + "layer_scale1.lambda1": official[o + "ls1.gamma"],
                    h + "layer_scale2.lambda1": official[o + "ls2.gamma"],
                    h + "mlp.up_proj.weight": official[o + "mlp.fc1.weight"],
                    h + "mlp.up_proj.bias": official[o + "mlp.fc1.bias"],
                    h + "mlp.down_proj.weight": official[o + "mlp.fc2.weight"],
                    h + "mlp.down_proj.bias": official[o + "mlp.fc2.bias"]})
        for n in ("norm1", "norm2"):
            for p in ("weight", "bias"):
                out[f"{h}{n}.{p}"] = official[f"{o}{n}.{p}"]
    return out


@needs_weights
@pytest.mark.parametrize("size", [(224, 224), (640, 640), (480, 640)])
def test_matches_huggingface(size):
    transformers = pytest.importorskip("transformers")
    official = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    config = transformers.DINOv3ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                                          intermediate_size=3072, num_register_tokens=4)
    ref = transformers.DINOv3ViTModel(config).eval()
    ref.load_state_dict({k: v.float() for k, v in _hf_state(official).items()}, strict=True)
    mine = DinoV3ViT("vitb16").eval()
    mine.load_official(str(WEIGHTS))
    torch.manual_seed(0)
    x = torch.randn(2, 3, *size)
    with torch.no_grad():
        r = ref(pixel_values=x).last_hidden_state[:, 5:]
        h, w = size[0] // 16, size[1] // 16
        r = r.transpose(1, 2).reshape(2, 768, h, w)
        m = mine(x)
    assert m.shape == r.shape
    assert ((m - r).norm() / r.norm()).item() < 1e-4


@needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 trunk test uses CUDA")
def test_backbone_levels_and_precision():
    """Strides 8/16/32 at 640 (no stride 64 by default); same frozen-trunk numerics with and without autocast (the
    evaluation path runs without it); the FPN is trainable, the ViT frozen."""
    bb = DinoV3SimpleFPNBackbone(str(WEIGHTS)).cuda()
    imgs = torch.randint(0, 255, (3, 3, 640, 640), dtype=torch.uint8, device="cuda")
    mean = torch.zeros(1, 3, 1, 1, device="cuda")
    std = torch.full((1, 3, 1, 1), 255.0, device="cuda")
    with torch.no_grad():
        plain, _ = bb(imgs, input_mean=mean, input_std=std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            amp, _ = bb(imgs, input_mean=mean, input_std=std)
        trunk_plain = bb._vit_features(imgs, mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trunk_amp = bb._vit_features(imgs, mean, std)
    assert [tuple(l.shape[-2:]) for l in plain] == [(80, 80), (40, 40), (20, 20)]
    assert all(l.dtype == torch.float32 for l in plain)       # (autocast keeps LayerNorm output fp32)
    assert torch.equal(trunk_plain, trunk_amp)
    assert not any(p.requires_grad for p in bb.vit.parameters())
    assert all(p.requires_grad for p in bb.fpn.parameters())


def test_fpn_levels_follow_image_size():
    """640 px: strides 8/16/32 only; 1280 px: stride 64 as well (config vit_fpn_levels: 0)."""
    assert fpn_levels_for(640, 640) == 3 and fpn_levels_for(1280, 1280) == 4 and fpn_levels_for(720, 1280) == 4
    fpn = SimpleFPN(32, 8, fpn_levels_for(1280, 1280))
    shapes = [tuple(l.shape[-2:]) for l in fpn(torch.zeros(1, 32, 80, 80))]     # 1280 / 16 = 80
    assert shapes == [(160, 160), (80, 80), (40, 40), (20, 20)]


def _sample(n, h, w, image_hw=None):
    s = {"imgs": torch.full((2, n, 3, h, w), 7, dtype=torch.uint8)}
    if image_hw is not None:
        s["image_hw"] = torch.tensor(image_hw)
    return s


def test_canvas_placement_native_aspect():
    """Mixed native sizes share a top-left canvas rounded to stride 32; MOYO-style samples padded
    to a long-side square shrink to the batch's largest valid size; legacy batches are untouched."""
    from unified_perception.data.collate import place_on_canvas
    gym = _sample(2, 480, 640)
    moyo = _sample(2, 640, 640, [[468, 640], [640, 468]])
    imgs, hw = place_on_canvas([gym, moyo])
    assert imgs.shape == (2, 2, 2, 3, 640, 640)
    assert hw.tolist() == [[[480, 640], [480, 640]], [[468, 640], [640, 468]]]
    assert imgs[0, :, :, :, 480:].abs().sum() == 0            # gym below its valid rows: padding
    landscape = _sample(1, 640, 640, [[468, 640]])
    imgs, hw = place_on_canvas([gym, landscape])
    assert imgs.shape[-2:] == (480, 640) and hw[1, 1].tolist() == [0, 0]   # missing camera
    assert place_on_canvas([_sample(2, 640, 640), _sample(2, 640, 640)]) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 trunk test uses CUDA")
def test_valid_region_groups_and_padding():
    """Images are grouped by valid size (rounded up to stride 32) and returned in input order; an
    image's levels equal the region-only run zero-padded bottom / right onto the canvas grid."""
    torch.manual_seed(0)
    bb = DinoV3SimpleFPNBackbone(None).cuda().eval()
    imgs = torch.randint(0, 255, (5, 3, 640, 640), dtype=torch.uint8, device="cuda")
    hw = torch.tensor([[480, 640], [640, 640], [468, 640], [480, 640], [640, 468]])
    for i, (h, w) in enumerate(hw.tolist()):
        imgs[i, :, h:] = 0
        imgs[i, :, :, w:] = 0
    mean = torch.zeros(1, 3, 1, 1, device="cuda")
    std = torch.full((1, 3, 1, 1), 255.0, device="cuda")
    with torch.no_grad():
        out, _ = bb(imgs, input_mean=mean, input_std=std, valid_hw=hw)
        landscape = bb._levels(imgs[[0, 2, 3], :, :480], mean, std)    # 468 rounds up to 480
        portrait = bb._levels(imgs[[4], :, :, :480], mean, std)
        square = bb._levels(imgs[[1]], mean, std)
    assert [tuple(l.shape) for l in out] == [(5, 256, 80, 80), (5, 256, 40, 40), (5, 256, 20, 20)]
    for level, a, p, q, s in zip(out, landscape, portrait, square, (8, 16, 32)):
        k = 480 // s
        assert torch.allclose(level[[0, 2, 3], :, :k], a, atol=1e-3)
        assert level[[0, 2, 3], :, k:].abs().max() == 0
        assert torch.allclose(level[[4], :, :, :k], p, atol=1e-3) and level[[4], :, :, k:].abs().max() == 0
        assert torch.allclose(level[[1]], q, atol=1e-3)


def test_crop_alignment_follows_levels():
    assert DinoV3SimpleFPNBackbone(None, fpn_levels=3).crop_alignment == 32
    assert DinoV3SimpleFPNBackbone(None, fpn_levels=4).crop_alignment == 64


def test_fp8_linear_matches_linear_cpu():
    """The FP8 path (here its CPU reference) stays close to the bf16 linear it replaces."""
    import torch.nn as nn
    from unified_perception.models.vit_backbone import Fp8Linear, DinoV3ViT, to_fp8_blocks
    torch.manual_seed(0)
    lin = nn.Linear(64, 96)
    x = torch.randn(3, 10, 64)
    ref = lin(x)
    out = Fp8Linear(lin)(x)
    assert out.shape == ref.shape
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    assert cos > 0.998
    vit = DinoV3ViT("vits16")
    assert to_fp8_blocks(vit, keep_bf16=1) == list(range(1, 11))
    assert isinstance(vit.blocks[5].mlp.fc1, Fp8Linear) and isinstance(vit.blocks[0].attn.qkv, nn.Linear)


def test_trunk_ahead_features_reproduce_forward_cpu():
    """Frozen-trunk outputs computed ahead (engine/step.py: TrunkAhead) give the same levels as a
    normal forward, across valid-size groups, and are consumed exactly once."""
    torch.manual_seed(0)
    bb = DinoV3SimpleFPNBackbone(None, arch="vits16", out_channels=32).eval()
    imgs = torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8)
    hw = torch.tensor([[192, 256], [256, 256], [180, 256], [256, 192]])
    mean = torch.zeros(1, 3, 1, 1)
    std = torch.full((1, 3, 1, 1), 255.0)
    with torch.no_grad():
        ref, _ = bb(imgs, input_mean=mean, input_std=std, valid_hw=hw)
        feats = bb.trunk_ahead(imgs, mean, std, valid_hw=hw)
        assert len(feats) == 3                         # 192x256, 256x256, 256x192 groups
        bb.use_given_features(feats)
        out, _ = bb(imgs, input_mean=mean, input_std=std, valid_hw=hw)
        bb.use_given_features(None)
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    bb.use_given_features(feats[:1])
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            bb(imgs, input_mean=mean, input_std=std, valid_hw=hw)
    bb.use_given_features(None)
