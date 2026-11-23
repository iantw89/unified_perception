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
