"""FP8 frozen trunk vs bf16: feature drift on real images, then validation metrics side by side.

    python -m unified_perception.tools.check_fp8_trunk CONFIG --checkpoint last.pt
        [--feature-batches 2] [--eval-batches 20] [--keep-bf16 1] [overrides...]

Feature drift: per-token cosine similarity and relative L2 error of the trunk's output grid
(what the SimpleFPN reads), on gym validation windows.  Metrics: ``run_eval`` on the first
``--eval-batches`` gym validation batches with the trained checkpoint, bf16 then FP8.
"""

from __future__ import annotations

import argparse

import torch

from ..config import load_config
from ..engine.evaluate import run_eval
from ..engine.train import build_loaders, resolve_device, set_seed
from ..models import UnifiedPerceptionModel


def set_fp8(backbone, enabled: bool, keep_bf16: int) -> None:
    backbone.fp8, backbone.fp8_keep_bf16 = enabled, keep_bf16
    backbone._drop_trunk()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--feature-batches", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--keep-bf16", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    model = UnifiedPerceptionModel(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    model.eval()
    backbone = model.backbone
    loader = build_loaders(cfg)["val_gym"]
    mean, std = model.input_mean.view(1, 3, 1, 1), model.input_std.view(1, 3, 1, 1)

    cos_all, rel_all = [], []
    for bi, batch in enumerate(loader):
        if bi >= args.feature_batches:
            break
        imgs = batch["imgs"].to(device)
        imgs = imgs.reshape(-1, *imgs.shape[-3:])
        feats = {}
        for enabled in (False, True):
            set_fp8(backbone, enabled, args.keep_bf16)
            feats[enabled] = backbone._vit_features(imgs, mean, std).float()       # (n, C, h, w)
        ref, fp8 = feats[False], feats[True]
        cos = torch.nn.functional.cosine_similarity(ref, fp8, dim=1).flatten()
        rel = (fp8 - ref).norm(dim=1).flatten() / ref.norm(dim=1).flatten().clamp(min=1e-6)
        cos_all.append(cos.cpu())
        rel_all.append(rel.cpu())
    cos, rel = torch.cat(cos_all), torch.cat(rel_all)
    q = torch.tensor([0.001, 0.01, 0.5])
    print(f"[fp8] keep_bf16={args.keep_bf16}: token cosine mean {cos.mean():.5f}, "
          f"quantiles(0.1%/1%/50%) {[round(v, 5) for v in cos.quantile(q).tolist()]}; "
          f"relative L2 mean {rel.mean():.4f}, 99% {rel.quantile(0.99):.4f} ({cos.numel()} tokens)")

    if args.eval_batches > 0:
        results = {}
        for enabled in (False, True):
            set_fp8(backbone, enabled, args.keep_bf16)
            results[enabled] = run_eval(model, loader, cfg, device, max_batches=args.eval_batches)
        keys = [k for k in results[False] if "/" not in k and isinstance(results[False][k], (int, float))]
        print(f"[fp8] val_gym, first {args.eval_batches} batches:   bf16   ->   fp8")
        for k in keys:
            print(f"[fp8]   {k:32s} {results[False][k]:12.4f} {results[True].get(k, float('nan')):12.4f}")


if __name__ == "__main__":
    main()
