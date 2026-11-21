"""Export a fast-reid person re-ID checkpoint to a self-contained TorchScript file.

Run with the fast-reid checkout's own interpreter (it needs the ``fastreid``
package); unified_perception then loads the result with ``torch.jit.load`` and
has no fast-reid dependency:

    ..\\fast-reid\\.venv\\Scripts\\python.exe unified_perception\\tools\\export_fastreid.py \\
        --fastreid ..\\fast-reid --config configs/MSMT17/sbs_R50.yml \\
        --weights ..\\fast-reid\\pretrained_download\\msmt_sbs_R50.pth \\
        --output weights\\reid\\msmt_sbs_R50.ts

Exported module: input (N, 3, H, W) float RGB in [0, 255] at the training crop
size (384 x 128 for SBS), output (N, D) L2-normalized embeddings (the BNNeck
features fast-reid uses for retrieval).  A JSON sidecar records provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fastreid", required=True, help="fast-reid checkout")
    ap.add_argument("--config", required=True, help="config path relative to the checkout")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.fastreid).resolve()
    sys.path.insert(0, str(root))
    import torch
    import torch.nn.functional as F
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model

    cfg = get_cfg()
    cfg.merge_from_file(str(root / args.config))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    model = build_model(cfg)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)["model"]
    # v0.1.x release checkpoints predate fast-reid's renames: BNNeck "heads.bnneck" is now
    # "heads.bottleneck.0"; normalization constants were persistent buffers (verified below).
    for key in [k for k in state if k.startswith("heads.bnneck.")]:
        state[key.replace("heads.bnneck.", "heads.bottleneck.0.")] = state.pop(key)
    for key in ("pixel_mean", "pixel_std"):
        if key in state:
            expected = torch.tensor(getattr(cfg.MODEL, key.upper())).view_as(state[key])
            if not torch.allclose(state.pop(key).float(), expected.float()):
                raise RuntimeError(f"checkpoint {key} differs from the config")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Training-only identity classifiers (1041 MSMT17 ids) are not needed for embeddings.
    missing = [k for k in missing if k != "heads.weight"]
    unexpected = [k for k in unexpected if not k.startswith(("heads.classifier", "heads.bnneck.num_batches"))]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.eval()

    class Embedder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, images):
            # fast-reid normalizes its input in place; never mutate the caller's tensor.
            return F.normalize(self.inner(images.clone()), dim=-1)

    height, width = cfg.INPUT.SIZE_TEST
    example = torch.rand(2, 3, height, width) * 255
    embedder = Embedder(model).eval()
    with torch.no_grad():
        reference = embedder(example)
        traced = torch.jit.trace(embedder, example)
        check = traced(torch.rand(3, 3, height, width) * 255)
    if not torch.allclose(traced(example), reference, atol=1e-4):
        raise RuntimeError("traced module differs from the eager model")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    meta = {
        "source_weights": str(Path(args.weights).resolve()), "source_sha256": sha256(Path(args.weights)),
        "fastreid_config": args.config, "input": f"(N, 3, {height}, {width}) float RGB 0-255",
        "embedding_dims": int(reference.shape[-1]), "normalized": True,
        "torch_version": torch.__version__, "output_sha256": sha256(output),
        "batch_shape_check": list(check.shape),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
