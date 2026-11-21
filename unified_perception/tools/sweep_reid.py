"""Sweep inference-time re-identification settings on one checkpoint (no retraining).

Streams every held-out clip of the streaming validation sets (``val_gym``,
``val_aligned``) whole for each (reid_wait_frames, reid_threshold) pair and
writes one JSON row per setting with the delayed-decision identity metrics
(``id_switches``, which apply re-id takeovers retroactively), the causal
``id_switches_online`` and ``id_reid_takeovers``.

    python -m unified_perception.tools.sweep_reid --config configs/gym_sixcam_track_v1.yaml \\
        --checkpoint runs/gym_sixcam_track_v1/best.pt --wait 5 10 20 40 80 --threshold 0.5 0.6 0.7
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unified_perception.config import load_config  # noqa: E402
from unified_perception.engine import resolve_device, run_eval  # noqa: E402
from unified_perception.engine.train import build_loaders  # noqa: E402
from unified_perception.models.detector import UnifiedPerceptionModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wait", type=int, nargs="+", required=True, help="reid_wait_frames values")
    ap.add_argument("--threshold", type=float, nargs="+", required=True, help="reid_threshold values")
    ap.add_argument("--output", default=None, help="JSON lines file (default: next to the checkpoint)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    if not (cfg.model.track_queries and cfg.model.reid_dims > 0):
        raise SystemExit("The config must enable model.track_queries and model.reid_dims")
    device = resolve_device(cfg.train.device)
    model = UnifiedPerceptionModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
    loaders = {k: v for k, v in build_loaders(cfg).items() if k in ("val_gym", "val_aligned")}
    if not loaders:
        raise SystemExit("No streaming validation sets for this provider")
    manager = model.head.track_manager
    output = Path(args.output or Path(args.checkpoint).with_suffix(".reid_sweep.jsonl"))
    with output.open("w", encoding="utf-8") as stream:
        for wait, threshold in itertools.product(args.wait, args.threshold):
            manager.reid_wait, manager.reid_threshold = wait, threshold
            for name, loader in loaders.items():
                metrics = run_eval(model, loader, cfg, device)
                row = {"split": name, "reid_wait_frames": wait, "reid_threshold": threshold,
                       **{k: v for k, v in metrics.items() if k.startswith(("id_", "f1", "mpjpe"))}}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
