#!/usr/bin/env python
"""Evaluate a trained checkpoint on the val split."""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch
from torch.utils.data import DataLoader

from unified_perception.config import load_config  # noqa: E402
from unified_perception.data import MultiViewPoseDataset, collate_batch  # noqa: E402
from unified_perception.engine import resolve_device, run_eval  # noqa: E402
from unified_perception.models import UnifiedPerceptionModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(cfg.train.device)

    model = UnifiedPerceptionModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    dataset = MultiViewPoseDataset(cfg, split=args.split)
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_batch,
    )
    metrics = run_eval(model, loader, cfg, device, score_threshold=args.score_threshold)
    print(json.dumps({k: v if math.isfinite(v) else None for k, v in metrics.items()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
