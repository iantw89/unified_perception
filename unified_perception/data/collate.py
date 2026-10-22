"""Batch collation: stack fixed-shape tensors, keep metadata as lists."""

from __future__ import annotations

import torch


def collate_batch(samples):
    out = {}
    tensor_keys = [
        "imgs",
        "world2img",
        "intrinsics",
        "gt_keypoints",
        "gt_valid",
    ]
    for k in tensor_keys:
        out[k] = torch.stack([s[k] for s in samples], dim=0)
    out["seq"] = [s["seq"] for s in samples]
    out["start"] = [s["start"] for s in samples]
    return out
