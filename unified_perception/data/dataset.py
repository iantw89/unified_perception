"""Multi-view temporal dataset via a provider registry.

The dataset class itself is intentionally thin: it delegates sequence discovery
and sample loading to a pluggable provider selected by ``cfg.data.provider``.
This keeps data access concerns decoupled from model/training code.
"""

from __future__ import annotations

import random as _rnd

import torch
from torch.utils.data import Dataset

from ..config import Config
from .registry import build_provider


class MultiViewPoseDataset(Dataset):
    def __init__(self, cfg: Config, split: str = "train"):
        self.provider = build_provider(cfg.data.provider, cfg, split)
        self.samples = self.provider.list_samples()
        if not self.samples:
            raise RuntimeError(
                f"No samples found for provider={cfg.data.provider} split={split}"
            )
        # Optionally cap the training set size for fast pipe-cleaner runs.
        max_s = getattr(cfg.data, "max_samples_per_epoch", 0)
        if split == "train" and max_s > 0 and len(self.samples) > max_s:
            rng = _rnd.Random(getattr(cfg.data, "split_seed", 0))
            self.samples = rng.sample(self.samples, max_s)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _MAX_RETRIES = 10
        for attempt in range(_MAX_RETRIES):
            try:
                # TODO: Fix the missing camera issue
                raw = self.provider.load_window(self.samples[(idx + attempt * 509) % len(self)])
                break
            except (ValueError, RuntimeError, OSError) as exc:
                if attempt == 0:
                    import warnings
                    warnings.warn(
                        f"[MultiViewPoseDataset] skipping sample {idx} "
                        f"(attempt {attempt}): {exc}",
                        stacklevel=2,
                    )
        else:
            raise RuntimeError(
                f"[MultiViewPoseDataset] failed to load {_MAX_RETRIES} "
                f"consecutive samples starting at index {idx}."
            )
        out = {
            "imgs": raw["imgs"],
            "world2img": torch.from_numpy(raw["world2img"]),
            "intrinsics": torch.from_numpy(raw["intrinsics"]),
            "gt_keypoints": torch.from_numpy(raw["gt_keypoints"]),
            "gt_valid": torch.from_numpy(raw["gt_valid"]),
            "seq": raw["seq"],
            "start": raw["start"],
        }
        if "frame_ids" in raw:
            out["frame_ids"] = torch.from_numpy(raw["frame_ids"])
        return out