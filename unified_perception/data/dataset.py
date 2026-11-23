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
from ..constants import ROOT_JOINTS, ROOT_SLICE
from .registry import build_provider


def _with_sparse_targets(sample):
    """Normalize optional sparse targets and supply dense-compatible defaults."""
    keypoints = sample["gt_keypoints"]
    valid = sample["gt_valid"].bool()
    frame_valid = torch.as_tensor(
        sample.get(
            "gt_keypoint_frame_valid",
            torch.ones(keypoints.shape[0], dtype=torch.bool, device=keypoints.device),
        ),
        dtype=torch.bool,
    )
    joint_valid = torch.as_tensor(
        sample.get(
            "gt_keypoint_valid",
            valid.unsqueeze(-1).expand(*valid.shape, keypoints.shape[-2]),
        ),
        dtype=torch.bool,
    )
    sample["gt_keypoint_frame_valid"] = frame_valid
    sample["gt_keypoint_valid"] = (
        joint_valid & valid.unsqueeze(-1) & frame_valid[:, None, None]
    )
    sample.setdefault("gt_root", keypoints[..., ROOT_SLICE, :].mean(dim=-2))
    sample.setdefault("gt_root_valid", valid)
    sample["gt_root"] = torch.as_tensor(sample["gt_root"], dtype=keypoints.dtype)
    sample["gt_root_valid"] = torch.as_tensor(sample["gt_root_valid"], dtype=torch.bool) & valid
    return sample


class MultiViewPoseDataset(Dataset):
    def __init__(self, cfg: Config, split: str = "train"):
        self.composed = None
        if cfg.data.composition_manifest:
            from .composition import CompositionDataset
            self.composed = CompositionDataset(cfg, split)
            return
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
        if self.composed is not None:
            return len(self.composed)
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.composed is not None:
            return _with_sparse_targets(self.composed[idx])
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
        for key in ("gt_keypoint_frame_valid", "gt_keypoint_valid", "gt_root", "gt_root_valid"):
            if key in raw:
                value = raw[key]
                out[key] = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if "frame_ids" in raw:
            out["frame_ids"] = torch.from_numpy(raw["frame_ids"])
        if "camera_ids" in raw:
            out["camera_ids"] = raw["camera_ids"]
        if "image_hw" in raw:
            out["image_hw"] = torch.as_tensor(raw["image_hw"], dtype=torch.long)
        return _with_sparse_targets(out)