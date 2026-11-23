"""Training-time camera dropout.

A lane chosen for dropout (one draw per streamed piece) masks a fresh random subset
of cameras on *every frame*, so the kept views change within and across windows:
the model sees fragmented, perspective-shifting coverage rather than whole regions of
the gym going dark for a whole recording.  Removing views teaches the model to estimate 3-D people from fewer cameras -- down
to one -- supervised by true 3-D ground truth on the mocap datasets (H36M, MPI,
MOYO) and by triangulated targets on the gym data.  A person that none of the
kept views sees is no longer a positive: they become *present but unobserved*
(``instance_exists`` without ``gt_valid``), so their track is kept alive without
supervision instead of being trained to vanish.
"""

from __future__ import annotations

import random

import torch

CAMERA_KEYS = ("imgs", "world2img", "intrinsics", "visibility", "gt_kp2d", "gt_kp2d_valid")


def choose_views(num_views: int, rng: random.Random, prob: float, min_views: int = 1) -> list[int] | None:
    """Kept view indices, or None to keep all (with probability 1 - prob)."""
    if num_views <= min_views or rng.random() >= prob:
        return None
    keep = rng.randint(min_views, num_views - 1)
    return sorted(rng.sample(range(num_views), keep))


def frame_camera_masks(num_frames: int, num_views: int, rng: random.Random,
                       min_views: int = 1) -> torch.Tensor:
    """(T, N) bool: an independent random subset of min_views..N-1 kept views per frame."""
    mask = torch.zeros(num_frames, num_views, dtype=torch.bool)
    for t in range(num_frames):
        keep = rng.randint(min_views, num_views - 1)
        mask[t, rng.sample(range(num_views), keep)] = True
    return mask


def mask_cameras(sample: dict, mask: torch.Tensor) -> dict:
    """Hide views per frame (``mask`` (T, N), True = kept) without changing tensor shapes.

    The mask becomes ``camera_mask`` (collated into ``camera_valid``, so hidden views are
    skipped by the backbone, attention and re-id crops); 2-D targets and visibility of
    hidden views are cleared, and people no kept view sees in a frame become present but
    unobserved in that frame.
    """
    mask = torch.as_tensor(mask, dtype=torch.bool)
    sample["camera_mask"] = mask & torch.as_tensor(sample.get("camera_mask", mask)).bool()
    if "gt_kp2d_valid" in sample:
        valid2d = torch.as_tensor(sample["gt_kp2d_valid"]).bool()
        sample["gt_kp2d_valid"] = valid2d & mask.reshape(*mask.shape, *([1] * (valid2d.dim() - 2)))
    if "visibility" in sample:
        vis = torch.as_tensor(sample["visibility"])
        sample["visibility"] = vis * mask.reshape(*mask.shape, *([1] * (vis.dim() - 2))).to(vis.dtype)
        _unseen_to_present(sample)
    return sample


def _unseen_to_present(sample: dict) -> None:
    valid = torch.as_tensor(sample["gt_valid"]).bool()
    seen = sample["visibility"].amax(dim=1) > 0                         # (T, P) over kept views
    exists = torch.as_tensor(sample.get("instance_exists", valid)).bool()
    sample["instance_exists"] = exists | (valid & ~seen)
    valid = valid & seen
    sample["gt_valid"] = valid
    if "gt_keypoint_valid" in sample:
        sample["gt_keypoint_valid"] = torch.as_tensor(sample["gt_keypoint_valid"]).bool() & valid[..., None]
    if "gt_root_valid" in sample:
        sample["gt_root_valid"] = torch.as_tensor(sample["gt_root_valid"]).bool() & valid


def drop_cameras(sample: dict, keep: list[int]) -> dict:
    index = torch.as_tensor(keep, dtype=torch.long)
    for key in CAMERA_KEYS:
        if key in sample:
            value = torch.as_tensor(sample[key])
            sample[key] = value.index_select(1, index)
    if "camera_ids" in sample:
        sample["camera_ids"] = [sample["camera_ids"][i] for i in keep]
    if "camera_mask" in sample:
        sample["camera_mask"] = torch.as_tensor(sample["camera_mask"]).index_select(1, index)
    if "visibility" in sample:
        _unseen_to_present(sample)
    return sample
