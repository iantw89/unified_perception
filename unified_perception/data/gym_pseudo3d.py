"""Sparse reviewed-gym cache and its mixture with the composition mocap pool.

The cached cameras are pinhole, z-up/mm, with a single train-fitted xy offset.
No pseudo joint is treated as observed unless its explicit mask is true.

Gym windows use subsets of the reviewed camera IDs recorded by their cache.
With ``allow_fewer_cameras``, num_cameras is a ceiling. A person exists in a window frame only when one of the chosen
views observed them (human 2-D track at the nearest labeled frame), so count,
root, and joint supervision always match the given views.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import JOINT_NAMES
from .identity import fill_identity_gaps
from .interface import DataProvider
from .registry import register_provider
from .transforms import resize_image_and_intrinsics

GYM_CAMERA_IDS = [0, 1, 2, 6]
GYM_CAPTURE_GROUP = "gym_20250505"


def labeled_window_starts(labeled, seq_len, mode="end"):
    """Window starts that keep every window inside one split and on a label.

    ``mode="end"``: full windows ending on labels (streaming evaluation), plus
    leading/trailing boundary windows provided they contain a labeled frame.
    ``mode="any"``: every in-split start whose window contains a labeled frame.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    labeled = np.asarray(labeled, dtype=bool)
    if mode == "any":
        return [s for s in range(len(labeled) - seq_len + 1) if labeled[s:s + seq_len].any()]
    if mode != "end":
        raise ValueError(f"Unknown gym window mode {mode!r}")
    starts = [int(end) - seq_len + 1 for end in np.flatnonzero(labeled) if end >= seq_len - 1]
    if len(labeled) >= seq_len and labeled[:seq_len].any() and (not starts or starts[0] != 0):
        starts.insert(0, 0)
    tail = len(labeled) - seq_len
    if tail >= 0 and labeled[tail:].any() and tail not in starts:
        starts.append(tail)
    return starts


def _view_subsets(num_cameras, camera_ids=GYM_CAMERA_IDS):
    if not 1 <= num_cameras <= len(camera_ids):
        raise ValueError(f"Gym supports 1..{len(camera_ids)} of its reviewed views {camera_ids}")
    return list(combinations(range(len(camera_ids)), num_cameras))


@register_provider("gym_pseudo3d")
class GymPseudo3DProvider(DataProvider):
    """Samples are ``(sequence, start, view_slots)``; every view subset is enumerated."""

    def __init__(self, cfg, split="train", window_mode=None):
        self.cfg, self.split = cfg, split
        self.root = Path(cfg.data.gym_pseudo3d_root)
        index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.camera_ids = index["camera_ids"]
        self.cache_hw = tuple(index["image_hw"])
        self.format_version = index.get("format_version", 1)
        self.image_pattern = index.get("image_pattern", "f{frame:03d}_c{camera:03d}.png")
        self.capture_group = index.get("dataset", GYM_CAPTURE_GROUP) if self.format_version >= 3 else GYM_CAPTURE_GROUP
        if (index["joint_names"] != JOINT_NAMES or not self.camera_ids
                or self.camera_ids != sorted(set(self.camera_ids))
                or any(c not in range(7) for c in self.camera_ids)):
            raise ValueError("Unexpected gym joint/camera order")
        if index.get("format_version", 1) < 2:
            raise ValueError("Gym cache lacks per-view observation masks; rerun import_gym_pseudo3d --overwrite")
        cameras = (min(cfg.data.num_cameras, len(self.camera_ids))
                   if cfg.data.allow_fewer_cameras else cfg.data.num_cameras)
        self.view_subsets = _view_subsets(cameras, self.camera_ids)
        lower, upper = np.asarray(index["valid_coordinate_bounds_mm"])
        position_range = np.asarray(cfg.model.position_range)
        if (lower < position_range[:3]).any() or (upper > position_range[3:]).any():
            raise ValueError("Gym coordinates exceed model.position_range; use the explicit gym mix/eval config bounds")
        if split not in index["splits"]:
            raise ValueError(f"No gym split {split!r}")
        window_mode = window_mode or ("any" if split == "train" else "end")
        self.sequences, self.samples, self.people, self.present = {}, [], [], {}
        for entry in index["splits"][split]:
            name = entry["seq"]
            folder = self.root / split / name
            with np.load(folder / "poses.npz", allow_pickle=False) as arrays:
                poses = {key: arrays[key] for key in arrays.files}
            with np.load(folder / "calib.npz", allow_pickle=False) as arrays:
                calib = {key: arrays[key] for key in arrays.files}
            if not np.array_equal(calib["camera_ids"], index["camera_ids"]):
                raise ValueError("Calibration camera order differs from cache index")
            # v3 (data/golden.py) may interpolate targets between labelled frames, with an
            # uncertainty that includes the motion; older caches never supervise unlabeled frames.
            for value, valid in (("keypoint_sigma", "keypoint_valid"), ("root_sigma", "root_valid"),
                                 ("keypoints", "keypoint_valid"), ("root", "root_valid")):
                if value in poses and not np.isfinite(poses[value][poses[valid]]).all():
                    raise ValueError(f"{folder}: non-finite {value} on valid targets; rebuild the cache")
            interpolated = poses.get("interpolated", np.zeros(poses["valid"].shape, bool))
            if (poses["keypoint_valid"] & ~poses["labeled"][:, None, None] & ~interpolated[..., None]).any():
                raise ValueError("Unlabeled gym frames must not supervise any joints")
            if not np.array_equal(poses["observed"].any(-1), poses["valid"]):
                raise ValueError("Gym existence must equal observation in some reviewed view")
            self.sequences[name] = (folder, poses, calib)
            # Present = observed in some reviewed view, or briefly unobserved by every
            # camera between two observations (track kept alive, not supervised).
            self.present[name] = fill_identity_gaps(poses["valid"], cfg.data.track_max_gap)
            for start in labeled_window_starts(poses["labeled"], cfg.data.seq_len, window_mode):
                for views in self.view_subsets:
                    sample = (name, start, views)
                    self.samples.append(sample)
                    self.people.append(len(self._slots(sample)))
        self.max_window_people = max(self.people, default=0)
        if self.max_window_people > cfg.data.max_people:
            raise ValueError(
                f"Gym windows need max_people >= {self.max_window_people}; do not truncate exact counts")

    def list_samples(self):
        return self.samples

    def _seen(self, sample):
        name, start, views = sample
        poses = self.sequences[name][1]
        stop = start + self.cfg.data.seq_len
        observed = poses["observed"][start:stop][:, :, list(views)]
        return observed, observed.any(-1) & poses["valid"][start:stop]

    def _present(self, sample):
        name, start, _ = sample
        stop = start + self.cfg.data.seq_len
        return self.present[name][start:stop]

    def _slots(self, sample):
        # Everyone present in the window (in any reviewed view), so identities stay
        # attached to slots even while the chosen views do not observe them.
        return np.flatnonzero(self._seen(sample)[1].any(0) | self._present(sample).any(0))

    def load_window(self, sample):
        if len(sample) == 2:
            sample = (*sample, self.view_subsets[0])
        name, start, views = sample
        folder, poses, calib = self.sequences[name]
        cfg = self.cfg.data
        length, size = cfg.seq_len, (cfg.image_height, cfg.image_width)
        if cfg.image_long_side:
            # Native aspect: the cache must already be at the long side (no letterbox, no resize).
            if max(self.cache_hw) != cfg.image_long_side:
                raise ValueError(f"Gym cache is {self.cache_hw}; image_long_side={cfg.image_long_side} "
                                 "needs a cache imported at that size (tools/import_golden.py --size)")
            size = self.cache_hw
        stop = start + length
        if start < 0 or stop > len(poses["frames"]):
            raise ValueError("Window extends past a split boundary")
        if tuple(views) not in self.view_subsets:
            raise ValueError(f"Invalid gym view subset {views}")
        n = len(views)
        imgs = torch.empty((length, n, 3, *size), dtype=torch.uint8)
        world2img = np.empty((length, n, 4, 4), dtype=np.float32)
        intrinsics = np.empty_like(world2img)
        rescale = {}  # view -> (cache K, resized K) when the cache size differs from the config
        for ni, ci in enumerate(views):
            extrinsic, intrinsic = np.eye(4), np.eye(4)
            extrinsic[:3, :3], extrinsic[:3, 3] = calib["R"][ci], calib["T"][ci]
            for ti, frame in enumerate(range(start, stop)):
                path = folder / "images" / self.image_pattern.format(frame=frame, camera=ci)
                image = cv2.imread(str(path))
                if image is None:
                    raise OSError(f"Missing cached image: {path}")
                K = calib["K"][ci]
                if image.shape[:2] != size:
                    image, K = resize_image_and_intrinsics(image, K, size)
                    rescale[ni] = (calib["K"][ci], K)
                imgs[ti, ni] = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                intrinsic[:3, :3] = K
                intrinsics[ti, ni] = intrinsic
                world2img[ti, ni] = intrinsic @ extrinsic
        observed, seen = self._seen(sample)
        slots = self._slots(sample)
        present = self._present(sample)[:, slots]
        people = cfg.max_people
        if len(slots) > people:
            raise ValueError("Gym window exceeds max_people")
        seen = seen[:, slots]
        targets = {
            "gt_valid": seen,
            "gt_keypoints": poses["keypoints"][start:stop, slots],
            "gt_keypoint_valid": poses["keypoint_valid"][start:stop, slots] & seen[..., None],
            "gt_root": poses["root"][start:stop, slots],
            "gt_root_valid": poses["root_valid"][start:stop, slots] & seen,
            "visibility": observed[:, slots].transpose(0, 2, 1).astype(np.float32),
        }
        out = {"imgs": imgs, "world2img": world2img, "intrinsics": intrinsics,
               "seq": f"gym/{name}/views" + "".join(str(self.camera_ids[v]) for v in views),
               "camera_ids": [self.camera_ids[v] for v in views],
               "start": int(poses["frames"][start]),
               "frame_ids": poses["frames"][start:stop].copy(),
               "gt_keypoint_frame_valid": (poses["labeled"][start:stop]
                                           | poses.get("interpolated", np.zeros(poses["valid"].shape, bool))
                                           [start:stop].any(-1)),
               "capture_group": self.capture_group}
        if "keypoint_sigma" in poses:
            # Approximate GT (data/golden.py): per-target uncertainty and 2-D detections for
            # people one camera sees but that could not be triangulated.
            targets["gt_keypoint_sigma"] = poses["keypoint_sigma"][start:stop, slots]
            targets["gt_root_sigma"] = poses["root_sigma"][start:stop, slots]
        if "match_root" in poses:
            # Interpolated roots between labels: a matching cue only, never a loss target.
            targets["gt_match_root"] = poses["match_root"][start:stop, slots]
            targets["gt_match_root_valid"] = poses["match_root_valid"][start:stop, slots] & seen
        if "kp2d" in poses:
            view_index = list(views)
            kp2d = poses["kp2d"][start:stop][:, slots][:, :, view_index]            # (T, S, n, J, 3)
            kp2d_valid = poses["kp2d_valid"][start:stop][:, slots][:, :, view_index] & seen[:, :, None, None]
            kp2d = kp2d.copy()
            for ni, (K0, K1) in rescale.items():
                kp2d[:, :, ni, :, 0] = (kp2d[:, :, ni, :, 0] - K0[0, 2]) * K1[0, 0] / K0[0, 0] + K1[0, 2]
                kp2d[:, :, ni, :, 1] = (kp2d[:, :, ni, :, 1] - K0[1, 2]) * K1[1, 1] / K0[1, 1] + K1[1, 2]
            targets["gt_kp2d"] = kp2d.transpose(0, 2, 1, 3, 4)
            targets["gt_kp2d_valid"] = kp2d_valid.transpose(0, 2, 1, 3)
        for key, values in targets.items():
            if key == "visibility":
                padded = np.zeros((length, n, people), dtype=np.float32)
                padded[..., :len(slots)] = values
            elif key in ("gt_kp2d", "gt_kp2d_valid"):
                padded = np.zeros((length, n, people, *values.shape[3:]), dtype=values.dtype)
                padded[:, :, :len(slots)] = values
            else:
                padded = np.zeros((length, people, *values.shape[2:]), dtype=values.dtype)
                padded[:, :len(slots)] = values
            out[key] = padded
        instance_ids = np.full((length, people), -1, dtype=np.int64)
        instance_ids[:, :len(slots)] = poses["person_ids"][slots]
        out["instance_ids"] = instance_ids
        # Present in some reviewed view but possibly not the chosen ones: tracks
        # stay alive without supervision (see models/tracking.py).
        exists = np.zeros((length, people), dtype=bool)
        exists[:, :len(slots)] = present | seen
        out["instance_exists"] = exists
        return out


def _to_sample(raw):
    from .dataset import _with_sparse_targets
    out = {key: (value if isinstance(value, (torch.Tensor, str, int)) or key == "camera_ids"
                 else torch.as_tensor(value))
           for key, value in raw.items()}
    return _with_sparse_targets(out)


class GymPseudo3DDataset(Dataset):
    """Gym windows as model-ready samples; ``count`` repeats windows uniformly."""

    def __init__(self, cfg, split="train", count=None):
        self.provider = GymPseudo3DProvider(cfg, split)
        samples = self.provider.list_samples()
        if not samples:
            raise ValueError(f"No label-containing gym windows for {split}")
        if count:
            order = np.random.default_rng(cfg.data.split_seed).permutation(len(samples))
            samples = [samples[int(order[i % len(order)])] for i in range(count)]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return _to_sample(self.provider.load_window(self.samples[index]))


class AlignedSingleDataset(Dataset):
    """Single-person aligned recordings (``human_prior_aligned``) shaped for the mix collate.

    ``count`` repeats/subsamples windows with a fixed permutation so the
    component holds an exact share of the mixture.
    """

    def __init__(self, cfg, split="train", count=None, limit=None):
        import copy

        from .dataset import MultiViewPoseDataset
        if not cfg.data.gym_mix_aligned_dirs:
            raise ValueError("AlignedSingleDataset requires data.gym_mix_aligned_dirs")
        aligned = copy.deepcopy(cfg)
        aligned.data.provider = "human_prior_aligned"
        aligned.data.composition_manifest = None
        aligned.data.human_prior_aligned_dirs = list(cfg.data.gym_mix_aligned_dirs)
        self.base = MultiViewPoseDataset(aligned, split)
        order = np.random.default_rng(cfg.data.split_seed).permutation(len(self.base))
        self.indices = [int(order[i % len(order)]) for i in range(count or len(order))][:limit]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return decorate_aligned(self.base[self.indices[index]])


def decorate_aligned(sample):
    """Mix-collate metadata for single-person aligned windows (identity = slot)."""
    valid = sample["gt_valid"]
    t, n = sample["imgs"].shape[:2]
    sample["instance_ids"] = torch.where(
        valid, torch.arange(valid.shape[1]).expand_as(valid), torch.full_like(valid, -1, dtype=torch.long))
    sample["visibility"] = valid[:, None, :].expand(t, n, valid.shape[1]).float().clone()
    sample["capture_group"] = "aligned"
    return sample


class GymMocapMixDataset(Dataset):
    """Composition-manifest mocap (the baseline data path) plus oversampled gym.

    Optional third component: single-person aligned recordings
    (``gym_mix_aligned_dirs``) at ``gym_mix_aligned_fraction`` of the mix.
    """

    def __init__(self, cfg, split="train"):
        if not cfg.data.composition_manifest:
            raise ValueError("gym_mocap_mix requires data.composition_manifest (baseline mocap path)")
        fraction = cfg.data.gym_sample_fraction
        aligned_fraction = cfg.data.gym_mix_aligned_fraction if cfg.data.gym_mix_aligned_dirs else 0.0
        if not 0 < fraction < 1:
            raise ValueError("gym_sample_fraction must be strictly between zero and one")
        if aligned_fraction < 0 or fraction + aligned_fraction >= 1:
            raise ValueError("gym_sample_fraction + gym_mix_aligned_fraction must be below one")
        if cfg.data.gym_mix_aligned_dirs and aligned_fraction == 0:
            raise ValueError("gym_mix_aligned_dirs is set but gym_mix_aligned_fraction is zero")
        from .dataset import MultiViewPoseDataset
        self.mocap = MultiViewPoseDataset(cfg, split)
        total = len(self.mocap) / (1 - fraction - aligned_fraction)
        self.gym = GymPseudo3DDataset(cfg, split, count=max(1, round(total * fraction)))
        self.aligned = (AlignedSingleDataset(cfg, split, count=max(1, round(total * aligned_fraction)))
                        if aligned_fraction else None)

    def __len__(self):
        return len(self.mocap) + len(self.gym) + (len(self.aligned) if self.aligned else 0)

    def __getitem__(self, index):
        if index < len(self.mocap):
            return self.mocap[index]
        index -= len(self.mocap)
        if index < len(self.gym):
            return self.gym[index]
        if self.aligned is None:
            raise IndexError(index)
        return self.aligned[index - len(self.gym)]
