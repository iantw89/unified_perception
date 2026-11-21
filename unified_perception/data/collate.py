"""Batch collation with masked camera padding and metadata lists."""

from __future__ import annotations

import torch

from ..constants import ROOT_JOINTS


def _tensor(value, *, dtype=None):
    return value if isinstance(value, torch.Tensor) and dtype is None else torch.as_tensor(value, dtype=dtype)


def collate_batch(samples):
    out = {}
    counts = [sample["imgs"].shape[1] for sample in samples]
    if min(counts) < 1:
        raise ValueError("Every sample must contain at least one camera")
    cameras = max(counts)

    def pad_cameras(value, *, calibration=False):
        tensor = _tensor(value)
        if tensor.shape[1] == cameras:
            return tensor
        shape = (tensor.shape[0], cameras - tensor.shape[1], *tensor.shape[2:])
        padding = tensor.new_zeros(shape)
        if calibration:
            padding[:] = torch.eye(4, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, padding), dim=1)

    for key in ("imgs", "world2img", "intrinsics"):
        out[key] = torch.stack([
            pad_cameras(sample[key], calibration=key != "imgs") for sample in samples
        ])
    out["camera_valid"] = torch.stack([
        (torch.arange(cameras) < count).expand(sample["imgs"].shape[0], -1)
        & (pad_cameras(sample["camera_mask"]).bool() if "camera_mask" in sample else True)
        for sample, count in zip(samples, counts)
    ])
    tensor_keys = ["gt_keypoints", "gt_valid"]
    for key in tensor_keys:
        out[key] = torch.stack([_tensor(sample[key]) for sample in samples], dim=0)

    frame_valid, joint_valid, roots, root_valid = [], [], [], []
    for sample in samples:
        kpts = _tensor(sample["gt_keypoints"])
        valid = _tensor(sample["gt_valid"], dtype=torch.bool)
        frame = _tensor(
            sample.get("gt_keypoint_frame_valid", torch.ones(kpts.shape[0], dtype=torch.bool)),
            dtype=torch.bool,
        )
        frame_valid.append(frame)
        joint = _tensor(
            sample.get(
                "gt_keypoint_valid",
                valid.unsqueeze(-1).expand(*valid.shape, kpts.shape[-2]),
            ),
            dtype=torch.bool,
        )
        joint_valid.append(joint & valid.unsqueeze(-1) & frame[:, None, None])
        if "gt_root" in sample:
            roots.append(_tensor(sample["gt_root"], dtype=kpts.dtype))
        else:
            roots.append(kpts[..., list(ROOT_JOINTS), :].mean(dim=-2))
        root_valid.append(_tensor(sample.get("gt_root_valid", valid), dtype=torch.bool) & valid)

    out["gt_keypoint_frame_valid"] = torch.stack(frame_valid, dim=0)
    out["gt_keypoint_valid"] = torch.stack(joint_valid, dim=0)
    out["gt_root"] = torch.stack(roots, dim=0)
    out["gt_root_valid"] = torch.stack(root_valid, dim=0)
    out["seq"] = [s["seq"] for s in samples]
    out["start"] = [s["start"] for s in samples]
    for key in ("instance_ids", "visibility"):
        if any(key in s for s in samples):
            if not all(key in s for s in samples):
                raise ValueError(f"Inconsistent optional metadata: {key}")
            out[key] = torch.stack([
                pad_cameras(s[key]) if key == "visibility" else _tensor(s[key])
                for s in samples
            ])
    # Approximate-GT extras (data/golden.py).  Exact-GT datasets get zero uncertainty and no
    # 2-D targets, so they are supervised exactly as before.
    if any("gt_keypoint_sigma" in s for s in samples):
        out["gt_keypoint_sigma"] = torch.stack([
            _tensor(s["gt_keypoint_sigma"], dtype=torch.float32) if "gt_keypoint_sigma" in s
            else torch.zeros(_tensor(s["gt_keypoints"]).shape[:-1]) for s in samples])
        out["gt_root_sigma"] = torch.stack([
            _tensor(s["gt_root_sigma"], dtype=torch.float32) if "gt_root_sigma" in s
            else torch.zeros(_tensor(s["gt_valid"]).shape) for s in samples])
    if any("gt_match_root" in s for s in samples):
        out["gt_match_root"] = torch.stack([
            _tensor(s["gt_match_root"], dtype=torch.float32) if "gt_match_root" in s
            else torch.zeros((*_tensor(s["gt_valid"]).shape, 3)) for s in samples])
        out["gt_match_root_valid"] = torch.stack([
            _tensor(s["gt_match_root_valid"], dtype=torch.bool) if "gt_match_root_valid" in s
            else torch.zeros(_tensor(s["gt_valid"]).shape, dtype=torch.bool) for s in samples])
    if any("gt_kp2d" in s for s in samples):
        kp2d, kp2d_valid = [], []
        for s in samples:
            T, P, J = _tensor(s["gt_keypoints"]).shape[:3]
            n = s["imgs"].shape[1]
            value = _tensor(s["gt_kp2d"], dtype=torch.float32) if "gt_kp2d" in s else torch.zeros(T, n, P, J, 3)
            valid = (_tensor(s["gt_kp2d_valid"], dtype=torch.bool) if "gt_kp2d_valid" in s
                     else torch.zeros(T, n, P, J, dtype=torch.bool))
            kp2d.append(pad_cameras(value))
            kp2d_valid.append(pad_cameras(valid))
        out["gt_kp2d"] = torch.stack(kp2d)
        out["gt_kp2d_valid"] = torch.stack(kp2d_valid)
    if any("instance_exists" in s for s in samples):
        # Providers without it observe everyone they list: existence == validity.
        out["instance_exists"] = torch.stack([
            _tensor(s.get("instance_exists", s["gt_valid"]), dtype=torch.bool) for s in samples
        ])
    if any("stream_reset" in s for s in samples):
        from .streaming import collate_stream_flags
        out.update(collate_stream_flags(samples))
    if all("capture_group" in s for s in samples):
        out["capture_group"] = [s["capture_group"] for s in samples]
    return out
