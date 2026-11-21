"""Data providers for unified_perception.

- human_prior_aligned: consumes human_prior alignment outputs directly
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import cv2
import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple
from torchvision.transforms import v2
import numpy as np

from ..config import Config
from ..constants import JOINT_INDEX, NUM_JOINTS, ROOT_JOINTS
from .interface import DataProvider, SampleRef
from .registry import register_provider
from .transforms import resize_image_and_intrinsics

if TYPE_CHECKING:
    from torchcodec.decoders import VideoDecoder  # type: ignore[reportPrivateImportUsage]


class DatasetConfigProtocol(Protocol):
    camera_fps: int
    mocap_fps: int

    def get_camera_definitions(self) -> Dict[str, Dict[str, Any]]:
        ...

    def get_images_for_camera(
        self,
        cam_id: str,
        max_images: int = 0,
        requested_mocap_indices: Optional[set[int]] = None,
    ) -> Iterable[Tuple[int, np.ndarray]]:
        ...

    def has_images(self, cam_id: str) -> bool:
        ...

    def get_camera_image_offset(self, cam_id: str) -> int:
        ...


def _window_starts(num_frames: int, seq_len: int, stride: int) -> List[int]:
    if num_frames <= 0:
        return []
    if num_frames <= seq_len:
        return [0]
    return list(range(0, num_frames - seq_len + 1, max(1, stride)))


def _split_sequences(seq_ids: List[str], split: str, val_fraction: float, seed: int) -> List[str]:
    if split not in ("train", "val"):
        raise ValueError("Raw provider supports train/val only; use an explicit composition manifest for test")
    seq_ids = sorted(seq_ids)
    rng = random.Random(seed)
    rng.shuffle(seq_ids)
    n_val = max(1, int(round(len(seq_ids) * val_fraction))) if seq_ids else 0
    val_set = set(seq_ids[:n_val])
    if split == "val":
        return [s for s in seq_ids if s in val_set]
    return [s for s in seq_ids if s not in val_set]


@dataclass
class HumanPriorSequence:
    seq_id: str
    out_dir: str
    source: str
    source_id: str
    num_frames: int
    stride: int
    mocap_indices: Optional[List[int]] = None


@register_provider("human_prior_aligned")
class HumanPriorAlignedProvider(DataProvider):
    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.target_fps = cfg.data.human_prior_target_fps
        self.seq_len = cfg.data.seq_len
        self.num_cameras = cfg.data.num_cameras
        self.out_hw = (cfg.data.image_height, cfg.data.image_width)
        self.max_people = cfg.data.max_people
        self.mean = cfg.data.pixel_mean
        self.std = cfg.data.pixel_std
        self.gpu_decode = cfg.data.gpu_decode

        if not cfg.data.human_prior_aligned_dirs:
            raise ValueError("data.human_prior_aligned_dirs must be set for human_prior_aligned provider")

        self._append_human_prior_repo_to_path(cfg.data.human_prior_repo)
        self._seqs = self._discover_sequences(cfg.data.human_prior_aligned_dirs)
        seq_ids = _split_sequences(
            list(self._seqs.keys()), split, cfg.data.val_fraction, cfg.data.split_seed
        )

        self._dataset_cfg_cache: Dict[str, DatasetConfigProtocol] = {}
        self._cam_defs_cache: Dict[str, Dict[str, Any]] = {}
        self._mocap_cache: Dict[str, List[Dict[str, Any]]] = {}

        self.samples: List[SampleRef] = []
        for seq_id in seq_ids:
            for start in _window_starts(self._seqs[seq_id].num_frames, self.seq_len, self.cfg.data.stride):
                self.samples.append((seq_id, start))

    def list_samples(self) -> List[SampleRef]:
        return self.samples

    def _append_human_prior_repo_to_path(self, repo_path: str) -> None:
        abs_repo = os.path.abspath(repo_path)
        if abs_repo not in sys.path:
            sys.path.insert(0, abs_repo)

    def _discover_sequences(self, roots: List[str]) -> Dict[str, HumanPriorSequence]:
        seqs: Dict[str, HumanPriorSequence] = {}
        for root in roots:
            abs_root = os.path.abspath(root)
            if not os.path.isdir(abs_root):
                continue
            for name in sorted(os.listdir(abs_root)):
                seq_dir = os.path.join(abs_root, name)
                if not os.path.isdir(seq_dir):
                    continue
                mkp = os.path.join(seq_dir, "mocap_keypoints.json")
                if not os.path.exists(mkp):
                    continue
                with open(mkp, "r") as f:
                    frames = json.load(f)
                traceability_path = os.path.join(seq_dir, "traceability.json")
                stride = 1
                indices = None
                if os.path.exists(traceability_path):
                    with open(traceability_path, "r") as f:
                        traceability = json.load(f)
                    indices = traceability.get("source_mocap_indices", [])
                    if len(indices) != len(frames) or any(b <= a for a, b in zip(indices, indices[1:])):
                        raise ValueError(f"Invalid source frame traceability in {seq_dir}")
                    if len(indices) >= 2:
                        stride = int(indices[1]) - int(indices[0])
                source, source_id = _parse_human_prior_source_id(name)
                seq_id = f"{os.path.basename(abs_root)}/{name}"
                seqs[seq_id] = HumanPriorSequence(
                    seq_id=seq_id,
                    out_dir=seq_dir,
                    source=source,
                    source_id=source_id,
                    num_frames=len(frames),
                    stride=stride,
                    mocap_indices=indices,
                )
        if not seqs:
            raise RuntimeError(
                "No usable sequence dirs found. Expected directories containing "
                "mocap_keypoints.json and alignment_results.json"
            )
        return seqs

    def _dataset_cfg_for_seq(self, seq: HumanPriorSequence) -> DatasetConfigProtocol:
        if seq.seq_id in self._dataset_cfg_cache:
            return self._dataset_cfg_cache[seq.seq_id]

        from src.data.align_camera_mocap.dataset_configs import (  # type: ignore
            Human36MConfig,
            MoyoConfig,
            MpiInf3dhpConfig,
        )

        if seq.source == "h36m":
            if not self.cfg.data.human_prior_h36m_dataset_path:
                raise ValueError("data.human_prior_h36m_dataset_path is required for H36M sequences")
            subj, act, sub = _parse_h36m_ids(seq.source_id)
            ds_cfg = Human36MConfig(
                self.cfg.data.human_prior_h36m_dataset_path,
                subject_idx=subj,
                action_idx=act,
                subaction_idx=sub,
            )
        elif seq.source == "mpi":
            if not self.cfg.data.human_prior_mpi_dataset_path:
                raise ValueError("data.human_prior_mpi_dataset_path is required for MPI sequences")
            subj, sq = _parse_mpi_ids(seq.source_id)
            ds_cfg = MpiInf3dhpConfig(self.cfg.data.human_prior_mpi_dataset_path, subject=subj, seq=sq)
        else:
            if not self.cfg.data.human_prior_moyo_dataset_path:
                raise ValueError("data.human_prior_moyo_dataset_path is required for MOYO sequences")
            ds_cfg = MoyoConfig(self.cfg.data.human_prior_moyo_dataset_path, sequence_name=seq.source_id)

        self._dataset_cfg_cache[seq.seq_id] = ds_cfg
        return ds_cfg

    def _cam_defs_for_seq(self, seq: HumanPriorSequence) -> Dict[str, Any]:
        if seq.seq_id not in self._cam_defs_cache:
            ds_cfg = self._dataset_cfg_for_seq(seq)
            cam_defs = ds_cfg.get_camera_definitions()
            _apply_alignment_camera_correction(seq.out_dir, cam_defs)
            cam_defs = self._filter_cams_by_coverage(seq, ds_cfg, cam_defs)
            self._cam_defs_cache[seq.seq_id] = cam_defs
        return self._cam_defs_cache[seq.seq_id]

    def _filter_cams_by_coverage(
        self,
        seq: HumanPriorSequence,
        ds_cfg: DatasetConfigProtocol,
        cam_defs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Remove cameras whose videos don't cover the full sequence frame range.

        Checks each camera's video length against the maximum video frame index
        required by any window in the sequence.  Excluded cameras are logged once
        here instead of producing per-window warnings during training.
        """
        seq_name = seq.seq_id.split("/", 1)[-1]
        segment_start = _segment_start_from_seq_name(seq_name)
        last_src_mocap = segment_start + (seq.num_frames - 1) * seq.stride
        fps_ratio = max(1, ds_cfg.mocap_fps // ds_cfg.camera_fps)

        valid: Dict[str, Any] = {}
        excluded: list = []
        for cam_id, cam_def in cam_defs.items():
            try:
                vp = _video_path_for_camera(ds_cfg, cam_id)
            except FileNotFoundError:
                excluded.append((cam_id, "video not found"))
                continue
            cam_offset = ds_cfg.get_camera_image_offset(cam_id)
            max_needed = last_src_mocap // fps_ratio - cam_offset
            # Use cv2 for frame count: works without FFmpeg shared DLLs.
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total <= 0 or max_needed >= total:
                excluded.append((cam_id, f"video too short ({total} frames, need >{max_needed})"))
            else:
                valid[cam_id] = cam_def

        if excluded:
            import warnings
            details = "; ".join(f"{c}: {r}" for c, r in excluded)
            warnings.warn(
                f"[HumanPriorAlignedProvider] excluded {len(excluded)} camera(s) from "
                f"{seq.seq_id}: {details}"
            )
        return valid

    def _mocap_frames(self, seq: HumanPriorSequence) -> List[Dict[str, Any]]:
        if seq.seq_id not in self._mocap_cache:
            with open(os.path.join(seq.out_dir, "mocap_keypoints.json"), "r") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError(f"Invalid mocap_keypoints format for {seq.seq_id}: expected list")
            self._mocap_cache[seq.seq_id] = loaded
        return self._mocap_cache[seq.seq_id]

    def _desired_frame_indices(
        self,
        ds_cfg: DatasetConfigProtocol,
        sample: SampleRef,
        cam_id: str,
    ) -> List[int]:
        """Compute 0-based video frame indices for the requested window on one camera."""
        seq_id, start = sample
        seq = self._seqs[seq_id]
        seq_name = seq_id.split("/", 1)[-1]
        segment_start = _segment_start_from_seq_name(seq_name)
        requested_source_mocap = (
            seq.mocap_indices[start:start + self.seq_len] if seq.mocap_indices is not None
            else [segment_start + (f + start) * seq.stride for f in range(self.seq_len)]
        )
        # Map source mocap index → 0-based video frame index.
        # fps_ratio = mocap_fps / camera_fps (integer, ≥1 for all current datasets).
        # Floor division avoids banker's-rounding artefacts: for MOYO (ratio=2) both
        # mocap frame 890 and 891 correctly land on video frame 445 (before offset),
        # whereas float round(891*0.5)=446 is one too high at sequence boundaries.
        # cam_offset = first_frame_index - 1: the 0-based start offset of the video
        # within the original image-file sequence (non-zero for MOYO).
        fps_ratio = max(1, ds_cfg.mocap_fps // ds_cfg.camera_fps)
        cam_offset = ds_cfg.get_camera_image_offset(cam_id)
        return [int(i // fps_ratio - cam_offset) for i in requested_source_mocap]

    def _load_window_images(
        self,
        ds_cfg: DatasetConfigProtocol,
        sample: SampleRef,
        candidate_cam_ids: List[str],
        num_cameras: int,
    ) -> Tuple[List[str], Dict[str, List[torch.Tensor]]]:
        """Load images for the requested window, skipping cameras with truncated videos.

        Returns the list of successfully decoded camera IDs (length == num_cameras)
        and the corresponding image map.  Raises ValueError if fewer than num_cameras
        valid cameras are available.
        """
        from torchcodec.decoders import VideoDecoder  # type: ignore[reportPrivateImportUsage]

        image_map: Dict[str, List[torch.Tensor]] = {}
        skipped: List[Tuple[str, str]] = []

        for cam_id in candidate_cam_ids:
            if len(image_map) >= num_cameras:
                break

            frame_indices = self._desired_frame_indices(ds_cfg, sample, cam_id)
            video_path = _video_path_for_camera(ds_cfg, cam_id)
            decoder = VideoDecoder(video_path, device="cuda" if self.gpu_decode else "cpu")
            try:
                decoded, _ = _decode_video_frames_torchcodec(decoder, video_path, frame_indices)
                image_map[str(cam_id)] = decoded
            except (ValueError, FileNotFoundError) as exc:
                skipped.append((cam_id, f"{str(exc)} {frame_indices}"))

        if len(image_map) < num_cameras:
            raise ValueError(
                f"Only {len(image_map)}/{num_cameras} valid cameras for "
                f"{sample[0]} start={sample[1]} (skipped: {skipped})"
            )
        used_cam_ids = list(image_map.keys())
        return used_cam_ids, image_map

    def load_window(self, sample: SampleRef, camera_ids: Optional[List[str]] = None) -> dict:
        seq_id, start = sample
        seq = self._seqs[seq_id]
        ds_cfg = self._dataset_cfg_for_seq(seq)
        frames = self._mocap_frames(seq)[start: start + self.seq_len]

        cam_defs = self._cam_defs_for_seq(seq)
        if not cam_defs:
            raise RuntimeError(f"No camera definitions for sequence {seq_id}")

        if seq.num_frames <= 0:
            raise RuntimeError(f"Sequence has no frames: {seq_id}")

        # Shuffle all cameras so the camera-level fallback (for truncated videos)
        # picks different cameras each call rather than always trying in sorted order.
        all_cam_ids = list(cam_defs.keys()) if camera_ids is None else list(camera_ids)
        if camera_ids is None:
            np.random.shuffle(all_cam_ids)
        wanted = (min(self.num_cameras, len(cam_defs))
                  if self.cfg.data.allow_fewer_cameras else self.num_cameras)
        if camera_ids is not None and (len(all_cam_ids) != wanted or len(set(all_cam_ids)) != len(all_cam_ids) or any(c not in cam_defs for c in all_cam_ids)):
            raise ValueError(f"Invalid fixed camera set for {seq_id}: {all_cam_ids}")
        cam_ids, images_rgb = self._load_window_images(
            ds_cfg, sample, all_cam_ids, wanted
        )

        keypoints = np.zeros((len(frames), self.max_people, NUM_JOINTS, 3), dtype=np.float32)
        valid = np.zeros((len(frames), self.max_people), dtype=bool)
        for fi, frame in enumerate(frames):
            person = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
            present = False
            for jname, jidx in JOINT_INDEX.items():
                value = frame.get(jname)
                if value is None:
                    continue
                person[jidx] = np.asarray(value, dtype=np.float32)
                present = True
            if present:
                keypoints[fi, 0] = person
                valid[fi, 0] = True

        K_by_cam = np.stack([np.asarray(cam_defs[c]["K"], dtype=np.float64) for c in cam_ids], axis=0)  # type: ignore[index]
        R_by_cam = np.stack([np.asarray(cam_defs[c]["R"], dtype=np.float64) for c in cam_ids], axis=0)
        T_by_cam = np.stack([np.asarray(cam_defs[c]["T"], dtype=np.float64).reshape(3) for c in cam_ids], axis=0)

        def _prepare_img(frame_idx: int, cam_idx: int, K: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
            rgb = images_rgb[cam_ids[cam_idx]][frame_idx]
            rgb, K2 = resize_image_and_intrinsics(rgb, K, self.out_hw)
            img = rgb.permute(2, 0, 1).contiguous()
            return img, K2

        output = _build_output_window(
            seq=seq_id,
            start=start,
            seq_len=self.seq_len,
            out_hw=self.out_hw,
            max_people=self.max_people,
            num_cameras=len(cam_ids),
            keypoints=keypoints,
            valid=valid,
            K_by_cam=K_by_cam,
            R_by_cam=R_by_cam,
            T_by_cam=T_by_cam,
            image_loader=_prepare_img,
        )
        output["camera_ids"] = cam_ids
        output["frame_ids"] = np.asarray(
            self._desired_frame_indices(ds_cfg, sample, cam_ids[0]), dtype=np.int64
        )
        return output


    def _get_video_decoder(self, ds_cfg: DatasetConfigProtocol, cam_id: str) -> Tuple[VideoDecoder, Tuple[int, int]]:
        from torchcodec.decoders import VideoDecoder  # type: ignore[reportPrivateImportUsage]

        cap = cv2.VideoCapture(_video_path_for_camera(ds_cfg, cam_id))
        assert cap.isOpened()
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if height > self.out_hw[0] or width > self.out_hw[1]:
            max_ratio = max(height / self.out_hw[0], width / self.out_hw[1])
            new_h = int(round(height / max_ratio))
            new_w = int(round(width / max_ratio))
        else:
            min_ratio = min(height / self.out_hw[0], width / self.out_hw[1])
            new_h = int(round(height / min_ratio))
            new_w = int(round(width / min_ratio))
        # Some videos have a strange resolution (1000, 1002), so if new resolution is not equal but is
        # very close, we just force them to be equal
        # TODO: Fix this more elegantly (perhaps resize source videos, or just crop).
        if width == 1000 and height == 1002:
            new_h = new_w = max(new_h, new_w)

        decoder = VideoDecoder(
            _video_path_for_camera(ds_cfg, cam_id),
            device="cuda" if self.gpu_decode else "cpu",
            transforms=[v2.Resize(size=(new_h, new_w), interpolation=v2.InterpolationMode.BILINEAR)],
            num_ffmpeg_threads=2,
        )
        return decoder, (height, width)

    def load_full_sequence(
        self, seq_id: str, num_cameras: int, chunk_frames: int
    ) -> Iterator[Dict[str, Any]]:
        """Decode a sequence for the shuffle buffer, yielding one chunk at a time.

        Shared setup (mocap JSON read, camera selection, fps-ratio arithmetic)
        happens once before any chunk is decoded.

        Parameters
        ----------
        seq_id:
            Key in ``self._seqs`` identifying the sequence.
        num_cameras:
            Number of cameras to include (randomly chosen from valid cameras).

        Yields
        ------
        dict with keys:
            imgs        (Tc, N, 3, H, W)  uint8  – resized RGB, CHW
            world2img   (Tc, N, 4, 4)     float32
            intrinsics  (Tc, N, 4, 4)     float32
            gt_keypoints (Tc, P, J, 3)    float32
            gt_valid    (Tc, P)           bool
            seq_id      str
            T           int              – number of frames in this chunk
        """
        seq = self._seqs[seq_id]
        ds_cfg = self._dataset_cfg_for_seq(seq)
        all_mocap_frames = self._mocap_frames(seq)

        cam_defs = self._cam_defs_for_seq(seq)
        if len(cam_defs) < num_cameras:
            raise ValueError(
                f"Only {len(cam_defs)} valid cameras for {seq_id}, need {num_cameras}"
            )

        seq_name = seq_id.split("/", 1)[-1]
        segment_start = _segment_start_from_seq_name(seq_name)
        fps_ratio = max(1, ds_cfg.mocap_fps // ds_cfg.camera_fps)

        # Some cameras (cam_offset > 0) cannot provide frames for the first few
        # source mocap indices.  Compute the tightest valid starting frame index
        # across all candidate cameras to avoid negative video-frame indices.
        all_cam_ids = list(cam_defs.keys())
        first_valid_f = 0
        for _cid in all_cam_ids:
            _off = ds_cfg.get_camera_image_offset(_cid)
            if _off * fps_ratio > segment_start:
                _fvf = math.ceil((_off * fps_ratio - segment_start) / seq.stride)
                first_valid_f = max(first_valid_f, _fvf)
        if first_valid_f > 0:
            all_mocap_frames = all_mocap_frames[first_valid_f:]
            if not all_mocap_frames:
                raise ValueError(
                    f"Sequence {seq_id} has 0 valid frames after trimming "
                    f"{first_valid_f} start frames (cam_offset conflict)."
                )

        T = len(all_mocap_frames)
        all_source_mocap = [
            segment_start + (first_valid_f + f) * seq.stride for f in range(T)
        ]

        # Shuffle cameras once so the same subset is used for every chunk.
        np.random.shuffle(all_cam_ids)

        N = num_cameras
        H, W = self.out_hw

        decoders = {}
        native_hw_by_cam: Dict[str, Optional[Tuple[int, int]]] = {}
        for chunk_start in range(0, T, chunk_frames):
            chunk_end = min(chunk_start + chunk_frames, T)
            chunk_mocap  = all_mocap_frames[chunk_start:chunk_end]
            chunk_source = all_source_mocap[chunk_start:chunk_end]
            Tc = chunk_end - chunk_start

            # Decode each camera's frames one at a time, resizing to out_hw
            # immediately.  This keeps peak memory at 1 camera × Tc frames at
            # native resolution rather than num_cameras × Tc × native_resolution,
            # which would exceed 12 GB for 256-frame chunks of 2048-px MPI videos.
            image_data_rgb: Dict[str, List[torch.Tensor]] = {}
            for cam_id in all_cam_ids:
                if len(image_data_rgb) >= num_cameras:
                    break
                if cam_id not in decoders:
                    decoders[cam_id], native_hw = self._get_video_decoder(ds_cfg, cam_id)
                    native_hw_by_cam[cam_id] = native_hw
                cam_offset = ds_cfg.get_camera_image_offset(cam_id)
                frame_indices = [i // fps_ratio - cam_offset for i in chunk_source]
                video_path = _video_path_for_camera(ds_cfg, cam_id)
                try:
                    frames, _ = _decode_video_frames_torchcodec(
                        decoders[cam_id],
                        video_path, frame_indices,
                        chunk_frames=chunk_frames,
                        out_hw=self.out_hw,
                    )
                    image_data_rgb[cam_id] = frames
                except (ValueError, FileNotFoundError):
                    pass

            if len(image_data_rgb) < num_cameras:
                # Not enough cameras for this chunk; skip it.
                continue

            cam_ids = list(image_data_rgb.keys())

            imgs       = torch.zeros((Tc, N, 3, H, W), dtype=torch.uint8)
            world2img  = np.zeros((Tc, N, 4, 4), dtype=np.float32)
            intrinsics = np.zeros((Tc, N, 4, 4), dtype=np.float32)

            for ci, cam_id in enumerate(cam_ids):
                K  = np.asarray(cam_defs[cam_id]["K"], dtype=np.float64)
                R  = np.asarray(cam_defs[cam_id]["R"], dtype=np.float64)
                Tv = np.asarray(cam_defs[cam_id]["T"], dtype=np.float64).reshape(3)

                # K2: if frames were pre-resized, compute from native dimensions;
                # otherwise use the first decoded frame (native resolution).
                native_hw = native_hw_by_cam[cam_id]
                if native_hw is not None:
                    native_h, native_w = native_hw
                    out_h, out_w = self.out_hw
                    s = min(out_w / native_w, out_h / native_h)
                    new_w2 = int(round(native_w * s))
                    new_h2 = int(round(native_h * s))
                    pad_top2  = (out_h - new_h2) // 2
                    pad_left2 = (out_w - new_w2) // 2
                    K2 = K.copy()
                    K2[0, :] *= s;  K2[0, 2] += pad_left2
                    K2[1, :] *= s;  K2[1, 2] += pad_top2
                else:
                    _, K2 = resize_image_and_intrinsics(image_data_rgb[cam_id][0], K, self.out_hw)

                extr = np.eye(4, dtype=np.float64)
                extr[:3, :3] = R
                extr[:3, 3]  = Tv
                k4 = np.eye(4, dtype=np.float64)
                k4[:3, :3] = K2
                w2i  = (k4 @ extr).astype(np.float32)
                intr = k4.astype(np.float32)

                for ti in range(Tc):
                    rgb: torch.Tensor = image_data_rgb[cam_id][ti]
                    if native_hw is None:
                        # Frames at native resolution — resize now.
                        resized, _ = resize_image_and_intrinsics(rgb, K, self.out_hw)
                        rgb = resized if isinstance(resized, torch.Tensor) else torch.as_tensor(resized)
                    imgs[ti, ci] = rgb.permute(2, 0, 1)  # HWC → CHW uint8
                    world2img[ti, ci]  = w2i
                    intrinsics[ti, ci] = intr

            keypoints = np.zeros((Tc, self.max_people, NUM_JOINTS, 3), dtype=np.float32)
            gt_valid  = np.zeros((Tc, self.max_people), dtype=bool)
            for fi, frame in enumerate(chunk_mocap):
                person  = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
                present = False
                for jname, jidx in JOINT_INDEX.items():
                    value = frame.get(jname)
                    if value is None:
                        continue
                    person[jidx] = np.asarray(value, dtype=np.float32)
                    present = True
                if present:
                    keypoints[fi, 0] = person
                    gt_valid[fi, 0]  = True

            yield {
                "imgs":         imgs,        # (Tc, N, 3, H, W) uint8
                "world2img":    world2img,   # (Tc, N, 4, 4)    float32
                "intrinsics":   intrinsics,  # (Tc, N, 4, 4)    float32
                "gt_keypoints": keypoints,   # (Tc, P, J, 3)    float32
                "gt_valid":     gt_valid,    # (Tc, P)           bool
                "seq_id":       seq_id,
                "T":            Tc,
            }


_SPLIT_SEGMENT_RE = re.compile(
    r"^(?P<base>.+)__seg(?P<idx>\d+)_(?P<start>-?\d+)_(?P<end>-?\d+)$"
)


def _parse_human_prior_source_id(seq_name: str) -> Tuple[str, str]:
    base_name = seq_name

    m = _SPLIT_SEGMENT_RE.match(seq_name)
    if m:
        base_name = m.group("base")

    if base_name.startswith("subj_"):
        return "h36m", base_name
    if base_name.startswith("S") and "_Seq" in base_name:
        return "mpi", base_name
    return "moyo", base_name


def _segment_start_from_seq_name(seq_name: str) -> int:
    m = _SPLIT_SEGMENT_RE.match(seq_name)
    if m is None:
        return 0
    return int(m.group("start"))


def _parse_h36m_ids(name: str) -> Tuple[int, int, int]:
    # subj_9_act_2_sub_1
    parts = name.split("_")
    return int(parts[1]), int(parts[3]), int(parts[5])


def _parse_mpi_ids(name: str) -> Tuple[int, int]:
    # S1_Seq2
    left, right = name.split("_Seq")
    return int(left[1:]), int(right)


def _video_path_for_camera(ds_cfg: DatasetConfigProtocol, cam_id: str) -> str:
    # Preferred path for Human36M and MOYO configs.
    video_fn = getattr(ds_cfg, "_video_path", None)
    if callable(video_fn):
        video_path = str(video_fn(cam_id))
        if os.path.exists(video_path):
            return video_path

    # MPI-INF-3DHP config exposes _seq_dir(); videos are under imageSequence/video_<cam>.avi.
    seq_dir_fn = getattr(ds_cfg, "_seq_dir", None)
    if callable(seq_dir_fn):
        seq_dir = str(seq_dir_fn())
        mpi_video = os.path.join(seq_dir, "imageSequence", f"video_{cam_id}.avi")
        if os.path.exists(mpi_video):
            return mpi_video

    raise FileNotFoundError(f"No video file found for camera {cam_id} in dataset config {ds_cfg}")


def _decode_video_frames_torchcodec(
    decoder: VideoDecoder,
    video_path: str,
    frame_indices: List[int],
    out_hw: Optional[Tuple[int, int]] = None,
    chunk_frames: int = 8,
) -> Tuple[List[torch.Tensor], Optional[Tuple[int, int]]]:
    """Decode specific frames from a video using torchcodec.

    Frames are decoded in small batches (``_BATCH`` at a time) so that each
    ``get_frames_at`` call issues a single seek-then-forward pass (efficient for
    any codec) while keeping peak memory bounded to ~100 MB per batch instead of
    the ~3 GB that a full 256-frame batch of 2048-px MPI-3DHP frames would need.

    Parameters
    ----------
    out_hw:
        When provided, each frame is resized to ``(H, W)`` immediately after
        decoding, before being appended to the output list.  This keeps the
        accumulated list small (640×640 instead of native 2048×2048).

    Returns
    -------
    decoded : list of (H, W, 3) uint8 torch.Tensors
        At ``out_hw`` resolution if given, otherwise at native resolution.
    native_hw : (int, int) or None
        Native source ``(H, W)`` before resize, or ``None`` when ``out_hw``
        was not provided (frames are at native resolution).

    Raises
    ------
    ValueError
        If any requested frame index is out of range for the video.
    """
    total = decoder.metadata.num_frames
    total = total if total is not None else 0
    oob = [i for i in frame_indices if i >= total or i < 0]
    if oob:
        raise ValueError(
            f"Frame indices {oob} are out of range for video with {total} frames: {video_path}"
        )

    decoded: List[torch.Tensor] = []
    native_h: Optional[int] = None
    native_w: Optional[int] = None

    # Decode in small batches: get_frames_at with a sorted batch issues a single
    # seek-then-forward pass, so codec efficiency is preserved.  Peak memory stays
    # at chunk_frames × native_frame_size rather than len(frame_indices) × native_frame_size.
    for batch_start in range(0, len(frame_indices), chunk_frames):
        batch_idx = frame_indices[batch_start : batch_start + chunk_frames]
        frames_nchw = decoder.get_frames_at(indices=batch_idx).data  # (B, C, H, W) uint8
        if native_h is None:
            native_h = frames_nchw.shape[2]
            native_w = frames_nchw.shape[3]
        for i in range(frames_nchw.shape[0]):
            rgb: torch.Tensor = frames_nchw[i].permute(1, 2, 0)  # (H, W, C) view
            if out_hw is not None and (native_h, native_w) != out_hw:
                resized, _ = resize_image_and_intrinsics(rgb, np.eye(3), out_hw)
                rgb = resized if isinstance(resized, torch.Tensor) else torch.as_tensor(resized)
            decoded.append(rgb)
        del frames_nchw

    assert native_w is not None
    assert native_h is not None
    return decoded, (native_h, native_w)

def _apply_alignment_camera_correction(
    seq_out_dir: str,
    cam_defs: Dict[str, Dict[str, Any]],
) -> None:
    """Correct camera T vectors for the world-frame Z shift applied during alignment.

    The alignment pipeline calls ``_apply_sequence_keypoints_to_z_floor`` which
    shifts every keypoint (and every camera-ray origin) upward in world Z so
    that the lowest joint sits at z = 0.  The camera rotation matrices are
    unaffected, but each camera centre moves to ``C_aligned = C_orig + [0,0,dz]``.

    ``alignment_results.json`` stores the camera centres in the *shifted* frame
    as ``ray_origin``.  We use them to recompute T = -R @ C_aligned, overwriting
    the T values returned by ``get_camera_definitions()`` which still reference
    the original (unshifted) world.

    Modifies ``cam_defs`` in place.  If the file is missing or a camera has no
    entry, the original T is kept unchanged.
    """
    align_path = os.path.join(seq_out_dir, "alignment_results.json")
    if not os.path.exists(align_path):
        return

    with open(align_path) as f:
        rows = json.load(f)

    # Extract the first ray_origin per camera_id (constant for a fixed camera).
    # Not all cameras appear in alignment_results.json (only those with detections).
    cam_centers: Dict[str, np.ndarray] = {}
    for row in rows:
        cid = str(row.get("camera_id"))
        if cid not in cam_centers and row.get("ray_origin") is not None:
            cam_centers[cid] = np.array(row["ray_origin"], dtype=np.float64)
            break  # only need one entry

    # Recover the scalar world-Z shift (dz_world) from the first camera that is
    # present in both alignment_results.json and cam_defs.  The shift is uniform
    # across the entire world frame, so one camera is enough.
    dz_world: Optional[float] = None
    for cid, center in cam_centers.items():
        if cid not in cam_defs:
            continue
        R = np.array(cam_defs[cid]["R"], dtype=np.float64)
        T = np.array(cam_defs[cid]["T"], dtype=np.float64).reshape(3)
        # Original camera centre in world: C_orig = -R^T @ T
        c_orig_z = (-R.T @ T)[2]
        dz_world = float(center[2] - c_orig_z)
        break

    if dz_world is None or abs(dz_world) < 1e-6:
        return

    # Apply the shift to every camera in cam_defs: T_new = T_orig + (-R @ [0,0,dz])
    for cid in cam_defs:
        R = np.array(cam_defs[cid]["R"], dtype=np.float64)
        T = np.array(cam_defs[cid]["T"], dtype=np.float64).reshape(3)
        cam_defs[cid]["T"] = T + (-R @ np.array([0.0, 0.0, dz_world]))


def _build_output_window(
    seq: str,
    start: int,
    seq_len: int,
    out_hw: Tuple[int, int],
    max_people: int,
    num_cameras: int,
    keypoints: np.ndarray,
    valid: np.ndarray,
    K_by_cam: np.ndarray,
    R_by_cam: np.ndarray,
    T_by_cam: np.ndarray,
    image_loader: Callable[[int, int, np.ndarray], Tuple[torch.Tensor, np.ndarray]],
    keypoint_frame_valid: Optional[np.ndarray] = None,
    gt_root: Optional[np.ndarray] = None,
    gt_root_valid: Optional[np.ndarray] = None,
    gt_keypoint_valid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    T = seq_len
    N = num_cameras
    J = keypoints.shape[2]

    imgs = torch.zeros((T, N, 3, out_hw[0], out_hw[1]), dtype=torch.uint8)
    world2img = np.zeros((T, N, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((T, N, 4, 4), dtype=np.float32)
    gt_kpts = np.zeros((T, max_people, J, 3), dtype=np.float32)
    gt_valid = np.zeros((T, max_people), dtype=bool)
    gt_kpt_valid = np.zeros((T, max_people, J), dtype=bool)
    roots = np.zeros((T, max_people, 3), dtype=np.float32)
    roots_valid_out = np.zeros((T, max_people), dtype=bool)
    frame_valid_out = (np.ones(T, dtype=bool) if keypoint_frame_valid is None
                       else np.asarray(keypoint_frame_valid, dtype=bool)[:T].copy())

    for ti in range(T):
        for c in range(N):
            img, K2 = image_loader(ti, c, K_by_cam[c])
            imgs[ti, c] = img

            extr = np.eye(4, dtype=np.float64)
            extr[:3, :3] = R_by_cam[c]
            extr[:3, 3] = T_by_cam[c]
            k4 = np.eye(4, dtype=np.float64)
            k4[:3, :3] = K2
            intrinsics[ti, c] = k4.astype(np.float32)
            world2img[ti, c] = (k4 @ extr).astype(np.float32)

        people = np.where(valid[ti])[0][:max_people]
        for pi, pj in enumerate(people):
            gt_kpts[ti, pi] = keypoints[ti, pj]
            gt_valid[ti, pi] = True
            if gt_keypoint_valid is not None:
                gt_kpt_valid[ti, pi] = np.asarray(gt_keypoint_valid[ti, pj], dtype=bool)
            else:
                gt_kpt_valid[ti, pi] = True
            gt_kpt_valid[ti, pi] &= frame_valid_out[ti]
            if gt_root is not None:
                roots[ti, pi] = np.asarray(gt_root[ti, pj], dtype=np.float32)
            else:
                roots[ti, pi] = gt_kpts[ti, pi, list(ROOT_JOINTS)].mean(axis=0)
            roots_valid_out[ti, pi] = (
                bool(gt_root_valid[ti, pj]) if gt_root_valid is not None else True
            )

    return {
        "imgs": imgs,
        "world2img": world2img,
        "intrinsics": intrinsics,
        "gt_keypoints": gt_kpts,
        "gt_valid": gt_valid,
        "gt_keypoint_frame_valid": frame_valid_out,
        "gt_keypoint_valid": gt_kpt_valid,
        "gt_root": roots,
        "gt_root_valid": roots_valid_out,
        "seq": seq,
        "start": start,
    }
