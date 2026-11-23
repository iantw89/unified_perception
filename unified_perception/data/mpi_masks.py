"""Official MPI-INF-3DHP segmentation, aligned to the RGB video frame indices."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .background import person_region
from .composition import Clip, file_hash
from .transforms import resize_image_and_intrinsics

METHOD = "mpi-official-v1"
LOGGER = logging.getLogger(__name__)
# mpii_get_sequence_info.m explicitly marks these background masks unreliable.
UNRELIABLE_BACKGROUND = {(1, 2), (4, 2)}


def recording_ids(source_id: str) -> tuple[int, int]:
    subject, sequence = source_id.split("__seg", 1)[0].split("_")
    if not subject.startswith("S") or not sequence.startswith("Seq"):
        raise ValueError(f"Invalid MPI source ID: {source_id}")
    ids = int(subject[1:]), int(sequence[3:])
    if ids[0] not in range(1, 9) or ids[1] not in (1, 2):
        raise ValueError(f"Invalid MPI recording: {source_id}")
    return ids


def mask_paths(root: Path, source_id: str, camera: str) -> tuple[Path, Path]:
    subject, sequence = recording_ids(source_id)
    if not camera.isdecimal() or int(camera) not in range(14):
        raise ValueError(f"Invalid MPI camera ID: {camera}")
    base = root / f"S{subject}" / f"Seq{sequence}"
    return (base / "FGmasks" / f"video_{camera}.avi",
            base / "ChairMasks" / f"video_{camera}.avi")


def capture_group(clip: Clip, source_id: str) -> str:
    """Group segments only when every camera, frame geometry, and sampling rate agree."""
    subject, sequence = recording_ids(source_id)
    geometry = hashlib.sha256()
    for array in (clip.world2img, clip.intrinsics):
        geometry.update(array.dtype.str.encode())
        geometry.update(str(array.shape).encode())
        geometry.update(array.tobytes())
    geometry.update(str(clip.metadata["fps"]).encode())
    cameras = ",".join(map(str, clip.metadata["camera_ids"]))
    return f"mpi:S{subject}_Seq{sequence}:c{cameras}:{geometry.hexdigest()}"


def person_mask(fg_rgb: np.ndarray, chair_rgb: np.ndarray, joints: np.ndarray,
                projection: np.ndarray) -> np.ndarray:
    """Red is non-greenscreen in FGmasks and non-chair in ChairMasks."""
    if fg_rgb.shape != chair_rgb.shape or fg_rgb.dtype != np.uint8 or chair_rgb.dtype != np.uint8:
        raise ValueError("MPI mask videos must have matching uint8 RGB geometry")
    region, pixels = person_region(joints, projection, fg_rgb.shape[:2])
    mask = (fg_rgb[..., 0] > 127) & (chair_rgb[..., 0] > 127) & region
    total, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = [i for i in range(1, total) if stats[i, cv2.CC_STAT_AREA] >= 8]
    mask = np.isin(labels, keep)
    near = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    inside = ((pixels >= 0).all(1) & (pixels[:, 0] < mask.shape[1]) & (pixels[:, 1] < mask.shape[0]))
    covered = np.zeros(len(pixels), dtype=bool)
    covered[inside] = near[pixels[inside, 1], pixels[inside, 0]]
    if mask.sum() < 32 or covered.mean() < .75:
        raise ValueError("Official MPI mask is incomplete at projected joints")
    return mask


def apply_official_masks(provider, clip: Clip) -> bool:
    """Replace footprint masks; return False only for an explicit background fallback."""
    mode = provider.cfg.data.composition_mpi_masks
    if mode not in ("background", "prefer", "required"):
        raise ValueError(f"Invalid composition_mpi_masks mode: {mode}")
    if clip.metadata["dataset"] != "mpi" or mode == "background":
        return False
    seq = provider._seqs[clip.metadata["sequence"]]
    ids = recording_ids(seq.source_id)
    root = Path(provider.cfg.data.human_prior_mpi_dataset_path)
    paths = {camera: mask_paths(root, seq.source_id, camera) for camera in clip.metadata["camera_ids"]}
    missing = [str(path) for pair in paths.values() for path in pair if not path.is_file()]
    reason = ("MPI marks this recording's background masks unreliable" if ids in UNRELIABLE_BACKGROUND
              else f"Missing official MPI mask videos: {missing}" if missing else None)
    if reason:
        if mode == "required":
            raise ValueError(reason)
        LOGGER.warning("%s: %s; using label-gated background extraction", seq.seq_id, reason)
        clip.metadata["official_mask_fallback"] = reason
        return False
    hashes = {}
    masks = np.zeros_like(clip.masks)
    definitions = provider._cam_defs_for_seq(seq)
    for n, camera in enumerate(clip.metadata["camera_ids"]):
        frames = []
        for path in paths[camera]:
            # Reuse readers and hashes for overlapping windows within a sequence.
            cache = getattr(provider, "_official_mask_cache", None)
            if cache is None:
                cache = provider._official_mask_cache = {}
            if path not in cache:
                reader = cv2.VideoCapture(str(path))
                if not reader.isOpened():
                    reader.release()
                    raise ValueError(f"Cannot open MPI mask video: {path}")
                native_path = root / f"S{ids[0]}" / f"Seq{ids[1]}" / "imageSequence" / path.name
                rgb_reader = cv2.VideoCapture(str(native_path))
                native_hw = (int(rgb_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                             int(rgb_reader.get(cv2.CAP_PROP_FRAME_WIDTH)))
                fps = rgb_reader.get(cv2.CAP_PROP_FPS)
                rgb_reader.release()
                if (native_hw != (int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                  int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)))
                        or not np.isclose(reader.get(cv2.CAP_PROP_FPS), fps)):
                    reader.release()
                    raise ValueError(f"MPI mask/RGB native geometry or FPS differs: {path}")
                cache[path] = reader, file_hash(path)
            reader, hashes[str(path)] = cache[path]
            selected = []
            for t, index in enumerate(clip.metadata["camera_frame_ids"][camera]):
                if index < 0 or not reader.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError(f"Cannot seek MPI mask {path} frame {index}")
                ok, bgr = reader.read()
                if not ok:
                    raise ValueError(f"Missing MPI mask frame: {path} frame {index}")
                rgb, _ = resize_image_and_intrinsics(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.eye(3), provider.out_hw)
                distortion = np.asarray(definitions[camera].get("dist", np.zeros(5)), dtype=np.float64)
                selected.append(cv2.undistort(rgb, clip.intrinsics[t, n, :3, :3], distortion))
            frames.append(selected)
        for t in range(len(clip.imgs)):
            try:
                masks[t, n] = person_mask(frames[0][t], frames[1][t], clip.joints[t], clip.world2img[t, n])
            except ValueError as exc:
                frame = clip.metadata["camera_frame_ids"][camera][t]
                raise ValueError(f"Official MPI camera {camera} video frame {frame}: {exc}") from exc
    digest = hashlib.sha256(json.dumps(
        dict(assets=hashes, opencv=cv2.__version__, method=METHOD), sort_keys=True).encode()).hexdigest()
    clip.masks = masks
    clip.metadata.update(mask_method=METHOD, mask_assets=hashes, extraction_hash=f"{METHOD}:{digest}")
    return True


def close_mask_readers(provider) -> None:
    for reader, _ in getattr(provider, "_official_mask_cache", {}).values():
        reader.release()
    provider._official_mask_cache = {}
