"""Build a training cache from a data_management 3-D ground-truth label set.

    python -m unified_perception.tools.import_golden \\
        --store ../data_management/data/labels \\
        --tracks-set golden_20250505071549_20250505080000.three_d_tracks.cvat_identity \\
        --calibration C:/cvat_labeling_data/20250505071549/pseudo3d_20261031/calibration/v5_tracks_ransac.json \\
        --videos C:/cvat_labeling_data/20250505071549/external_autofit_v5mono_20261102/input/pilot/normalized \\
        --root data_cache/golden_20250505_640 --height 640 --width 640

Sources are read-only.  Videos must be the normalized clips whose frame k is
aligned index k (``{camera number}.mp4``).  The cache uses the gym layout
(``format_version`` 3, see data/golden.py) so ``data.gym_pseudo3d_root`` can point
at it; the val split is the tail of the session after a purge gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from unified_perception.constants import JOINT_NAMES
from unified_perception.data.golden import (
    GoldenConfig, build_targets, coverage_summary, load_cameras, load_golden_sets, _sha256,
)
from unified_perception.data.transforms import resize_image_and_intrinsics
from unified_perception.tools.import_gym_pseudo3d import recommended_position_range


def pinhole_to_cache(uv: np.ndarray, K_from: np.ndarray, K_to: np.ndarray) -> np.ndarray:
    """Undistorted video pixels -> cache pixels (uniform letterbox scale + shift)."""
    out = uv.copy()
    out[..., 0] = (uv[..., 0] - K_from[0, 2]) * K_to[0, 0] / K_from[0, 0] + K_to[0, 2]
    out[..., 1] = (uv[..., 1] - K_from[1, 2]) * K_to[1, 1] / K_from[1, 1] + K_to[1, 2]
    return out


def convert(store: Path, tracks_set: str, calibration_path: Path, videos: Path, root: Path,
            out_hw=(640, 640), val_fraction=0.2, purge_frames=50, image_ext="jpg", jpeg_quality=95,
            cfg: GoldenConfig | None = None, overwrite=False, frame_limit: int | None = None,
            targets_only: bool = False):
    cfg = cfg or GoldenConfig()
    root = Path(root)
    if targets_only:
        previous = json.loads((root / "index.json").read_text(encoding="utf-8"))
        if previous["image_hw"] != list(out_hw) or previous["sources"]["videos"] != str(Path(videos).resolve()):
            raise ValueError("--targets-only needs the existing cache's image size and videos")
    elif (root / "index.json").exists() and not overwrite:
        raise FileExistsError(f"{root} already contains a cache; use --overwrite deliberately")
    tracks, tracklets, alignment = load_golden_sets(store, tracks_set)
    tracks_doc = tracks.files["tracks.json"]
    tracklet_docs = {doc["camera"]: doc for doc in tracklets.files.values()}
    align_doc = alignment.files["alignment.json"]
    camera_names = sorted(align_doc["cameras"], key=lambda n: int(n.rsplit("-", 1)[-1]))
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    cameras = load_cameras(calibration, camera_names, cfg.native_to_video)
    arrays = build_targets(tracks_doc, tracklet_docs, align_doc, cameras, cfg)
    rejected = arrays.pop("rejected")
    F = len(arrays["frames"]) if frame_limit is None else min(frame_limit, len(arrays["frames"]))
    arrays = {k: (v[:F] if k not in ("person_ids", "person_source_ids") else v) for k, v in arrays.items()}

    # People never inside the undistorted (alpha = 0) image of a camera are not observed there.
    h_video, w_video = cameras[0].video_hw
    pin = arrays.pop("kp2d_video")
    inside = ((pin[..., 0] >= 0) & (pin[..., 0] < w_video) & (pin[..., 1] >= 0) & (pin[..., 1] < h_video)
              & np.isfinite(pin).all(-1))
    reliable = inside & (pin[..., 2] >= cfg.min_confidence)
    grid = arrays["labeled"]
    seen_at_grid = reliable.any(-1)
    labels = np.flatnonzero(grid)
    nearest = np.array([labels[np.argmin(np.abs(labels - f) + (labels > f) * 0.1)] for f in range(F)])
    arrays["observed"] &= seen_at_grid[nearest]
    arrays["valid"] = arrays["observed"].any(-1)

    split_frame = int(round(F * (1 - val_fraction)))
    splits = {"train": (0, max(split_frame - purge_frames, 0)), "val": (split_frame, F)}
    train = slice(*splits["train"])
    train_roots = arrays["root"][train][arrays["root_valid"][train]]
    offset = (train_roots.min(0) + train_roots.max(0)) / 2
    offset[2] = 0.0
    for key, valid_key in (("root", "root_valid"), ("keypoints", "keypoint_valid"),
                           ("match_root", "match_root_valid")):
        arrays[key] = np.where(arrays[valid_key][..., None], arrays[key] - offset, 0).astype(np.float32)

    sequence = tracks.meta["dataset"]
    index = {
        "format_version": 3, "dataset": sequence, "fps": float(align_doc.get("nominal_fps", 10.0)),
        "image_hw": list(out_hw), "camera_ids": [int(n.rsplit("-", 1)[-1]) for n in camera_names],
        "camera_names": camera_names, "joint_names": JOINT_NAMES,
        "person_ids": arrays["person_ids"].tolist(), "person_source_ids": arrays["person_source_ids"].tolist(),
        "image_pattern": "f{frame:05d}_c{camera:03d}." + image_ext,
        "world": f"{tracks_doc.get('calibration', {}).get('world', 'world')} -> millimetres; "
                 "X_cache = X_world_mm - world_offset_mm",
        "world_offset_mm": offset.tolist(),
        "sources": {"tracks_set": tracks.name, "tracklets_set": tracklets.name, "alignment_set": alignment.name,
                    "set_files_sha256": {s.name: s.meta["data_files"] for s in (tracks, tracklets, alignment)},
                    "calibration": str(Path(calibration_path).resolve()),
                    "calibration_sha256": _sha256(Path(calibration_path)), "videos": str(Path(videos).resolve())},
        "config": vars(cfg),
        "fields": {
            "observed": "(F,P,C) member 2-D detection with a reliable joint inside the undistorted image, at the "
                        "nearest grid frame; valid = observed.any(-1)",
            "keypoint_sigma": "(F,P,J) mm uncertainty of triangulated/interpolated targets (0 = no target)",
            "kp2d": "(F,P,C,J,3) member detections in cache pixels (u, v, confidence), grid frames only",
            "triangulated": "(F,P) 3-D target measured on this grid frame",
            "interpolated": "(F,P) 3-D target linearly interpolated between grid frames (sigma adds motion); "
                            "only with --interpolated-use loss",
            "match_root": "(F,P,3) mm root linearly interpolated between grid frames: a query-matching cue "
                          "only, never a loss target (match_root_valid)",
        },
        "coverage": coverage_summary(arrays, camera_names), "rejected_outliers": rejected, "splits": {},
        "undistortion": "OpenCV full D in video space (canvas K shifted by the letterbox padding), "
                        "getOptimalNewCameraMatrix alpha=0; letterbox resize",
    }
    for split, (a, b) in splits.items():
        index["splits"][split] = [{"seq": sequence, "num_frames": b - a, "first_aligned_frame": a,
                                   "last_aligned_frame": b - 1}]
        (root / split / sequence / "images").mkdir(parents=True, exist_ok=True)

    Ks, Rs, kp2d = [], [], np.zeros(pin.shape, np.float32)
    for ci, cam in enumerate(cameras):
        if targets_only:  # keep the cached images; only the intrinsics are needed
            h, w = cam.video_hw
            resized_K = resize_image_and_intrinsics(np.zeros((h, w, 3), np.uint8), cam.K_pinhole, out_hw)[1]
            uv = pinhole_to_cache(pin[:, :, ci, :, :2], cam.K_pinhole, resized_K)
            kp2d[:, :, ci, :, :2] = np.nan_to_num(uv)
            kp2d[:, :, ci, :, 2] = np.nan_to_num(pin[:, :, ci, :, 2])
            Ks.append(resized_K)
            Rs.append(cam.R)
            continue
        video = Path(videos) / f"{cam.name.rsplit('-', 1)[-1]}.mp4"
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise OSError(f"Cannot open {video}")
        try:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (h, w) != cam.video_hw or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < F:
                raise ValueError(f"{video}: expected {cam.video_hw[1]}x{cam.video_hw[0]} with >= {F} frames")
            maps = cv2.initUndistortRectifyMap(cam.K_video, cam.D, None, cam.K_pinhole, (w, h), cv2.CV_32FC1)
            resized_K = None
            for frame in range(F):
                ok, image = cap.read()
                if not ok:
                    raise OSError(f"{video} ended before frame {frame}")
                image = cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR)
                image, resized_K = resize_image_and_intrinsics(image, cam.K_pinhole, out_hw)
                split = "train" if frame < splits["train"][1] else ("val" if frame >= splits["val"][0] else None)
                if split is None:
                    continue
                local = frame - splits[split][0]
                path = root / split / sequence / "images" / index["image_pattern"].format(frame=local, camera=ci)
                params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if image_ext == "jpg" else []
                if not cv2.imwrite(str(path), image, params):
                    raise OSError(f"Failed to write {path}")
        finally:
            cap.release()
        uv = pinhole_to_cache(pin[:, :, ci, :, :2], cam.K_pinhole, resized_K)
        kp2d[:, :, ci, :, :2] = np.nan_to_num(uv)
        kp2d[:, :, ci, :, 2] = np.nan_to_num(pin[:, :, ci, :, 2])
        Ks.append(resized_K)
        Rs.append(cam.R)
        print(f"Imported {cam.name}: {F} frames", flush=True)
    unit = {"in": 25.4, "mm": 1.0, "m": 1000.0, "cm": 10.0}[tracks_doc.get("units", "mm")]
    Ts = [cam.tvec * unit + cam.R @ offset for cam in cameras]
    arrays["kp2d"] = kp2d
    arrays["kp2d_valid"] = reliable & arrays["labeled"][:, None, None, None]
    arrays["root_source"] = np.zeros(arrays["root_valid"].shape, np.int8)
    for split, (a, b) in splits.items():
        folder = root / split / sequence
        np.savez_compressed(folder / "poses.npz", **{
            k: (v if k in ("person_ids", "person_source_ids") else v[a:b]) for k, v in arrays.items()})
        np.savez_compressed(folder / "calib.npz", K=np.array(Ks), R=np.array(Rs), T=np.array(Ts),
                            camera_ids=np.array(index["camera_ids"]), world_offset_mm=offset)
    points = np.concatenate([arrays["root"][arrays["root_valid"]], arrays["keypoints"][arrays["keypoint_valid"]]])
    index["valid_coordinate_bounds_mm"] = [points.min(0).tolist(), points.max(0).tolist()]
    index["recommended_position_range_mm"] = recommended_position_range(*index["valid_coordinate_bounds_mm"])
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, required=True, help="data_management data/labels directory")
    ap.add_argument("--tracks-set", required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--purge-frames", type=int, default=50)
    ap.add_argument("--calibration-sigma-mm", type=float, default=GoldenConfig.calibration_sigma_mm)
    ap.add_argument("--no-interpolate", action="store_true")
    ap.add_argument("--interpolated-use", choices=("match", "loss"), default=GoldenConfig.interpolated_use,
                    help="match: interpolated roots only assign queries between labels (default); "
                         "loss: legacy interpolated 3-D loss targets")
    ap.add_argument("--frame-limit", type=int, default=None, help="first N aligned frames only (smoke builds)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--targets-only", action="store_true",
                    help="rewrite targets/calibration/index of an existing cache, keep its images")
    args = ap.parse_args()
    cfg = GoldenConfig(calibration_sigma_mm=args.calibration_sigma_mm, interpolate=not args.no_interpolate,
                       interpolated_use=args.interpolated_use)
    index = convert(args.store, args.tracks_set, args.calibration, args.videos, args.root,
                    (args.height, args.width), args.val_fraction, args.purge_frames, cfg=cfg,
                    overwrite=args.overwrite, frame_limit=args.frame_limit, targets_only=args.targets_only)
    print(json.dumps({k: index[k] for k in ("coverage", "splits", "recommended_position_range_mm")}, indent=2))


if __name__ == "__main__":
    main()
