"""Import reviewed gym videos and sparse human-label pseudo-3D into a pinhole cache.

Run from the repository root with ``python -m unified_perception.tools.import_gym_pseudo3d``.
Source files are read-only; the output records the calibration and coordinate transform.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from unified_perception.constants import JOINT_NAMES, SKELETON_EDGES
from unified_perception.data.transforms import resize_image_and_intrinsics

CAMERA_IDS = (0, 1, 2, 6)
SOURCE = Path(r"C:\cvat_labeling_data\20250505071549")
DEFAULT_ROOT = Path("data_cache") / "gym_pseudo3d_20250505"


def recommended_position_range(low, high, margin_mm=300.0, step_mm=500.0):
    """Symmetric xy / fixed-floor z volume covering all targets plus a margin (never below the v2 range)."""
    def up(value):
        return float(np.ceil((value + margin_mm) / step_mm) * step_mm)
    x = max(6000.0, up(max(-low[0], high[0])))
    y = max(12000.0, up(max(-low[1], high[1])))
    z = max(3000.0, up(high[2]))
    if low[2] < -200.0 + 1e-6:
        raise ValueError(f"Targets below the -200 mm floor margin: {low[2]:.0f} mm")
    return [-x, -y, -200.0, x, y, z]


def transform_targets(source):
    """Keep z-up/mm; subtract the train-only xy bounding-box centre (no scale)."""
    if list(source["joint_names"]) != JOINT_NAMES:
        raise ValueError("Source joint_names must exactly match constants.JOINT_NAMES")
    frames = np.asarray(source["frames"], dtype=np.int64)
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Expected contiguous session/video frame indices starting at zero")
    exists = source["exists"].astype(bool)
    labeled = source["labeled"].astype(bool)
    root_valid = source["root_valid"].astype(bool) & exists
    joint_valid = source["keypoint_valid"].astype(bool) & exists[..., None] & labeled[:, None, None]
    roots = np.asarray(source["root"], dtype=np.float64)
    joints = np.asarray(source["keypoints"], dtype=np.float64)
    if not np.isfinite(roots[root_valid]).all() or not np.isfinite(joints[joint_valid]).all():
        raise ValueError("Nonfinite coordinates marked valid")
    train_roots = roots[root_valid & (frames[:, None] < 240)]
    if not len(train_roots):
        raise ValueError("No valid training roots")
    offset = (train_roots.min(axis=0) + train_roots.max(axis=0)) / 2
    offset[2] = 0
    roots = np.where(root_valid[..., None], roots - offset, 0).astype(np.float32)
    joints = np.where(joint_valid[..., None], joints - offset, 0).astype(np.float32)
    arrays = dict(frames=frames, person_ids=source["person_ids"], valid=exists,
                  root=roots, root_valid=root_valid, keypoints=joints,
                  keypoint_valid=joint_valid, labeled=labeled,
                  root_source=source["root_source"])
    return arrays, offset


def observation_masks(observations, identity, person_ids, labeled, exists, in_view=None,
                      camera_ids=CAMERA_IDS):
    """Per-frame reviewed-view observation mask (F,P,len(CAMERA_IDS)).

    Labeled frames use the human 2-D tracks mapped through the BA identity map
    ("camera:track" -> person id). Unlabeled frames copy the nearest labeled
    frame (ties go to the earlier label, matching pseudo_gt ``exists``).
    ``in_view(item)`` may veto observations whose labeled joints all fall
    outside the cached (undistorted, cropped) image; the uncropped union must
    still equal ``exists``.
    """
    labeled = np.asarray(labeled, dtype=bool)
    person_index = {int(p): i for i, p in enumerate(person_ids)}
    observed = np.zeros((len(labeled), len(person_index), len(camera_ids)), dtype=bool)
    kept = observed.copy()
    for item in observations:
        if item["camera"] not in camera_ids:
            raise ValueError(f"Unexpected observation camera {item['camera']}")
        frame = int(item["session_frame"])
        if not labeled[frame]:
            raise ValueError(f"Observation on unlabeled frame {frame}")
        key = f"{item['camera']}:{item['track']}"
        if key not in identity:
            raise ValueError(f"Track {key} missing from the identity map")
        cell = frame, person_index[int(identity[key])], camera_ids.index(item["camera"])
        observed[cell] = True
        kept[cell] |= in_view is None or bool(in_view(item))
    label_frames = np.flatnonzero(labeled)
    for frame in np.flatnonzero(~labeled):
        nearest = label_frames[np.argmin(np.abs(label_frames - frame))]
        observed[frame], kept[frame] = observed[nearest], kept[nearest]
    if not np.array_equal(observed.any(-1), np.asarray(exists, dtype=bool)):
        raise ValueError("Reviewed-view observations disagree with pseudo_gt exists")
    return kept


def crop_visibility(definitions, size=(1280, 960), camera_ids=CAMERA_IDS):
    """``in_view`` callback: any labeled joint lands inside the alpha=0 pinhole crop."""
    cameras = {}
    for camera_id in camera_ids:
        definition = definitions[str(camera_id)]
        K, D = np.array(definition["K"], dtype=np.float64), np.array(definition["D"], dtype=np.float64)
        cameras[camera_id] = K, D, cv2.getOptimalNewCameraMatrix(K, D, size, 0, size)[0]

    def in_view(item):
        points = [joint["xy"] for joint in item["joints"].values()]
        if not points:
            return False
        K, D, new_K = cameras[item["camera"]]
        uv = cv2.undistortPoints(np.asarray(points, np.float64).reshape(-1, 1, 2), K, D, P=new_K).reshape(-1, 2)
        return bool(((uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])).any())
    return in_view


def _project(points, K, R, T):
    cam = points @ R.T + T
    pix = cam @ K.T
    uv = pix[..., :2] / np.maximum(pix[..., 2:3], 1e-8)
    return uv, cam[..., 2] > 0


def render_debug(root: Path, index: dict):
    """Draw valid joints and rough roots in two labeled four-view mosaics."""
    outputs = []
    for split in ("train", "val"):
        entry = index["splits"][split][0]
        folder = root / split / entry["seq"]
        with np.load(folder / "poses.npz") as poses, np.load(folder / "calib.npz") as calib:
            ti = int(np.flatnonzero(poses["labeled"])[min(1, int(poses["labeled"].sum()) - 1)])
            tiles = []
            for ci, camera_id in enumerate(index["camera_ids"]):
                image_path = folder / "images" / f"f{ti:03d}_c{ci:03d}.png"
                image = cv2.imread(str(image_path))
                if image is None:
                    raise OSError(f"Missing cached debug image: {image_path}")
                for pi in np.flatnonzero(poses["valid"][ti]):
                    color = tuple(int(v) for v in np.random.default_rng(int(pi)).integers(50, 255, 3))
                    uv, front = _project(poses["keypoints"][ti, pi], calib["K"][ci], calib["R"][ci], calib["T"][ci])
                    mask = poses["keypoint_valid"][ti, pi] & front
                    h, w = image.shape[:2]
                    inside = mask & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                    for a, b in SKELETON_EDGES:
                        if inside[a] and inside[b]:
                            cv2.line(image, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), color, 1, cv2.LINE_AA)
                    for pt in uv[inside]:
                        cv2.circle(image, tuple(pt.astype(int)), 2, color, -1)
                    if poses["root_valid"][ti, pi]:
                        pt, ahead = _project(poses["root"][ti, pi], calib["K"][ci], calib["R"][ci], calib["T"][ci])
                        if ahead and 0 <= pt[0] < w and 0 <= pt[1] < h:
                            xy = tuple(pt.astype(int))
                            cv2.drawMarker(image, xy, color, cv2.MARKER_CROSS, 10, 1)
                            cv2.putText(image, str(poses["person_ids"][pi]), (xy[0]+4, xy[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
                cv2.putText(image, f"cam {camera_id}, frame {poses['frames'][ti]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                tiles.append(image)
            columns = min(3, len(tiles))
            while len(tiles) % columns:
                tiles.append(np.zeros_like(tiles[0]))
            mosaic = np.concatenate([np.concatenate(tiles[i:i+columns], axis=1)
                                     for i in range(0, len(tiles), columns)], axis=0)
            path = root / f"debug_{split}_frame{poses['frames'][ti]:03d}.jpg"
            if not cv2.imwrite(str(path), mosaic):
                raise OSError(f"Failed to write {path}")
            outputs.append(str(path))
    return outputs


def convert(pseudo_gt: Path, cameras: Path, videos: Path, root: Path,
            out_hw=(320, 320), overwrite=False, observations: Path | None = None,
            identity: Path | None = None, camera_ids=CAMERA_IDS):
    root = Path(root)
    if (root / "index.json").exists() and not overwrite:
        raise FileExistsError(f"{root} already contains a cache; use --overwrite deliberately")
    if observations is None or identity is None:
        raise ValueError("observations and identity map are required for per-view existence masks")
    camera_ids = tuple(camera_ids)
    if not camera_ids or list(camera_ids) != sorted(set(camera_ids)) or any(c not in range(7) for c in camera_ids):
        raise ValueError("camera_ids must be distinct sorted camera IDs in 0..6")
    with np.load(pseudo_gt, allow_pickle=False) as source:
        arrays, offset = transform_targets(source)
    camera_document = json.loads(Path(cameras).read_text(encoding="utf-8-sig"))
    definitions = camera_document["cameras"]
    exists = arrays["valid"]
    arrays["observed"] = observation_masks(
        json.loads(Path(observations).read_text(encoding="utf-8-sig")),
        json.loads(Path(identity).read_text(encoding="utf-8-sig"))["identity"],
        arrays["person_ids"], arrays["labeled"], exists,
        crop_visibility(definitions, camera_ids=camera_ids), camera_ids=camera_ids)
    # People whose reviewed tracks all fall outside the undistortion crop are not in the images.
    visible = arrays["observed"].any(-1)
    arrays["valid"] = visible
    arrays["root_valid"] &= visible
    arrays["keypoint_valid"] &= visible[..., None]
    arrays["root"][~arrays["root_valid"]] = 0
    arrays["keypoints"][~arrays["keypoint_valid"]] = 0
    if len(arrays["frames"]) != 301:
        raise ValueError("This session must contain frames 0..300 (train 0..239, val 240..300)")
    root.mkdir(parents=True, exist_ok=True)
    splits = {"train": slice(0, 240), "val": slice(240, 301)}
    index = {"format_version": 2, "dataset": "gym_pseudo3d_20250505", "fps": 10,
             "image_hw": list(out_hw), "camera_ids": list(camera_ids),
             "joint_names": JOINT_NAMES, "person_ids": arrays["person_ids"].tolist(),
             "world": "z up, floor z=0, millimetres; X_cache = X_source - world_offset_mm",
             "world_offset_mm": offset.tolist(),
             "origin_fit": "training roots only; same offset shared by both splits",
             "source_pseudo_gt": str(Path(pseudo_gt).resolve()),
             "source_cameras": str(Path(cameras).resolve()),
             "source_observations": str(Path(observations).resolve()),
             "source_identity": str(Path(identity).resolve()),
             "observed": ("poses.npz observed (F,P,cameras): reviewed 2-D track at the nearest labeled frame "
                          "with at least one labeled joint inside the undistorted crop; valid = observed.any(-1)"),
             "dropped_crop_person_frames": int((exists & ~visible).sum()),
             "undistortion": "OpenCV full D, getOptimalNewCameraMatrix alpha=0; letterbox resize",
             "splits": {}}
    sequence = "gym_20250505"
    for split, selection in splits.items():
        folder = root / split / sequence
        (folder / "images").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "poses.npz", **{k: (v if k == "person_ids" else v[selection]) for k, v in arrays.items()})
        index["splits"][split] = [{"seq": sequence, "num_frames": len(arrays["frames"][selection]),
                                   "first_session_frame": selection.start,
                                   "last_session_frame": selection.stop - 1}]
    Ks, Rs, Ts = [], [], []
    for ci, camera_id in enumerate(camera_ids):
        definition = definitions[str(camera_id)]
        K, D = np.array(definition["K"], dtype=np.float64), np.array(definition["D"], dtype=np.float64)
        R, T = np.array(definition["R"], dtype=np.float64), np.array(definition["T_mm"], dtype=np.float64)
        cap = cv2.VideoCapture(str(Path(videos) / f"{camera_id}.mp4"))
        try:
            if not cap.isOpened():
                raise OSError(f"Cannot open camera {camera_id}")
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (w, h) != (1280, 960) or abs(cap.get(cv2.CAP_PROP_FPS) - 10) > .05:
                raise ValueError(f"Expected 1280x960/10fps video for camera {camera_id}")
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
            # Map directly onto the alpha=0 pinhole canvas, then use the pool's letterbox transform.
            maps = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
            for frame in arrays["frames"]:
                ok, image = cap.read()
                if not ok:
                    raise OSError(f"Camera {camera_id} ended before frame {frame}")
                image = cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR)
                image, resized_K = resize_image_and_intrinsics(image, new_K, out_hw)
                split, local = ("train", int(frame)) if frame < 240 else ("val", int(frame) - 240)
                path = root / split / sequence / "images" / f"f{local:03d}_c{ci:03d}.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"Failed to write {path}")
            Ks.append(resized_K)
            Rs.append(R)
            Ts.append(T + R @ offset)
            print(f"Imported camera {camera_id}: {len(arrays['frames'])} frames", flush=True)
        finally:
            cap.release()
    for split in splits:
        np.savez_compressed(root / split / sequence / "calib.npz", K=np.array(Ks), R=np.array(Rs),
                            T=np.array(Ts), camera_ids=np.array(camera_ids), world_offset_mm=offset)
    valid_points = np.concatenate([arrays["root"][arrays["root_valid"]], arrays["keypoints"][arrays["keypoint_valid"]]])
    index["valid_coordinate_bounds_mm"] = [valid_points.min(0).tolist(), valid_points.max(0).tolist()]
    index["recommended_position_range_mm"] = recommended_position_range(*index["valid_coordinate_bounds_mm"])
    index["debug_images"] = render_debug(root, index)
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pseudo3d = SOURCE / "pseudo3d_20261031"  # calibration v5_mono, identity_v5 (cameras 0,1,2,6 labels)
    parser.add_argument("--pseudo-gt", type=Path, default=pseudo3d / "pseudo_gt" / "pseudo_gt.npz")
    parser.add_argument("--cameras", type=Path, default=pseudo3d / "pseudo_gt" / "cameras.json")
    parser.add_argument("--videos", type=Path, default=SOURCE / "pilot" / "normalized")
    parser.add_argument("--observations", type=Path, default=pseudo3d / "pseudo_gt" / "observations.json")
    parser.add_argument("--identity", type=Path, default=pseudo3d / "analysis" / "identity_v5.json")
    parser.add_argument("--camera-ids", default="0,1,2,6",
                        help="sorted comma-separated reviewed camera IDs (default: 0,1,2,6)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        parser.error("height/width must be positive")
    index = convert(args.pseudo_gt, args.cameras, args.videos, args.root,
                    (args.height, args.width), args.overwrite, args.observations, args.identity,
                    tuple(int(c) for c in args.camera_ids.split(",")))
    print(json.dumps({"root": str(args.root.resolve()), "splits": index["splits"],
                      "world_offset_mm": index["world_offset_mm"],
                      "bounds_mm": index["valid_coordinate_bounds_mm"]}, indent=2))


if __name__ == "__main__":
    main()
