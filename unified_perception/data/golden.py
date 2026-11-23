"""Adapt data_management 3-D ground truth (e.g. the golden gym dataset) into training targets.

Inputs (read-only, SHA-256 verified against each label set's metadata):

* a ``three_d_tracks`` set: persistent identities, their member 2-D tracklets and
  triangulated world keypoints (COCO-17, ``null`` = unmeasured) on a sampled grid;
* its ``two_d_tracklets`` dependency: per-camera machine 2-D poses in native
  pixels, linked to identities by (human-reviewed) tracklet ids;
* its ``frame_alignment`` dependency: exact native frame per camera for every
  aligned index (aligned index == normalized-video frame, verified);
* the calibration the triangulation used (OpenCV K/D/rvec/tvec in a letterboxed
  canvas: ``canvas = native * native_to_video + padding``) and the normalized videos.

Output: a pinhole image cache in the gym cache layout (``format_version`` 3) read
by :class:`~unified_perception.data.gym_pseudo3d.GymPseudo3DProvider`.  Besides
the v2 arrays it stores what is needed to train on approximate, incomplete GT:

* ``keypoint_sigma`` / ``root_sigma`` (mm): per-joint uncertainty of each
  triangulated target, from its reprojection residuals, depth/focal length of
  every supporting view and the ray geometry, plus a calibration floor; linearly
  interpolated targets between grid frames add a motion term;
* ``observed`` (F, P, C): the identity has a member 2-D detection in camera C;
* ``kp2d`` / ``kp2d_valid`` (F, P, C, J): those detections in cache-image pixels
  (undistorted, resized), for reprojection supervision of people that one camera
  sees but that could not be triangulated (dead zones under cameras 0-3, 5);
* ``triangulated`` / ``interpolated`` (F, P) flags.

So a person can be (a) triangulated -> noisy 3-D target; (b) observed but not
triangulated -> present (never trained to vanish) with 2-D reprojection
targets only; (c) unobserved -> existence handled by the provider's gap rule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..constants import JOINT_NAMES, SKELETON_EDGES
from .transforms import resize_image_and_intrinsics

COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
COCO_TO_JOINTS = [COCO17.index(name) for name in JOINT_NAMES]
UNIT_MM = {"mm": 1.0, "m": 1000.0, "cm": 10.0, "in": 25.4}


# --------------------------------------------------------------------------- #
# Label store access                                                           #
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class LabelSet:
    name: str
    meta: dict
    files: dict[str, object]


def load_label_set(store: Path, name: str) -> LabelSet:
    """Read a published set and verify every data file against its recorded SHA-256."""
    store = Path(store)
    meta = json.loads((store / "sets" / f"{name}.json").read_text(encoding="utf-8"))
    root = store / meta["data_dir"]  # "data/<set>", relative to the store root
    files = {}
    for relative, expected in meta["data_files"].items():
        path = root / relative
        if _sha256(path) != expected:
            raise RuntimeError(f"{name}: {relative} does not match its published SHA-256")
        files[relative] = json.loads(path.read_text(encoding="utf-8"))
    return LabelSet(name, meta, files)


def load_golden_sets(store: Path, tracks_set: str) -> tuple[LabelSet, LabelSet, LabelSet]:
    tracks = load_label_set(store, tracks_set)
    if tracks.meta["label_type"] != "three_d_tracks":
        raise ValueError(f"{tracks_set} is not a three_d_tracks set")
    deps = {load_label_set(store, d).meta["label_type"]: d for d in tracks.meta["dependencies"]}
    return tracks, load_label_set(store, deps["two_d_tracklets"]), load_label_set(store, deps["frame_alignment"])


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    name: str
    K: np.ndarray       # canvas intrinsics (as calibrated)
    D: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    native_to_video: float
    padding: np.ndarray  # canvas = video * scale + padding
    scale: float
    video_hw: tuple[int, int]
    R: np.ndarray = field(init=False)
    K_video: np.ndarray = field(init=False)
    K_pinhole: np.ndarray = field(init=False)   # undistorted video canvas (alpha = 0)

    def __post_init__(self):
        self.R = cv2.Rodrigues(self.rvec)[0]
        # canvas = scale * video + padding  =>  video intrinsics = (K - padding shift) / scale.
        A = np.array([[1 / self.scale, 0, -self.padding[0] / self.scale],
                      [0, 1 / self.scale, -self.padding[1] / self.scale], [0, 0, 1]])
        self.K_video = A @ self.K
        h, w = self.video_hw
        self.K_pinhole = cv2.getOptimalNewCameraMatrix(self.K_video, self.D, (w, h), 0, (w, h))[0]

    def native_to_canvas(self, uv: np.ndarray) -> np.ndarray:
        return uv * self.native_to_video * self.scale + self.padding

    def project_canvas(self, X: np.ndarray) -> np.ndarray:
        return cv2.projectPoints(X.reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.D)[0].reshape(-1, 2)

    def depth(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.R.T + self.tvec)[:, 2]

    def native_to_pinhole(self, uv: np.ndarray) -> np.ndarray:
        """Native detector pixels -> undistorted (alpha=0) video pixels."""
        video = uv * self.native_to_video
        return cv2.undistortPoints(video.reshape(-1, 1, 2).astype(np.float64), self.K_video, self.D,
                                   P=self.K_pinhole).reshape(-1, 2)


def load_cameras(calibration: dict, names: list[str], native_to_video: float, video_hw=(960, 1280)):
    cams = []
    for name in names:
        c = calibration["cameras"][name.rsplit("-", 1)[-1]]
        cams.append(Camera(name, np.array(c["K"], float), np.array(c["D"], float),
                           np.array(c["rvec"], float).reshape(3), np.array(c["tvec"], float).reshape(3),
                           native_to_video, np.array(calibration.get("padding", [0, 0]), float),
                           float(calibration.get("scale", 1.0)), tuple(video_hw)))
    return cams


def joint_sigma(X: np.ndarray, observations: list[tuple[Camera, np.ndarray]], residual_floor_px: float,
                min_sine: float) -> float:
    """Uncertainty (world units) of a triangulated point from its supporting views.

    Each view constrains the point laterally to ~ residual * depth / focal; the views
    combine inverse-variance, and the depth direction is inflated by the best ray
    angle (1 / sin(theta), capped by ``min_sine``).
    """
    lateral, rays = [], []
    for cam, uv_canvas in observations:
        residual = float(np.linalg.norm(cam.project_canvas(X)[0] - uv_canvas))
        depth = float(cam.depth(X[None])[0])
        if depth <= 0:
            continue
        lateral.append(max(residual, residual_floor_px) * depth / cam.K[0, 0])
        centre = -cam.R.T @ cam.tvec
        rays.append((X - centre) / np.linalg.norm(X - centre))
    if len(lateral) < 2:
        return float("nan")
    combined = 1.0 / np.sqrt(np.sum(1.0 / np.square(lateral)))
    best = max(np.linalg.norm(np.cross(a, b)) for i, a in enumerate(rays) for b in rays[i + 1:])
    return combined / max(best, min_sine)


# --------------------------------------------------------------------------- #
# Targets                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class GoldenConfig:
    grid_step: int = 5                # frames between labelled (triangulated) frames
    snap_frames: int = 2              # a 2-D detection labels the nearest grid frame within this
    min_confidence: float = 0.3       # reliable 2-D joint (the triangulation's own threshold)
    residual_floor_px: float = 1.5    # canvas pixels; detector/annotation noise floor
    min_ray_sine: float = 0.25        # depth inflation cap (rays closer than ~14 degrees)
    calibration_sigma_mm: float = 25.0
    # A root without any measured joint (root-only observations: partial detections whose
    # shoulders/hips locate the person but not the pose) has no residual-based sigma.
    root_only_sigma_mm: float = 150.0
    interpolate: bool = True          # linear interpolation between consecutive grid frames
    # How interpolated frames are used.  "match" (default): only the interpolated *root* is kept,
    # as ``match_root`` -- a matching cue so detection queries can be assigned to present people
    # (existence / recall supervision) -- and frames between labels carry no 3-D loss target, since
    # linear interpolation is biased for fast athletic motion.  "loss": legacy, interpolated
    # keypoints and roots are loss targets with a motion-inflated sigma.
    interpolated_use: str = "match"
    native_to_video: float = 0.625    # 2048x1536 -> 1280x960 (verified by reprojection)
    # Triangulation outliers the residual-based sigma does not reveal.  Height alone is never a
    # reason: people jump, hang from pull-up bars and climb ropes.  Suspicion becomes
    # uncertainty unless the joint's geometry is also weak (<= weak_views reliable views).
    # * below the floor (z up, floor z = 0): dropped;
    # * body-inconsistent: every bone of the joint deviates from that person's median bone length
    #   (whole track) by more than max(ratio * median, mm) -> dropped if weak, else its sigma is
    #   raised to at least the deviation;
    # * spike: more than spike_mm from the midpoint of its two neighbouring label frames, while
    #   those neighbours are closer than that to each other -> dropped if weak, or if beyond
    #   spike_hard_mm.  On the golden set most such "spikes" are well-observed fast arm motion
    #   (kettlebell swings look like spikes at 2 Hz labelling), so they are otherwise kept.
    floor_tolerance_mm: float = 100.0
    bone_tolerance_ratio: float = 0.5
    bone_tolerance_mm: float = 120.0
    spike_mm: float = 350.0
    spike_hard_mm: float = 1000.0
    weak_views: int = 2


def _spikes(points: np.ndarray, frames: np.ndarray, limit: float) -> np.ndarray:
    """(F, ..., 3) points: True where a label frame deviates from both label-frame neighbours."""
    out = np.zeros(points.shape[:-1], bool)
    for prev, cur, nxt in zip(frames[:-2], frames[1:-1], frames[2:]):
        a, b, c = points[prev], points[cur], points[nxt]
        ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1) & np.isfinite(c).all(-1)
        deviation = np.linalg.norm(b - (a + c) / 2, axis=-1)
        neighbours = np.linalg.norm(c - a, axis=-1)
        out[cur] = ok & (deviation > limit) & (neighbours < deviation)
    return out


def reject_outliers(keypoints: np.ndarray, root: np.ndarray, grid: np.ndarray, unit: float,
                    cfg: "GoldenConfig", reliable_views: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Invalidate (NaN) implausible triangulated points in place.

    ``reliable_views`` (F, P, J): supporting views per joint (None = all treated as strong).
    Returns (counts per reason, minimum sigma (F, P, J) in world units for suspicious joints kept).
    """
    weak = (np.zeros(keypoints.shape[:-1], bool) if reliable_views is None
            else reliable_views <= cfg.weak_views)
    min_sigma = np.zeros(keypoints.shape[:-1])
    below = np.isfinite(keypoints[..., 2]) & (keypoints[..., 2] * unit < -cfg.floor_tolerance_mm)
    keypoints[below] = np.nan
    root_below = np.isfinite(root[..., 2]) & (root[..., 2] * unit < -cfg.floor_tolerance_mm)
    root[root_below] = np.nan
    # Body consistency: per person, the median length of each bone over the whole track.
    edges = np.array(SKELETON_EDGES)
    lengths = np.linalg.norm(keypoints[:, :, edges[:, 0]] - keypoints[:, :, edges[:, 1]], axis=-1)  # (F, P, E)
    median = np.nanmedian(np.where(np.isfinite(lengths), lengths, np.nan), axis=0)                  # (P, E)
    tolerance = np.maximum(cfg.bone_tolerance_ratio * median, cfg.bone_tolerance_mm / unit)
    bad_bone = np.abs(lengths - median[None]) > tolerance[None]                                     # NaN -> False
    measured_bone = np.isfinite(lengths)
    deviation = np.abs(lengths - median[None])
    inconsistent = np.zeros(keypoints.shape[:-1], bool)
    for j in range(keypoints.shape[2]):
        incident = np.flatnonzero((edges == j).any(1))
        known = measured_bone[..., incident]
        inconsistent[..., j] = known.any(-1) & (bad_bone[..., incident] | ~known).all(-1)
        min_sigma[..., j] = np.where(inconsistent[..., j],
                                     np.nanmin(np.where(known, deviation[..., incident], np.inf), axis=-1), 0)
    drop_inconsistent = inconsistent & weak
    keypoints[drop_inconsistent] = np.nan
    min_sigma[~(inconsistent & ~weak)] = 0
    frames = np.flatnonzero(grid)
    spikes = _spikes(keypoints, frames, cfg.spike_mm / unit)
    drop_spikes = spikes & (weak | _spikes(keypoints, frames, cfg.spike_hard_mm / unit))
    keypoints[drop_spikes] = np.nan
    min_sigma[drop_spikes] = 0
    root_spikes = _spikes(root, frames, cfg.spike_hard_mm / unit)
    root[root_spikes] = np.nan
    return {"keypoints_below_floor": int(below.sum()), "roots_below_floor": int(root_below.sum()),
            "keypoints_body_inconsistent_dropped": int(drop_inconsistent.sum()),
            "keypoints_body_inconsistent_sigma_raised": int((inconsistent & ~weak & np.isfinite(keypoints).all(-1)).sum()),
            "keypoints_spikes_dropped": int(drop_spikes.sum()),
            "keypoints_spikes_kept_well_observed": int((spikes & ~drop_spikes).sum()),
            "root_spikes_dropped": int(root_spikes.sum())}, min_sigma


def build_targets(tracks_doc: dict, tracklet_docs: dict[str, dict], alignment: dict, cameras: list[Camera],
                  cfg: GoldenConfig) -> dict[str, np.ndarray]:
    """All per-frame arrays in world millimetres (before the cache's xy offset)."""
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
    unit = UNIT_MM[tracks_doc.get("units", "mm")]
    if tracks_doc.get("keypoint_format", "coco17") != "coco17":
        raise ValueError("Only coco17 keypoints are supported")
    lo, hi = alignment.get("source", {}).get("valid_aligned_range",
                                             [0, max(cp["aligned_index"] for cp in alignment["checkpoints"])])
    F = hi + 1
    tracks = sorted(tracks_doc["tracks"], key=lambda t: t["track_id"])
    P, C, J = len(tracks), len(cameras), len(JOINT_NAMES)
    slot = {t["track_id"]: p for p, t in enumerate(tracks)}
    cam_index = {cam.name: c for c, cam in enumerate(cameras)}
    grid = np.zeros(F, bool)
    grid[lo:F:cfg.grid_step] = True

    keypoints = np.full((F, P, J, 3), np.nan)   # world units until the end
    root = np.full((F, P, 3), np.nan)
    for t in tracks:
        for pos in t.get("positions", []):
            a = pos["aligned_index"]
            if not lo <= a < F:
                continue
            kp = pos.get("keypoints")
            if kp:
                coco = np.array([[np.nan] * 3 if k is None else k for k in kp], float)
                keypoints[a, slot[t["track_id"]]] = coco[COCO_TO_JOINTS]
            if pos.get("xyz") is not None:
                root[a, slot[t["track_id"]]] = pos["xyz"]


    member = {(m["camera"], m["tracklet_id"]): slot[t["track_id"]] for t in tracks for m in t["members"]}
    native_to_aligned = {cam.name: {} for cam in cameras}
    for cp in alignment["checkpoints"]:
        for name, frame in cp["frames"].items():
            if frame is not None and name in native_to_aligned:
                native_to_aligned[name][frame] = cp["aligned_index"]
    # Detections snapped to the label grid; per (frame, person, camera) the closest in time wins.
    canvas2d = np.full((F, P, C, J, 3), np.nan)
    pinhole2d = np.full((F, P, C, J, 3), np.nan)
    snap_distance = np.full((F, P, C), np.inf)
    for name, doc in tracklet_docs.items():
        if name not in cam_index:
            continue
        c, cam = cam_index[name], cameras[cam_index[name]]
        for record in doc["frames"]:
            a = native_to_aligned[name].get(record["frame"])
            if a is None:
                continue
            g = int(round((a - lo) / cfg.grid_step)) * cfg.grid_step + lo
            if not lo <= g < F or abs(a - g) > cfg.snap_frames:
                continue
            for det in record["detections"]:
                p = member.get((name, det["tracklet_id"]))
                if p is None or abs(a - g) >= snap_distance[g, p, c]:
                    continue
                coco = np.array(det["keypoints"], float)[COCO_TO_JOINTS]
                snap_distance[g, p, c] = abs(a - g)
                canvas2d[g, p, c, :, :2] = cam.native_to_canvas(coco[:, :2])
                canvas2d[g, p, c, :, 2] = coco[:, 2]
                pinhole2d[g, p, c, :, :2] = cam.native_to_pinhole(coco[:, :2])
                pinhole2d[g, p, c, :, 2] = coco[:, 2]
    observed = np.isfinite(snap_distance)
    reliable_views = (observed[..., None] & (np.nan_to_num(canvas2d[..., 2]) >= cfg.min_confidence)).sum(2)
    rejected, min_sigma = reject_outliers(keypoints, root, grid, unit, cfg, reliable_views)

    # Per-joint uncertainty of every triangulated grid target.
    sigma = np.full((F, P, J), np.nan)
    for a, p in zip(*np.nonzero(np.isfinite(keypoints).all(-1).any(-1))):
        for j in range(J):
            X = keypoints[a, p, j]
            if not np.isfinite(X).all():
                continue
            views = [(cameras[c], canvas2d[a, p, c, j, :2]) for c in range(C)
                     if observed[a, p, c] and canvas2d[a, p, c, j, 2] >= cfg.min_confidence]
            sigma[a, p, j] = joint_sigma(X, views, cfg.residual_floor_px, cfg.min_ray_sine)
    calibration = cfg.calibration_sigma_mm / unit
    sigma = np.where(np.isfinite(sigma), np.maximum(sigma, min_sigma), sigma)
    # A target without two reliable supporting views (or failed geometry) gets the worst
    # supported sigma of its frame, never a falsely tight one.
    measured = np.isfinite(keypoints).all(-1)
    fallback = np.nanmax(np.where(np.isfinite(sigma), sigma, np.nan), axis=-1, initial=np.nan)
    sigma = np.where(measured & ~np.isfinite(sigma), fallback[..., None], sigma)
    sigma = np.where(measured & ~np.isfinite(sigma), 4 * calibration, sigma)
    sigma = np.sqrt(np.square(sigma) + calibration ** 2)
    hips = [JOINT_NAMES.index("left_hip"), JOINT_NAMES.index("right_hip")]
    root_sigma = np.where(np.isfinite(root).all(-1), np.nanmean(sigma[..., hips], axis=-1), np.nan)
    root_sigma = np.where(np.isfinite(root).all(-1) & ~np.isfinite(root_sigma),
                          np.nanmean(np.where(measured, sigma, np.nan), axis=-1), root_sigma)
    root_sigma = np.where(np.isfinite(root).all(-1) & ~np.isfinite(root_sigma),
                          cfg.root_only_sigma_mm / unit, root_sigma)

    triangulated = np.isfinite(root).all(-1) | measured.any(-1)
    interpolated = np.zeros((F, P), bool)
    if cfg.interpolate:
        starts = np.flatnonzero(grid)
        for a, b in zip(starts[:-1], starts[1:]):
            for f in range(a + 1, b):
                w = (f - a) / (b - a)
                motion = min(w, 1 - w)
                both = measured[a] & measured[b]
                kp = (1 - w) * keypoints[a] + w * keypoints[b]
                keypoints[f] = np.where(both[..., None], kp, keypoints[f])
                gap = np.linalg.norm(keypoints[b] - keypoints[a], axis=-1)
                sigma[f] = np.where(both, (1 - w) * sigma[a] + w * sigma[b] + motion * gap, sigma[f])
                root_both = np.isfinite(root[a]).all(-1) & np.isfinite(root[b]).all(-1)
                root[f] = np.where(root_both[..., None], (1 - w) * root[a] + w * root[b], root[f])
                root_gap = np.linalg.norm(root[b] - root[a], axis=-1)
                root_sigma[f] = np.where(root_both, (1 - w) * root_sigma[a] + w * root_sigma[b]
                                         + motion * root_gap, root_sigma[f])
                interpolated[f] = both.any(-1) | root_both
    match_root = np.full_like(root, np.nan)
    if cfg.interpolated_use not in ("match", "loss"):
        raise ValueError(f"interpolated_use must be 'match' or 'loss', not {cfg.interpolated_use!r}")
    if cfg.interpolate and cfg.interpolated_use == "match":
        between = ~grid
        match_root[between] = root[between]
        keypoints[between] = np.nan
        sigma[between] = np.nan
        root[between] = np.nan
        root_sigma[between] = np.nan
        interpolated[:] = False
    # Unlabelled frames copy the existence of the nearest grid frame (ties go earlier).
    labeled = grid.copy()
    nearest = np.empty(F, int)
    labels = np.flatnonzero(grid)
    for f in range(F):
        nearest[f] = labels[np.argmin(np.abs(labels - f) + (labels > f) * 0.1)]
    observed_all = observed[nearest]
    kp2d = np.where(grid[:, None, None, None, None], pinhole2d, np.nan)
    keypoint_valid = np.isfinite(keypoints).all(-1)
    root_valid = np.isfinite(root).all(-1)
    for name, valid, value in (("keypoint", keypoint_valid, sigma), ("root", root_valid, root_sigma)):
        bad = valid & ~np.isfinite(value)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} valid {name} targets have a non-finite sigma")
    return {
        "frames": np.arange(F), "person_ids": np.array([t["track_id"] for t in tracks]),
        "person_source_ids": np.array([str(t.get("source_id", t["track_id"])) for t in tracks]),
        "labeled": labeled, "observed": observed_all,
        "keypoints": np.where(keypoint_valid[..., None], keypoints * unit, 0).astype(np.float32),
        "keypoint_valid": keypoint_valid,
        "keypoint_sigma": np.where(keypoint_valid, sigma * unit, 0).astype(np.float32),
        "root": np.where(root_valid[..., None], root * unit, 0).astype(np.float32),
        "root_valid": root_valid,
        "root_sigma": np.where(root_valid, root_sigma * unit, 0).astype(np.float32),
        "triangulated": triangulated & grid[:, None], "interpolated": interpolated,
        "match_root": np.nan_to_num(match_root * unit).astype(np.float32),
        "match_root_valid": np.isfinite(match_root).all(-1),
        "kp2d_video": kp2d,  # undistorted video pixels; converted to cache pixels at write time
        "rejected": rejected,
    }


def coverage_summary(arrays: dict, camera_names: list[str]) -> dict:
    grid = arrays["labeled"]
    observed = arrays["observed"][grid]
    views = observed.sum(-1)
    single = (views == 1)
    tri = arrays["triangulated"][grid]
    sig = arrays["keypoint_sigma"][grid][arrays["keypoint_valid"][grid]]
    return {
        "grid_frames": int(grid.sum()),
        "person_frames_by_views": {int(k): int((views == k).sum()) for k in range(1, observed.shape[-1] + 1)},
        "observed_not_triangulated": int(((views > 0) & ~tri).sum()),
        "single_view_by_camera": {camera_names[c]: int((single & observed[..., c]).sum())
                                  for c in range(len(camera_names))},
        "keypoint_sigma_mm_percentiles": {str(q): float(np.percentile(sig, q)) for q in (10, 50, 90, 99)}
        if len(sig) else {},
    }
