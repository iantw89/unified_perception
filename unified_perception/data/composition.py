"""Versioned, source-isolated multi-person cutout composition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..constants import NUM_JOINTS, SKELETON_EDGES

VERSION = 2


class CompositionRejected(ValueError):
    """A candidate fails a documented composition quality constraint."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class Clip:
    metadata: dict[str, Any]
    imgs: np.ndarray
    masks: np.ndarray
    joints: np.ndarray
    world2img: np.ndarray
    intrinsics: np.ndarray
    mask_boxes: np.ndarray | None = field(default=None, repr=False)

    def validate(self) -> None:
        if self.imgs.ndim != 5 or self.imgs.shape[2] != 3 or self.imgs.dtype != np.uint8:
            raise ValueError("Cached imgs must be uint8 (T,N,3,H,W)")
        t, n, _, h, w = self.imgs.shape
        if t < 1 or n < 1 or self.masks.shape != (t, n, h, w) or self.masks.dtype != bool:
            raise ValueError("Cached masks must be bool (T,N,H,W)")
        if self.joints.shape != (t, NUM_JOINTS, 3):
            raise ValueError("Cached joints must be (T,12,3) in millimetres")
        for array in (self.joints, self.world2img, self.intrinsics):
            if not np.isfinite(array).all():
                raise ValueError("Non-finite cached geometry")
        if self.world2img.shape != (t, n, 4, 4) or self.intrinsics.shape != (t, n, 4, 4):
            raise ValueError("Cached calibration must be (T,N,4,4)")
        if not np.allclose(self.intrinsics[..., 2, :], [0, 0, 1, 0]) or not np.allclose(self.intrinsics[..., 3, :], [0, 0, 0, 1]):
            raise ValueError("Expected pinhole 4x4 intrinsics")
        if not self.masks.any(axis=(-1, -2)).all():
            raise ValueError("Every cached source view needs a foreground mask")
        meta = self.metadata
        for key in ("id", "split", "split_group", "capture_group", "subject", "sequence",
                    "camera_ids", "frame_ids", "fps", "extraction_hash", "geometry_verified",
                    "source_sequence"):
            if key not in meta:
                raise ValueError(f"Missing cache metadata: {key}")
        if meta["split"] not in ("train", "val", "test"):
            raise ValueError("Cache split must be train, val, or test")
        if not meta["geometry_verified"] or not meta["capture_group"] or not meta["split_group"]:
            raise ValueError("Explicit verified capture and source split groups are required")
        if len(meta["camera_ids"]) != n or len(set(meta["camera_ids"])) != n:
            raise ValueError("Camera IDs must match the cache axis and be unique")
        frames = np.asarray(meta["frame_ids"])
        if frames.shape != (t,) or (t > 1 and not np.all(np.diff(frames) > 0)):
            raise ValueError("Source frame IDs must be strictly increasing")
        if not np.isfinite(meta["fps"]) or meta["fps"] <= 0:
            raise ValueError("A positive effective window FPS is required")
        # The model assumes undistorted pinhole inputs.
        if meta.get("distortion", "unknown") != "undistorted":
            raise ValueError("Composition requires verified undistorted pinhole images")
        for ti in range(t):
            for ni in range(n):
                extr = np.linalg.solve(self.intrinsics[ti, ni], self.world2img[ti, ni])
                rotation = extr[:3, :3]
                if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
                    raise ValueError("world2img/intrinsics do not define a rigid camera")
                if not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
                    raise ValueError("Camera rotation must be right-handed")


def _asset_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Cache paths must stay inside the manifest directory")
    return path


def verify_clip_assets(root: Path, entry: dict[str, Any], *, full_hash: bool = False) -> None:
    assets = entry.get("assets", [dict(path=entry["path"], sha256=entry["sha256"])])
    for asset in assets:
        path = _asset_path(root, asset["path"])
        stat = path.stat()
        if "size" in asset and (stat.st_size != asset["size"] or stat.st_mtime_ns != asset["mtime_ns"]):
            raise ValueError(f"Stale or modified cache: {path}")
        if (full_hash or "size" not in asset) and file_hash(path) != asset["sha256"]:
            raise ValueError(f"Stale or modified cache: {path}")


def save_clip(root: Path, clip: Clip, *, storage: str = "npz") -> None:
    """Write lossless sources; NPY pixels are mapped lazily by training workers."""
    path = _asset_path(root, clip.metadata["path"])
    arrays = dict(imgs=clip.imgs, masks=clip.masks, joints=clip.joints,
                  world2img=clip.world2img, intrinsics=clip.intrinsics)
    if storage == "npz":
        np.savez_compressed(path, **arrays)
    elif storage == "npy":
        assets = []
        for name in ("imgs", "masks"):
            pixel_path = path.with_name(path.stem + f"-{name}.npy")
            np.save(pixel_path, arrays.pop(name), allow_pickle=False)
            assets.append(pixel_path)
            clip.metadata[name + "_path"] = pixel_path.name
        np.savez(path, **arrays)
        assets.append(path)
        clip.metadata["mask_boxes"] = [[cv2.boundingRect(mask.astype(np.uint8)) for mask in frame]
                                      for frame in clip.masks]
        clip.metadata.update(storage="npy", assets=[dict(
            path=asset.name, sha256=file_hash(asset), size=asset.stat().st_size,
            mtime_ns=asset.stat().st_mtime_ns,
        ) for asset in assets])
    else:
        raise ValueError(f"Unknown composition storage: {storage}")
    clip.metadata["sha256"] = file_hash(path)


def load_clip(root: Path, entry: dict[str, Any], verify_hash: bool = True) -> Clip:
    path = _asset_path(root, entry["path"])
    if verify_hash:
        verify_clip_assets(root, entry, full_hash=True)
    with np.load(path, allow_pickle=False) as data:
        if entry.get("storage") == "npy":
            imgs = np.load(_asset_path(root, entry["imgs_path"]), mmap_mode="r", allow_pickle=False)
            masks = np.load(_asset_path(root, entry["masks_path"]), mmap_mode="r", allow_pickle=False)
        else:
            imgs, masks = data["imgs"], data["masks"]
        clip = Clip(entry, imgs, masks, data["joints"], data["world2img"], data["intrinsics"])
    clip.validate()
    if "mask_boxes" in entry:
        clip.mask_boxes = np.asarray(entry["mask_boxes"], dtype=int)
    return clip


def validate_catalog(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str], str] = {}
    for entry in entries:
        identity = entry["id"]
        if identity in by_id:
            raise ValueError(f"Duplicate source clip ID: {identity}")
        by_id[identity] = entry
        for key in ("split_group", "sequence", "source_sequence"):
            group = (key, entry.get(key, entry["sequence"]))
            previous = groups.setdefault(group, entry["split"])
            if previous != entry["split"]:
                raise ValueError(f"Source split leakage for {key}={entry[key]}")
    return by_id


def validate_source_datasets(entries: list[dict[str, Any]], allowed: list[str]) -> None:
    if not allowed or not set(allowed) <= {"h36m", "mpi"}:
        raise ValueError("Composition supports H36M and MPI only; MOYO is excluded")
    for entry in entries:
        dataset = entry.get("dataset", entry.get("source_sequence", "").split(":")[0])
        if dataset not in allowed:
            raise ValueError(f"Source {entry['id']} belongs to excluded composition dataset {dataset!r}")


def _compatible(clips: list[Clip]) -> None:
    base = clips[0]
    for clip in clips[1:]:
        for key in ("split", "capture_group", "camera_ids", "fps", "extraction_hash"):
            if clip.metadata[key] != base.metadata[key]:
                raise CompositionRejected(f"incompatible_{key}")
        if clip.imgs.shape != base.imgs.shape:
            raise CompositionRejected("incompatible_image_shape")
        # Equality is intentionally strict: sequence-specific alignment offsets
        # must not be mistaken for a shared world frame.
        if not np.array_equal(clip.world2img, base.world2img) or not np.array_equal(clip.intrinsics, base.intrinsics):
            raise CompositionRejected("incompatible_calibration")


def _body_boxes(joints: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(SKELETON_EDGES + [(0, 3), (1, 2)])
    points = joints[:, edges]  # (T,bones,2,3)
    lower = points.min(axis=2) - radius
    upper = points.max(axis=2) + radius
    # Include a conservative head region inferred from shoulders/hips; the
    # source skeleton has no head joint.
    shoulders = joints[:, 2:4].mean(axis=1)
    hips = joints[:, :2].mean(axis=1)
    head = shoulders + 0.6 * (shoulders - hips)
    lower = np.concatenate((lower, (head - radius)[:, None]), axis=1)
    upper = np.concatenate((upper, (head + radius)[:, None]), axis=1)
    if len(joints) > 1:
        lower = np.minimum(lower[:-1], lower[1:])
        upper = np.maximum(upper[:-1], upper[1:])
    return lower, upper


def _check_collision(clips: list[Clip], radius: float) -> None:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Body proxy radius must be positive")
    boxes = [_body_boxes(clip.joints, radius) for clip in clips]
    for i, (lo, hi) in enumerate(boxes):
        for other_lo, other_hi in boxes[:i]:
            overlap = (lo[:, :, None] <= other_hi[:, None]).all(axis=-1)
            overlap &= (hi[:, :, None] >= other_lo[:, None]).all(axis=-1)
            if overlap.any():
                raise CompositionRejected("body_proxy_collision")


def _blend_intersections(image: np.ndarray, owner: np.ndarray, masks: list[np.ndarray],
                         rng: np.random.Generator) -> np.ndarray:
    """Mix a 3x3 neighborhood only at visible, overlapping person boundaries."""
    h, w = owner.shape
    if not any(np.logical_and(masks[i], masks[j]).any()
               for i in range(len(masks)) for j in range(i)):
        rng.bit_generator.advance(8 * h * w)
        return image
    padded_owner = np.pad(owner, 1, constant_values=-1)
    mask_stack = np.stack(masks)
    padded_masks = np.pad(mask_stack, ((0, 0), (1, 1), (1, 1)))
    seam = np.zeros_like(owner, dtype=bool)
    neighbors = [(dy, dx) for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)]
    for dy, dx in neighbors:
        neighbor = padded_owner[dy:dy + h, dx:dx + w]
        boundary = (owner >= 0) & (neighbor >= 0) & (owner != neighbor)
        if not boundary.any():
            continue
        yy, xx = np.nonzero(boundary)
        overlap = mask_stack[neighbor[yy, xx], yy, xx]
        overlap |= padded_masks[owner[yy, xx], yy + dy, xx + dx]
        seam[yy[overlap], xx[overlap]] = True
    yy, xx = np.nonzero(seam)
    if not len(yy):
        # uniform(float64) consumes one PCG64 draw per pixel. Preserve the
        # reference stream for subsequent views without allocating random grids.
        rng.bit_generator.advance(8 * h * w)
        return image
    linear = yy * w + xx
    first, last = int(linear.min()), int(linear.max()) + 1
    total = image[:, yy, xx].astype(np.float32)
    weight_sum = np.ones(len(yy), dtype=np.float32)
    padded_image = np.pad(image, ((0, 0), (1, 1), (1, 1)))
    for dy, dx in neighbors:
        neighbor = padded_owner[yy + dy, xx + dx]
        people = (owner[yy, xx] >= 0) & (neighbor >= 0)
        rng.bit_generator.advance(first)
        weights = rng.uniform(0.25, 1.0, last - first).astype(np.float32)[linear - first] * people
        rng.bit_generator.advance(h * w - last)
        total += padded_image[:, yy + dy, xx + dx].astype(np.float32) * weights[None]
        weight_sum += weights
    result = image.copy()
    result[:, yy, xx] = np.rint(total / weight_sum[None]).clip(0, 255).astype(np.uint8)
    return result


def compose(clips: list[Clip], position_range: list[float], radius_mm: float = 150.0,
            depth_margin_mm: float = 150.0, *,
            blend_intersections: bool = True, geometry_only: bool = False) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Return RGB (None for geometry-only), visible fractions and near-to-far orders."""
    if not clips:
        raise ValueError("A composition needs a base clip")
    if not np.isfinite(depth_margin_mm) or depth_margin_mm <= 0:
        raise ValueError("Depth margin must be positive")
    bounds = np.asarray(position_range)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or np.any(bounds[:3] >= bounds[3:]):
        raise ValueError("Invalid position_range")
    for clip in clips:
        # Only immutable mapped masks have reusable foreground bounds.
        if clip.masks.flags.writeable:
            clip.mask_boxes = None
        clip.validate()
        if np.any(clip.joints < bounds[:3]) or np.any(clip.joints > bounds[3:]):
            raise CompositionRejected("outside_capture_volume")
    _compatible(clips)
    _check_collision(clips, radius_mm)
    t_count, n_count = clips[0].imgs.shape[:2]
    images = None if geometry_only else clips[0].imgs.copy()
    visible = np.zeros((t_count, n_count, len(clips)), dtype=np.float32)
    orders = np.zeros((t_count, n_count, len(clips)), dtype=np.int64)
    signature = json.dumps([
        (clip.metadata["id"], clip.metadata["extraction_hash"], clip.metadata["frame_ids"])
        for clip in clips
    ], sort_keys=True).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(signature).digest()[:8], "little"))
    for t in range(t_count):
        for n in range(n_count):
            masks = [clip.masks[t, n] for clip in clips]
            ranges = []
            for clip in clips:
                shoulders = clip.joints[t, 2:4].mean(axis=0)
                hips = clip.joints[t, :2].mean(axis=0)
                points = np.concatenate((clip.joints[t], (shoulders + 0.6 * (shoulders - hips))[None]))
                depth = points @ clip.world2img[t, n, 2, :3] + clip.world2img[t, n, 2, 3]
                margin = max(depth_margin_mm, radius_mm)
                ranges.append((depth.min() - margin, depth.max() + margin))
            if any(low <= 0 for low, _ in ranges):
                raise CompositionRejected("behind_camera")
            # Edges encode only proven ordering at overlapping foreground pixels.
            edges: list[tuple[int, int]] = []
            for i in range(len(clips)):
                for j in range(i):
                    if not np.logical_and(masks[i], masks[j]).any():
                        continue
                    if ranges[i][1] < ranges[j][0]:
                        edges.append((i, j))
                    elif ranges[j][1] < ranges[i][0]:
                        edges.append((j, i))
                    else:
                        raise CompositionRejected("ambiguous_depth_overlap")
            remaining = set(range(len(clips)))
            order = []
            while remaining:
                ready = sorted(i for i in remaining if not any(b == i and a in remaining for a, b in edges))
                if not ready:
                    raise CompositionRejected("depth_order_cycle")
                order.extend(ready)
                remaining.difference_update(ready)
            occupied = np.zeros_like(masks[0])
            for i in order:
                shown = masks[i] & ~occupied
                visible[t, n, i] = shown.sum() / masks[i].sum()
                occupied |= masks[i]
            if not geometry_only:
                owner = np.full(masks[0].shape, -1, dtype=np.int64)
                for i in reversed(order):
                    if clips[i].mask_boxes is None:
                        clips[i].mask_boxes = np.asarray([
                            [cv2.boundingRect(mask.astype(np.uint8)) for mask in frame]
                            for frame in clips[i].masks], dtype=int)
                    x, y, w, h = clips[i].mask_boxes[t, n]
                    area = np.s_[y:y + h, x:x + w]
                    np.copyto(images[t, n][:, area[0], area[1]],
                              clips[i].imgs[t, n][:, area[0], area[1]],
                              where=masks[i][area][None])
                    owner[area][masks[i][area]] = i
                if blend_intersections and len(clips) > 1:
                    if edges:
                        images[t, n] = _blend_intersections(images[t, n], owner, masks, rng)
                    else:
                        rng.bit_generator.advance(8 * masks[0].size)
            orders[t, n] = order
    return images, visible, orders


def generate_recipes(entries: list[dict[str, Any]], root: Path, split: str, count: int,
                     people: list[int], seed: int, position_range: list[float],
                     radius_mm: float = 150.0, depth_margin_mm: float = 150.0,
                     max_attempts: int = 10000, *,
                     clip_cache: dict[str, Clip] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_catalog(entries)
    if split not in ("train", "val", "test") or count <= 0 or not people or any(p < 2 or p > 8 for p in people):
        raise ValueError("Require a valid split, positive count, and requested counts in 2..8")
    if len(set(people)) != len(people) or max_attempts < count:
        raise ValueError("Counts must be unique and attempt budget must cover requested recipes")
    eligible = [e for e in entries if e["split"] == split and e.get("approved") is True]
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in eligible:
        groups.setdefault(entry["capture_group"], []).append(entry)
    if not groups:
        raise ValueError(f"No visually approved sources for {split}")
    rng = np.random.default_rng(seed)
    recipes = []
    seen = set()
    rejected: Counter[str] = Counter()
    cache: dict[str, Clip] = {} if clip_cache is None else clip_cache
    for _ in range(max_attempts):
        p = people[len(recipes) % len(people)]
        candidates = [g for g in groups.values() if len(g) >= p]
        if not candidates:
            rejected[f"insufficient_sources_{p}"] += 1
            break
        group = candidates[int(rng.integers(len(candidates)))]
        selection = [group[i] for i in rng.choice(len(group), p, replace=False)]
        ids = [e["id"] for e in selection]
        # Preserve the base role, but canonicalize donor permutations.
        ids = ids[:1] + sorted(ids[1:])
        signature = tuple(ids)
        if signature in seen:
            rejected["duplicate_recipe"] += 1
            continue
        by_id = {e["id"]: e for e in selection}
        # Bound decoded RGB/mask memory to the current candidate, not the corpus.
        if clip_cache is None:
            cache = {i: clip for i, clip in cache.items() if i in ids}
        for identity in ids:
            if identity not in cache:
                cache[identity] = load_clip(root, by_id[identity])
        try:
            _, _, order = compose([cache[i] for i in ids], position_range, radius_mm, depth_margin_mm, geometry_only=True)
        except CompositionRejected as exc:
            rejected[str(exc)] += 1
            continue
        seen.add(signature)
        recipes.append({"id": f"{split}-{len(recipes):07d}", "split": split, "sources": ids,
                        "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()})
        if len(recipes) == count:
            break
    report = {"requested": count, "achieved": len(recipes), "seed": seed,
              "counts": dict(Counter(len(r["sources"]) for r in recipes)),
              "rejections": dict(rejected)}
    return recipes, report


class CompositionDataset(Dataset):
    """A frozen recipe bank; DataLoader shuffling supplies training sampling."""

    def __init__(self, cfg: Config, split: str):
        self.cfg = cfg
        path = Path(cfg.data.composition_manifest or "")
        self.root = path.resolve().parent
        manifest = read_json(path)
        if manifest["version"] != VERSION:
            raise ValueError("Unsupported composition manifest version")
        if not manifest.get("quality_approved"):
            raise ValueError("Visually review the generated pilot before training/evaluation")
        self.entries = validate_catalog(manifest["clips"])
        validate_source_datasets(manifest["clips"], cfg.data.composition_datasets)
        if cfg.model.num_joints != NUM_JOINTS:
            raise ValueError("Composition uses the canonical 12-joint skeleton")
        self.recipes = [r for r in manifest["recipes"] if r["split"] == split and r.get("approved") is True]
        self.originals = [e for e in manifest["clips"] if e["split"] == split and e.get("approved") is True]
        self.parameters = manifest["parameters"]
        manifest_range = np.asarray(self.parameters["position_range"], dtype=float)
        model_range = np.asarray(cfg.model.position_range, dtype=float)
        # Composites are placed inside the manifest volume; a model volume may
        # contain it (e.g. the wider gym mixture) but must never clip it.
        if (model_range[:3] > manifest_range[:3]).any() or (model_range[3:] < manifest_range[3:]).any():
            raise ValueError("Manifest and model capture volumes differ")
        # Verify every asset once on construction, then never trust arbitrary paths.
        for entry in (e for e in manifest["clips"] if e["split"] == split):
            verify_clip_assets(self.root, entry)
        for recipe in self.recipes:
            if "pixel_cache" in recipe:
                cached = recipe["pixel_cache"]
                verify_clip_assets(self.root, cached)
                signature = hashlib.sha256(json.dumps(recipe["sources"]).encode()).hexdigest()
                parameters = hashlib.sha256(json.dumps(self.parameters, sort_keys=True).encode()).hexdigest()
                if cached["sources_sha256"] != signature or cached["parameters_sha256"] != parameters:
                    raise ValueError("Cached pixels differ from recipe sources or parameters")
            ids = recipe["sources"]
            if len(ids) < 2 or len(set(ids)) != len(ids) or len(ids) > cfg.data.max_people:
                raise ValueError("Invalid or over-capacity recipe")
            for identity in ids:
                source = self.entries[identity]
                if source["split"] != split or source.get("approved") is not True:
                    raise ValueError("Recipe source is unapproved or crosses a split")
        fraction = cfg.data.composition_original_fraction
        if not 0 <= fraction <= 1:
            raise ValueError("composition_original_fraction must be in [0,1]")
        # Evaluation defaults to composites; fraction=1 explicitly selects originals.
        self.original_only = fraction == 1
        if self.original_only:
            self.n_original = len(self.originals)
            self.recipes = []
        elif split == "train":
            self.n_original = round(len(self.recipes) * fraction / (1 - fraction))
        else:
            self.n_original = 0
        if self.n_original and not self.originals:
            raise ValueError(f"No original sources for {split}")
        if not len(self):
            raise ValueError(f"No composition samples for {split}")

    def __len__(self) -> int:
        return len(self.recipes) + self.n_original

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        expected_order = None
        if index < len(self.recipes):
            recipe = self.recipes[index]
            ids = recipe["sources"]
            identity = recipe["id"]
            expected_order = recipe["order_sha256"]
        else:
            source = self.originals[(index - len(self.recipes)) % len(self.originals)]
            ids = [source["id"]]
            identity = source["id"]
        if expected_order is not None and "pixel_cache" in recipe:
            cached = recipe["pixel_cache"]
            imgs = np.load(_asset_path(self.root, cached["imgs_path"]), mmap_mode="r", allow_pickle=False).copy()
            with np.load(_asset_path(self.root, cached["path"]), allow_pickle=False) as packet:
                joints = packet["joints"]
                world2img, intrinsics = packet["world2img"], packet["intrinsics"]
                visibility, order = packet["visibility"], packet["order"]
            if joints.shape != (len(imgs), len(ids), NUM_JOINTS, 3):
                raise ValueError("Cached recipe joint shape differs from sources")
        else:
            clips = [load_clip(self.root, self.entries[i], verify_hash=False) for i in ids]
            imgs, visibility, order = compose(clips, **self.parameters)
            joints = np.stack([clip.joints for clip in clips], axis=1)
            world2img, intrinsics = clips[0].world2img, clips[0].intrinsics
        if expected_order is not None and hashlib.sha256(order.tobytes()).hexdigest() != expected_order:
            raise ValueError("Composition implementation/order differs from frozen recipe")
        if imgs.ndim != 5 or imgs.shape[2] != 3 or imgs.dtype != np.uint8:
            raise ValueError("Cached recipe RGB must be uint8 (T,N,3,H,W)")
        t, n, _, h, w = imgs.shape
        if visibility.shape != (t, n, len(ids)) or order.shape != (t, n, len(ids)):
            raise ValueError("Cached recipe geometry shape differs from sources")
        if not all(np.isfinite(array).all() for array in (joints, world2img, intrinsics, visibility)):
            raise ValueError("Non-finite cached recipe geometry")
        d = self.cfg.data
        valid_cameras = (1 <= n <= d.num_cameras if d.allow_fewer_cameras else n == d.num_cameras)
        if (t, h, w) != (d.seq_len, d.image_height, d.image_width) or not valid_cameras:
            raise ValueError("Cache window/image/camera shape differs from config")
        gt = np.zeros((t, d.max_people, NUM_JOINTS, 3), dtype=np.float32)
        valid = np.zeros((t, d.max_people), dtype=bool)
        vis = np.zeros((t, n, d.max_people), dtype=np.float32)
        instance_ids = np.full((t, d.max_people), -1, dtype=np.int64)
        for p in range(len(ids)):
            gt[:, p] = joints[:, p]
            valid[:, p] = True
            instance_ids[:, p] = p
        vis[:, :, :len(ids)] = visibility
        return {"imgs": torch.from_numpy(imgs), "world2img": torch.from_numpy(world2img.copy()),
                "intrinsics": torch.from_numpy(intrinsics.copy()),
                "gt_keypoints": torch.from_numpy(gt), "gt_valid": torch.from_numpy(valid),
                "instance_ids": torch.from_numpy(instance_ids), "visibility": torch.from_numpy(vis),
                "seq": identity, "start": 0, "capture_group": self.entries[ids[0]]["capture_group"]}
