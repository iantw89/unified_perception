"""Prepare local foreground caches, review pilots, and freeze composition recipes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import load_config
from ..constants import JOINT_NAMES
from ..data.backends import HumanPriorAlignedProvider
from ..data.background import EXTRACTION_METHOD, BackgroundMean, foreground_mask, person_region
from ..data.mpi_masks import apply_official_masks, close_mask_readers
from ..data.composition import (
    VERSION, Clip, compose, file_hash, generate_recipes, load_clip, read_json,
    validate_catalog, validate_source_datasets,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _source_clip(provider: HumanPriorAlignedProvider, source: dict, margin_mm: float) -> Clip:
    identity = source["id"]
    if not identity or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identity):
        raise ValueError("Cache IDs may contain only letters, digits, '-' and '_'")
    seq = provider._seqs[source["sequence"]]
    cameras = [str(c) for c in source["camera_ids"]]
    start = int(source["start"])
    if start < 0 or start + provider.seq_len > seq.num_frames:
        raise ValueError(f"Source {identity} does not cover a complete temporal window")
    if seq.mocap_indices is None:
        raise ValueError(f"Source {identity} needs traceability.json for verified extraction")
    indices = seq.mocap_indices[start:start + provider.seq_len]
    if len(indices) > 1 and not np.all(np.diff(indices) == seq.stride):
        raise ValueError("Composition requires uniform source temporal sampling")
    for frame in provider._mocap_frames(seq)[start:start + provider.seq_len]:
        if any(frame.get(joint) is None for joint in JOINT_NAMES):
            raise ValueError(f"Source {identity} is missing required joint annotations")
    raw = provider.load_window((seq.seq_id, start), camera_ids=cameras)
    if not raw["gt_valid"][:, 0].all() or raw["gt_valid"][:, 1:].any():
        raise ValueError("Extraction requires exactly one labeled person in every source frame")
    imgs = raw["imgs"].cpu().numpy().copy()
    # Rectify using the resized camera matrix, including its padding offset.
    definitions = provider._cam_defs_for_seq(seq)
    for n, camera in enumerate(cameras):
        distortion = np.asarray(definitions[camera].get("dist", np.zeros(5)), dtype=np.float64)
        for t in range(len(imgs)):
            rgb = imgs[t, n].transpose(1, 2, 0)
            imgs[t, n] = cv2.undistort(
                rgb, raw["intrinsics"][t, n, :3, :3], distortion
            ).transpose(2, 0, 1)
    joints = raw["gt_keypoints"][:, 0]
    footprints = np.zeros(imgs.shape[:2] + imgs.shape[-2:], dtype=bool)
    for t in range(len(imgs)):
        for n in range(imgs.shape[1]):
            footprints[t, n], _ = person_region(joints[t], raw["world2img"][t, n], imgs.shape[-2:], margin_mm)
    metadata = dict(source)
    metadata.update(
        fps=float(provider._dataset_cfg_for_seq(seq).mocap_fps / seq.stride),
        frame_ids=raw["frame_ids"].tolist(), camera_ids=cameras,
        extraction_hash="pending-background", distortion="undistorted", approved=False,
        source_sequence=f"{seq.source}:{seq.source_id}",
        dataset=seq.source,
        source_mocap_indices=indices,
        camera_frame_ids={c: provider._desired_frame_indices(
            provider._dataset_cfg_for_seq(seq), (seq.seq_id, start), c) for c in cameras},
        path=f"{identity}.npz",
    )
    clip = Clip(metadata, imgs, footprints, joints, raw["world2img"], raw["intrinsics"])
    clip.validate()
    apply_official_masks(provider, clip)
    return clip


def prepare(args) -> None:
    cfg = load_config(args.config)
    spec = read_json(Path(args.spec))
    if not spec.get("split_protocol") or not spec["clips"]:
        raise ValueError("A nonempty source spec and documented split protocol are required")
    provider = HumanPriorAlignedProvider(cfg, "train")
    sources = [dict(s, source_sequence=f"{provider._seqs[s['sequence']].source}:{provider._seqs[s['sequence']].source_id}")
               for s in spec["clips"]]
    validate_catalog(sources)
    validate_source_datasets(sources, cfg.data.composition_datasets)
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh cache directory; do not overwrite previous extraction")
    root.mkdir(parents=True, exist_ok=True)
    parameters = {"margin_mm": args.background_margin_mm, "min_count": args.min_background_count,
                  "pixel_threshold": args.pixel_threshold, "noise_scale": args.noise_scale}
    parameter_hash = hashlib.sha256(json.dumps(
        {"method": EXTRACTION_METHOD, "opencv": cv2.__version__, **parameters}, sort_keys=True
    ).encode()).hexdigest()
    groups: dict[tuple[str, str], list[dict]] = {}
    for source in sources:
        groups.setdefault((source["split"], source["capture_group"]), []).append(source)
    entries, backgrounds = [], []
    for group, group_sources in groups.items():
        temporary_paths = []
        group_entries = []
        accumulators = None
        reference = None
        observed: set[tuple[str, str, int]] = set()
        try:
            for source in group_sources:
                clip = _source_clip(provider, source, args.background_margin_mm)
                if reference is None:
                    reference = {"camera_ids": clip.metadata["camera_ids"], "fps": clip.metadata["fps"],
                                 "world2img": clip.world2img, "intrinsics": clip.intrinsics}
                    accumulators = [BackgroundMean((*clip.imgs.shape[-2:], 3)) for _ in range(clip.imgs.shape[1])]
                elif (
                    clip.metadata["camera_ids"] != reference["camera_ids"] or clip.metadata["fps"] != reference["fps"]
                    or not np.array_equal(clip.world2img, reference["world2img"])
                    or not np.array_equal(clip.intrinsics, reference["intrinsics"])
                ):
                    raise ValueError(f"Background group {group} contains incompatible cameras/world coordinates")
                if accumulators is None:
                    raise RuntimeError("Background accumulators were not initialized")
                for t in range(len(clip.imgs)):
                    for n, accumulator in enumerate(accumulators):
                        camera = clip.metadata["camera_ids"][n]
                        observation = (clip.metadata["source_sequence"], camera, clip.metadata["camera_frame_ids"][camera][t])
                        if observation in observed:
                            continue
                        observed.add(observation)
                        accumulator.update(clip.imgs[t, n].transpose(1, 2, 0), clip.masks[t, n])
                temporary = root / (source["id"] + ".raw.npz")
                temporary_paths.append(temporary)
                np.savez_compressed(temporary, imgs=clip.imgs, joints=clip.joints,
                                    world2img=clip.world2img, intrinsics=clip.intrinsics, masks=clip.masks)
                group_entries.append(clip.metadata)
            if accumulators is None:
                raise RuntimeError("No background observations")
            arrays = [accumulator.arrays() for accumulator in accumulators]
            means, stds, counts = (np.stack([array[i] for array in arrays]) for i in range(3))
            background_id = hashlib.sha256(json.dumps(group).encode()).hexdigest()[:16]
            background_path = root / (f"background-{background_id}.npz")
            np.savez_compressed(background_path, mean=means, std=stds, count=counts)
            background_hash = file_hash(background_path)
            backgrounds.append({"split": group[0], "capture_group": group[1], "path": background_path.name,
                                "sha256": background_hash, "sources": [s["id"] for s in group_sources]})
            for entry, temporary in zip(group_entries, temporary_paths):
                with np.load(temporary, allow_pickle=False) as data:
                    imgs, joints = data["imgs"], data["joints"]
                    projections, intrinsics = data["world2img"], data["intrinsics"]
                    masks = data["masks"].copy()
                for t in range(0 if entry.get("mask_method") else len(imgs)):
                    for n in range(imgs.shape[1]):
                        try:
                            masks[t, n] = foreground_mask(
                                imgs[t, n].transpose(1, 2, 0), joints[t], projections[t, n],
                                means[n], stds[n], counts[n], **parameters,
                            )
                        except ValueError as exc:
                            raise ValueError(f"{entry['id']} frame={t} camera={n}: {exc}") from exc
                entry["background_sha256"] = background_hash
                if not entry.get("mask_method"):
                    entry["extraction_hash"] = hashlib.sha256((parameter_hash + background_hash).encode()).hexdigest()
                Clip(entry, imgs, masks, joints, projections, intrinsics).validate()
                path = root / entry["path"]
                np.savez_compressed(path, imgs=imgs, masks=masks, joints=joints,
                                    world2img=projections, intrinsics=intrinsics)
                entry["sha256"] = file_hash(path)
                entries.append(entry)
        finally:
            close_mask_readers(provider)
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)
    write_json(root / "catalog.json", {
        "version": VERSION, "split_protocol": spec["split_protocol"], "clips": entries,
        "extraction_method": ("mixed" if any(e.get("mask_method") for e in entries) else EXTRACTION_METHOD),
        "mpi_mask_mode": cfg.data.composition_mpi_masks, "extraction_parameters": parameters,
        "backgrounds": backgrounds,
    })


def review(args) -> None:
    path = Path(args.manifest)
    document = read_json(path)
    if "recipes" in document and document["version"] != VERSION:
        raise ValueError("Regenerate and visually review recipes for the current compositor version")
    decisions = read_json(Path(args.review))
    if not decisions.get("reviewer") or not decisions.get("rubric"):
        raise ValueError("A named reviewer and documented visual quality rubric are required")
    accepted, rejected = set(decisions["approved"]), set(decisions["rejected"])
    if accepted & rejected or not (accepted | rejected):
        raise ValueError("Review decisions must be nonempty and disjoint")
    key = "recipes" if "recipes" in document else "clips"
    records = document[key]
    known = {r["id"] for r in records}
    if not (accepted | rejected) <= known:
        raise ValueError("Review references unknown IDs")
    rate = len(accepted) / len(accepted | rejected)
    rollout = decisions.get("approve_remaining", False)
    if rollout:
        entries = {e["id"]: e for e in document["clips"]}
        def stratum(record):
            source = entries[record["sources"][0]] if key == "recipes" else record
            return source["capture_group"], len(record.get("sources", [record["id"]]))
        reviewed_strata = {stratum(r) for r in records if r["id"] in accepted | rejected}
        if reviewed_strata != {stratum(r) for r in records}:
            raise ValueError("Pilot rollout requires review coverage of every capture-group/person-count stratum")
    for record in records:
        record["approved"] = record["id"] in accepted or (
            rollout and rate >= 0.95 and record["id"] not in rejected
        )
    document["review"] = {**decisions, "pass_rate": rate}
    document["quality_approved"] = rate >= 0.95
    write_json(path, document)
    if rate < 0.95:
        raise ValueError(f"Pilot review pass rate {rate:.1%} is below 95%; revise extraction/composition")


def generate(args) -> None:
    cfg = load_config(args.config)
    path = Path(args.catalog).resolve()
    catalog = read_json(path)
    validate_source_datasets(catalog["clips"], cfg.data.composition_datasets)
    if not catalog.get("quality_approved"):
        raise ValueError("Review and approve source caches before generation")
    output = Path(args.output).resolve()
    if output.parent != path.parent:
        raise ValueError("Store the manifest alongside its catalog/cache assets")
    if output.exists():
        raise FileExistsError("Do not overwrite a frozen recipe manifest")
    recipes = []
    reports = {}
    parameters = {"position_range": cfg.model.position_range, "radius_mm": args.radius_mm,
                  "depth_margin_mm": args.depth_margin_mm}
    for i, split in enumerate(("train", "val", "test")):
        requested = getattr(args, split + "_count")
        if not requested:
            continue
        generated, report = generate_recipes(
            catalog["clips"], path.parent, split, requested, args.people, args.seed + i,
            **parameters, max_attempts=args.max_attempts,
        )
        reports[split] = report
        recipes.extend(generated)
    write_json(output.with_suffix(".report.json"), reports)
    if not recipes or any(r["requested"] != r["achieved"] for r in reports.values()):
        raise RuntimeError("Generation did not reach requested counts; inspect the rejection report before revising scope")
    write_json(output, {"version": VERSION, "clips": catalog["clips"], "recipes": recipes,
                        "parameters": parameters, "reports": reports,
                        "split_protocol": catalog["split_protocol"], "quality_approved": False})


def preview(args) -> None:
    path = Path(args.manifest).resolve()
    manifest = read_json(path)
    if "recipes" in manifest and manifest["version"] != VERSION:
        raise ValueError("Regenerate and visually review recipes for the current compositor version")
    entries = validate_catalog(manifest["clips"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = manifest.get("recipes", manifest["clips"])
    if args.ids:
        wanted = set(args.ids)
        records = [r for r in records if r["id"] in wanted]
        if {r["id"] for r in records} != wanted:
            raise ValueError("Unknown preview IDs")
    for record in records[:args.limit]:
        ids = record.get("sources", [record["id"]])
        clips = [load_clip(path.parent, entries[i]) for i in ids]
        if "sources" in record:
            imgs, _, _ = compose(clips, **manifest["parameters"])
        else:
            imgs = clips[0].imgs.copy()
            # Highlight mask boundaries for source extraction review.
            for t in range(len(imgs)):
                for n in range(imgs.shape[1]):
                    edge = cv2.morphologyEx(clips[0].masks[t, n].astype(np.uint8),
                                            cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                    imgs[t, n, :, edge] = np.array([0, 255, 0], dtype=np.uint8)
        h, w = imgs.shape[-2:]
        writer = cv2.VideoWriter(str(output / (record["id"] + ".avi")),
                                 cv2.VideoWriter.fourcc(*"MJPG"), clips[0].metadata["fps"],
                                 (w * imgs.shape[1], h))
        if not writer.isOpened():
            raise RuntimeError("Could not open preview video writer")
        try:
            for frame in imgs:
                canvas = np.concatenate([image.transpose(1, 2, 0) for image in frame], axis=1)
                writer.write(canvas[..., ::-1].copy())
        finally:
            writer.release()


def audit(args) -> None:
    cfg = load_config(args.config)
    provider = HumanPriorAlignedProvider(cfg, "train")
    rows = []
    for seq in provider._seqs.values():
        cameras = provider._cam_defs_for_seq(seq)
        fingerprint = hashlib.sha256()
        for identity, definition in sorted(cameras.items()):
            fingerprint.update(str(identity).encode())
            for key in ("K", "R", "T", "dist"):
                fingerprint.update(np.asarray(definition.get(key, []), dtype=np.float64).tobytes())
        rows.append({"sequence": seq.seq_id, "dataset": seq.source, "source_id": seq.source_id,
                     "frames": seq.num_frames, "camera_ids": sorted(cameras),
                     "calibration_sha256": fingerprint.hexdigest(),
                     "fps": provider._dataset_cfg_for_seq(seq).mocap_fps / seq.stride})
    write_json(Path(args.output), {"sequences": rows,
               "candidate_calibration_groups": dict(Counter(r["calibration_sha256"] for r in rows)),
               "note": "Matching fingerprints are candidates, not proof of capture/world compatibility. "
                       "Assign verified capture_group and protocol-specific split_group in the source spec."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ap = sub.add_parser("audit")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.set_defaults(func=audit)
    ap = sub.add_parser("prepare")
    for name in ("config", "spec", "output"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--background-margin-mm", type=float, default=200.)
    ap.add_argument("--min-background-count", type=int, default=3)
    ap.add_argument("--pixel-threshold", type=float, default=20.)
    ap.add_argument("--noise-scale", type=float, default=3.)
    ap.set_defaults(func=prepare)
    ap = sub.add_parser("review")
    for name in ("manifest", "review"):
        ap.add_argument("--" + name, required=True)
    ap.set_defaults(func=review)
    ap = sub.add_parser("generate")
    for name in ("config", "catalog", "output"):
        ap.add_argument("--" + name, required=True)
    for split in ("train", "val", "test"):
        ap.add_argument("--" + split + "-count", type=int, default=0)
    ap.add_argument("--people", type=int, nargs="+", default=[2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius-mm", type=float, default=150.0)
    ap.add_argument("--depth-margin-mm", type=float, default=150.0)
    ap.add_argument("--max-attempts", type=int, default=10000)
    ap.set_defaults(func=generate)
    ap = sub.add_parser("preview")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--limit", type=int, default=20)
    ap.set_defaults(func=preview)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
