"""Prepare, train, and evaluate a source-isolated local augmentation experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import cv2
import numpy as np

from ..config import load_config, save_config
from ..data.backends import (
    HumanPriorAlignedProvider, _decode_video_frames_torchcodec, _video_path_for_camera,
)
from ..data.background import EXTRACTION_METHOD, BackgroundMean, foreground_mask
from ..data.composition import VERSION, compose, file_hash, generate_recipes, load_clip, read_json, save_clip, verify_clip_assets, validate_catalog
from ..data.mpi_masks import capture_group, close_mask_readers, mask_paths
from .compose import _source_clip, write_json as _write_json

def write_json(path: Path, value: object) -> None:
    staging = path.with_name(path.name + ".writing")
    _write_json(staging, value)
    staging.replace(path)


PROTOCOL = (
    "Subject-disjoint: H36M subjects 1,5,6,7,8 train / 9 val / 11 test; "
    "MPI subjects S1-S6 train / S7 val / S8 test. All segments of a source "
    "recording remain together. Background templates use their own sequence only. "
    "Official MPI masks and cross-segment pooling are opt-in; pooling requires "
    "the same recording, camera IDs, FPS and exact calibration. "
    "MOYO excluded. Fresh model heads; common pretrained YOLO backbone. "
    "Single-person evaluation uses unchanged cached source RGB, not composites. "
    "Current extraction quality accepted provisionally by user; not a certified "
    "95-percent visual-quality benchmark. Rejected masks are excluded and reported."
)


def source_split(dataset: str, source_id: str) -> str:
    if dataset == "h36m":
        subject = int(source_id.split("_")[1])
        if subject not in (1, 5, 6, 7, 8, 9, 11):
            raise ValueError(f"Unexpected H36M subject: {subject}")
        return {9: "val", 11: "test"}.get(subject, "train")
    if dataset == "mpi":
        subject = int(source_id.split("_")[0][1:])
        if subject not in range(1, 9):
            raise ValueError(f"Unexpected MPI subject: {subject}")
        return {7: "val", 8: "test"}.get(subject, "train")
    raise ValueError(f"Excluded dataset: {dataset}")


class CachedDecoderProvider(HumanPriorAlignedProvider):
    """Keep decoder handles only for the current sequence."""

    def __init__(self, cfg):
        super().__init__(cfg, "train")
        self.decoders = {}

    def _load_window_images(self, ds_cfg, sample, candidate_cam_ids, num_cameras):
        from torchcodec.decoders import VideoDecoder  # type: ignore[reportPrivateImportUsage]

        images = {}
        for camera in candidate_cam_ids:
            path = _video_path_for_camera(ds_cfg, camera)
            if path not in self.decoders:
                self.decoders[path] = VideoDecoder(path, num_ffmpeg_threads=1)
            indices = self._desired_frame_indices(ds_cfg, sample, camera)
            images[camera], _ = _decode_video_frames_torchcodec(self.decoders[path], path, indices)
        if len(images) != num_cameras:
            raise ValueError("Incomplete camera selection")
        return list(images), images


def prepare_sequence(provider, sequence: str, root: Path, windows: int, *,
                     camera_ids: list[str] | None = None, cross_segment: bool = False,
                     storage: str = "npz") -> dict:
    seq = provider._seqs[sequence]
    split = source_split(seq.source, seq.source_id)
    available = provider._cam_defs_for_seq(seq)
    cameras = camera_ids or (
        ["1", "2", "3", "4"] if seq.source == "h36m" else ["0", "2", "4", "6"])[:provider.num_cameras]
    if (len(cameras) != provider.num_cameras or not set(cameras) <= set(available)
            or seq.num_frames < provider.seq_len):
        reason = (f"Insufficient views or frames: {sequence}; required cameras={cameras}, "
                  f"available={sorted(available)}, frames={seq.num_frames}, required={provider.seq_len}")
        print(f"[sequence rejected] {reason}", flush=True)
        provider.decoders.clear()
        return dict(sequence=sequence, split=split, clips=[], failures=[dict(sequence=sequence, errors=[reason])],
                    requested=windows, decoded=0, accepted=0)
    starts = np.unique(np.linspace(0, seq.num_frames - provider.seq_len, windows).astype(int))
    accumulators = [BackgroundMean((*provider.out_hw, 3)) for _ in cameras]
    clips = []
    source_failures = []
    observed = set()
    tag = preparation_tag(sequence, camera_ids)
    for i, start in enumerate(starts):
        source = dict(
            id=f"{seq.source}-{tag}-{i:03d}", sequence=sequence,
            subject=f"{seq.source}:{seq.source_id.split('_act_')[0].split('_Seq')[0]}",
            split=split, split_group=f"{seq.source}:{seq.source_id}",
            capture_group=sequence, geometry_verified=True, camera_ids=cameras, start=int(start),
        )
        try:
            clip = _source_clip(provider, source, 200.)
            bounds = np.asarray(provider.cfg.model.position_range)
            if np.any(clip.joints < bounds[:3]) or np.any(clip.joints > bounds[3:]):
                raise ValueError("outside_capture_volume")
            if clip.metadata.get("mask_method") and cross_segment:
                # Alignment corrections can differ between segments of one recording.
                clip.metadata["capture_group"] = capture_group(clip, seq.source_id)
        except ValueError as exc:
            print(f"[source rejected] {source['id']}: {exc}", flush=True)
            source_failures.append(dict(id=source["id"], errors=[str(exc)]))
            continue
        clips.append(clip)
        for t in range(len(clip.imgs)):
            for n, camera in enumerate(cameras):
                key = camera, clip.metadata["camera_frame_ids"][camera][t]
                if key not in observed:
                    observed.add(key)
                    accumulators[n].update(clip.imgs[t, n].transpose(1, 2, 0), clip.masks[t, n])
    arrays = [a.arrays() for a in accumulators]
    means, stds, counts = (np.stack([a[i] for a in arrays]) for i in range(3))
    bg_path = root / f"{tag}-background.npz"
    np.savez_compressed(bg_path, mean=means, std=stds, count=counts)
    bg_hash = file_hash(bg_path)
    entries, failures = [], source_failures
    for clip in clips:
        errors = []
        for t in range(0 if clip.metadata.get("mask_method") else len(clip.imgs)):
            for n, camera in enumerate(cameras):
                try:
                    clip.masks[t, n] = foreground_mask(
                        clip.imgs[t, n].transpose(1, 2, 0), clip.joints[t],
                        clip.world2img[t, n], means[n], stds[n], counts[n],
                    )
                except ValueError as exc:
                    errors.append(f"frame {t} camera {camera}: {exc}")
        if errors:
            failures.append(dict(id=clip.metadata["id"], errors=errors))
            print(f"[mask rejected] {clip.metadata['id']}: {errors[0]} ({len(errors)} errors)", flush=True)
            continue
        if not clip.metadata.get("mask_method"):
            clip.metadata["extraction_hash"] = f"{EXTRACTION_METHOD}:{cv2.__version__}:{bg_hash}"
        clip.metadata.update(
            background_sha256=bg_hash, approved=True,
            approval_basis="User authorized provisional experiment despite imperfect masks",
        )
        clip.validate()
        save_clip(root, clip, storage=storage)
        entries.append(clip.metadata)
    provider.decoders.clear()
    close_mask_readers(provider)
    return dict(sequence=sequence, split=split, clips=entries, failures=failures,
                background=bg_path.name, background_sha256=bg_hash,
                requested=len(starts), decoded=len(clips), accepted=len(entries))


def preparation_tag(sequence: str, cameras: list[str] | None = None) -> str:
    identity = sequence if cameras is None else sequence + ":cameras=" + ",".join(cameras)
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


_WORKER_PROVIDER = None
_WORKER_DLL = None


def _initialize_fast_worker():
    global _WORKER_DLL
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if sys.platform == "win32":
        ffmpeg = os.environ["FFMPEG_SHARED_BIN"]
        _WORKER_DLL = os.add_dll_directory(ffmpeg)
        os.environ["PATH"] = ffmpeg + os.pathsep + os.environ.get("PATH", "")


def _initialize_preparation_worker(cfg):
    global _WORKER_PROVIDER
    _initialize_fast_worker()
    _WORKER_PROVIDER = CachedDecoderProvider(cfg)


def _prepare_worker(sequence, root, budget, cameras, cross_segment, storage):
    started = time.perf_counter()
    result = prepare_sequence(_WORKER_PROVIDER, sequence, root, budget,
                              camera_ids=cameras, cross_segment=cross_segment, storage=storage)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def prepare_corpus(cfg, root: Path, windows: int, status, *, mpi_windows: int = 0,
                   mpi_camera_sets: list[list[str]] | None = None, cross_segment: bool = False,
                   workers: int = 1, storage: str = "npz") -> dict:
    provider = CachedDecoderProvider(cfg)
    sequences = [(s, cameras) for s in sorted(provider._seqs)
                 for cameras in ((mpi_camera_sets or [None])
                                 if provider._seqs[s].source == "mpi" else [None])]
    entries, results = [], []
    asset_hashes = {}
    pending = []
    completed = {}
    for i, (sequence, cameras) in enumerate(sequences):
        budget = mpi_windows if provider._seqs[sequence].source == "mpi" and mpi_windows else windows
        tag = preparation_tag(sequence, cameras)
        report_path = root / f"{tag}-preparation.json"
        settings = dict(windows=budget, cameras=cameras, cross_segment=cross_segment,
                        mpi_masks=cfg.data.composition_mpi_masks)
        if storage != "npz":
            settings["storage"] = storage
        seq = provider._seqs[sequence]
        if seq.source == "mpi" and cfg.data.composition_mpi_masks != "background":
            selected = cameras or ["0", "2", "4", "6"][:provider.num_cameras]
            assets = {}
            for camera in selected:
                for path in mask_paths(Path(cfg.data.human_prior_mpi_dataset_path), seq.source_id, camera):
                    if path not in asset_hashes:
                        asset_hashes[path] = file_hash(path) if path.is_file() else None
                    assets[str(path)] = asset_hashes[path]
            settings["mask_assets"] = assets
        if report_path.exists():
            result = read_json(report_path)
            if result.get("settings") != settings:
                raise ValueError("Preparation settings differ; use a fresh output directory")
            for entry in result["clips"]:
                verify_clip_assets(root, entry)
            completed[i] = result
        else:
            pending.append((i, sequence, cameras, budget, report_path, settings))
    def record(item, result):
        i, sequence, cameras, budget, report_path, settings = item
        result["settings"] = settings
        write_json(report_path, result)
        completed[i] = result
        status("preparing", sequence=sequence, completed_sequences=len(completed),
               total_sequences=len(sequences), workers=workers)
        print(f"[prepare {len(completed)}/{len(sequences)}] {sequence}: "
              f"{result['accepted']}/{result['requested']} accepted", flush=True)
    if workers > 1 and pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_preparation_worker,
                                 initargs=(cfg,)) as pool:
            futures = {pool.submit(_prepare_worker, item[1], root, item[3], item[2],
                                   cross_segment, storage): item for item in pending}
            for future in as_completed(futures):
                record(futures[future], future.result())
    else:
        for item in pending:
            started = time.perf_counter()
            result = prepare_sequence(provider, item[1], root, item[3],
                                      camera_ids=item[2], cross_segment=cross_segment, storage=storage)
            result["elapsed_seconds"] = time.perf_counter() - started
            record(item, result)
    for i in range(len(sequences)):
        result = completed[i]
        entries.extend(result["clips"])
        results.append({k: v for k, v in result.items() if k != "clips"})
    validate_catalog(entries)
    catalog = dict(version=VERSION, clips=entries, split_protocol=PROTOCOL,
                   quality_approved=True, provisional_quality=True, preparation=results,
                   accepted_by_split_dataset=dict(Counter(
                       f"{e['split']}/{e['dataset']}" for e in entries
                   )))
    write_json(root / "catalog.json", catalog)
    for split in ("train", "val", "test"):
        for dataset in ("h36m", "mpi"):
            if sum(e["split"] == split and e["dataset"] == dataset for e in entries) < 2:
                raise RuntimeError(f"Insufficient accepted {dataset}/{split} sources; inspect catalog.json")
    return catalog


def _generate_group(root, cfg, group_index, group, sources, multiplier, attempts):
    started = time.perf_counter()
    split = sources[0]["split"]
    target = max(1, round(len(sources) * multiplier / 7))
    tag = hashlib.sha256(group.encode()).hexdigest()[:16]
    clip_cache = {e["id"]: load_clip(root, e, verify_hash=False) for e in sources}
    group_recipes, group_reports = [], {}
    for people in range(2, 9):
        generated, report = generate_recipes(
            sources, root, split, target, [people],
            cfg.train.seed + 100 * group_index + people,
            position_range=cfg.model.position_range, radius_mm=150., depth_margin_mm=150.,
            max_attempts=max(target, attempts), clip_cache=clip_cache,
        )
        for recipe in generated:
            recipe.update(id=f"{split}-{tag}-p{people}-{recipe['id']}", approved=True,
                          approval_basis="User authorized provisional experiment")
        group_recipes.extend(generated)
        group_reports[f"people_{people}"] = report
        print(f"[recipes] {group} people={people}: {len(generated)}/{target}; "
              f"rejections={report['rejections']}", flush=True)
    return dict(recipes=group_recipes, reports=group_reports,
                elapsed_seconds=time.perf_counter() - started)


def freeze_recipes(catalog, root: Path, cfg, multiplier: int, attempts: int, status, *,
                   workers: int = 1) -> Path:
    path = root / "experiment.json"
    if path.exists():
        document = read_json(path)
        if document["version"] != VERSION:
            raise ValueError("Frozen compositor version mismatch")
        validate_catalog(document["clips"])
        return path
    parameters = dict(position_range=cfg.model.position_range, radius_mm=150., depth_margin_mm=150.)
    recipes, reports = [], {}
    groups = {}
    for entry in catalog["clips"]:
        groups.setdefault(entry["capture_group"], []).append(entry)
    completed, pending = {}, []
    for group_index, (group, sources) in enumerate(sorted(groups.items())):
        tag = hashlib.sha256(group.encode()).hexdigest()[:16]
        group_path = root / f"{tag}-recipes.json"
        if group_path.exists():
            completed[group_index] = read_json(group_path)
        else:
            pending.append((group_index, group, sources, group_path))
    def record(item, document):
        group_index, group, sources, group_path = item
        write_json(group_path, document)
        completed[group_index] = document
        status("generating", capture_group=group, completed_groups=len(completed),
               total_groups=len(groups), workers=workers)
    if workers > 1 and pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_fast_worker) as pool:
            futures = {pool.submit(_generate_group, root, cfg, item[0], item[1], item[2],
                                   multiplier, attempts): item for item in pending}
            for future in as_completed(futures):
                record(futures[future], future.result())
    else:
        for item in pending:
            record(item, _generate_group(root, cfg, item[0], item[1], item[2], multiplier, attempts))
    for group_index, (group, sources) in enumerate(sorted(groups.items())):
        document = completed[group_index]
        recipes.extend(document["recipes"])
        reports[group] = document["reports"]
    write_json(root / "generation-progress.json", reports)
    by_id = {e["id"]: e for e in catalog["clips"]}
    for split in ("train", "val", "test"):
        for dataset in ("h36m", "mpi"):
            if not any(r["split"] == split and by_id[r["sources"][0]]["dataset"] == dataset for r in recipes):
                raise RuntimeError(f"No feasible {dataset}/{split} composites; inspect generation-progress.json")
    write_json(path, dict(
        version=VERSION, clips=catalog["clips"], recipes=recipes, parameters=parameters,
        reports=reports, split_protocol=PROTOCOL, quality_approved=True, provisional_quality=True,
        achieved_counts=dict(Counter(f"{r['split']}/people_{len(r['sources'])}" for r in recipes)),
        requested_multiplier=multiplier,
        achieved_multiplier=len(recipes) / len(catalog["clips"]),
        quality_caveat="Partial counts are explicitly reported, not replaced by single-person samples.",
    ))
    return path


def _materialize_group(root, entries, recipes, parameters):
    started = time.perf_counter()
    clips = {entry["id"]: load_clip(root, entry, verify_hash=False) for entry in entries}
    records = {}
    for recipe in recipes:
        identity = recipe["id"]
        if not identity or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identity):
            raise ValueError("Unsafe materialized recipe ID")
        selected = [clips[identity] for identity in recipe["sources"]]
        images, visibility, order = compose(selected, **parameters)
        if hashlib.sha256(order.tobytes()).hexdigest() != recipe["order_sha256"]:
            raise ValueError("Materialized recipe order differs from frozen geometry")
        path = root / "renders" / (recipe["id"] + ".npz")
        image_path = path.with_suffix(".npy")
        np.save(image_path, images, allow_pickle=False)
        np.savez(path, joints=np.stack([clip.joints for clip in selected], axis=1),
                 world2img=selected[0].world2img, intrinsics=selected[0].intrinsics,
                 visibility=visibility, order=order)
        assets = [dict(path=str(asset.relative_to(root)), sha256=file_hash(asset),
                       size=asset.stat().st_size, mtime_ns=asset.stat().st_mtime_ns)
                  for asset in (image_path, path)]
        records[recipe["id"]] = dict(path=str(path.relative_to(root)), sha256=assets[1]["sha256"],
            imgs_path=str(image_path.relative_to(root)), assets=assets,
            sources_sha256=hashlib.sha256(json.dumps(recipe["sources"]).encode()).hexdigest(),
            parameters_sha256=hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest())
    return dict(records=records, elapsed_seconds=time.perf_counter() - started)


def materialize_recipes(manifest: Path, status, *, workers: int = 1) -> None:
    """Freeze deterministic lossless RGB once; resume at capture-group boundaries."""
    root = manifest.parent
    document = read_json(manifest)
    (root / "renders").mkdir(exist_ok=True)
    entries = {entry["id"]: entry for entry in document["clips"]}
    groups = {}
    for recipe in document["recipes"]:
        group = entries[recipe["sources"][0]]["capture_group"]
        groups.setdefault(group, []).append(recipe)
    pending, completed = [], {}
    for group, recipes in sorted(groups.items()):
        tag = hashlib.sha256(group.encode()).hexdigest()[:16]
        report = root / "renders" / f"{tag}-rendered.json"
        if report.exists():
            result = read_json(report)
            if set(result["records"]) != {recipe["id"] for recipe in recipes}:
                raise ValueError("Rendered group differs from frozen recipes")
            for recipe in recipes:
                cached = result["records"][recipe["id"]]
                verify_clip_assets(root, cached)
                if (cached["sources_sha256"] != hashlib.sha256(json.dumps(recipe["sources"]).encode()).hexdigest()
                        or cached["parameters_sha256"] != hashlib.sha256(
                            json.dumps(document["parameters"], sort_keys=True).encode()).hexdigest()):
                    raise ValueError("Rendered sources or parameters differ; use a fresh output directory")
            completed[group] = result
        else:
            identities = {identity for recipe in recipes for identity in recipe["sources"]}
            pending.append((group, recipes, [entries[i] for i in sorted(identities)], report))
    if pending:
        shape = load_clip(root, pending[0][2][0], verify_hash=False).imgs.shape
        required = sum(len(item[1]) for item in pending) * (int(np.prod(shape)) + 65536)
        if shutil.disk_usage(root).free < required * 1.05:
            raise RuntimeError("Insufficient disk space for lossless frozen recipe RGB")
    def record(item, result):
        group, recipes, sources, report = item
        write_json(report, result)
        completed[group] = result
        status("materializing", completed_groups=len(completed), total_groups=len(groups), workers=workers)
        print(f"[materialize {len(completed)}/{len(groups)}] {group}: {len(recipes)} recipes", flush=True)
    if workers > 1 and pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_fast_worker) as pool:
            futures = {pool.submit(_materialize_group, root, item[2], item[1], document["parameters"]): item
                       for item in pending}
            for future in as_completed(futures):
                record(futures[future], future.result())
    else:
        for item in pending:
            record(item, _materialize_group(root, item[2], item[1], document["parameters"]))
    for recipes in groups.values():
        for recipe in recipes:
            group = entries[recipe["sources"][0]]["capture_group"]
            recipe["pixel_cache"] = completed[group]["records"][recipe["id"]]
    document["pixel_cache_format"] = "lossless-npy-v1"
    write_json(manifest, document)


def run_command(command: list[str], log: Path) -> None:
    print(f"[run] {subprocess.list2cmdline(command)} -> {log}", flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{datetime.now(timezone.utc).isoformat()} {subprocess.list2cmdline(command)}\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def evaluate_arm(cfg, checkpoint: Path, output: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ..data import MultiViewPoseDataset, collate_batch
    from ..engine.evaluate import run_eval
    from ..models import UnifiedPerceptionModel

    device = torch.device(cfg.train.device)
    model = UnifiedPerceptionModel(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    results = {}
    for split in ("val", "test"):
        for name, fraction in (("original", 1.), ("synthetic", 0.)):
            cfg.data.composition_original_fraction = fraction
            dataset = MultiViewPoseDataset(cfg, split=split)
            loader = DataLoader(dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
                                collate_fn=collate_batch)
            metrics = run_eval(model, loader, cfg, device)
            results[f"{split}/{name}"] = {
                k: float(v) if np.isfinite(v) else None for k, v in metrics.items()
            }
            write_json(output, dict(checkpoint=str(checkpoint), optimizer_steps=state["optimizer_steps"],
                                    metrics=results))
    return results


EVAL_SETS = {"val/original", "val/synthetic", "test/original", "test/synthetic"}


def experiment_stages(baseline_steps: int) -> list[tuple[str, str, int, float, str]]:
    """Return (result name, arm directory, optimizer steps, original fraction, metrics file)."""
    final = 3 * baseline_steps
    return [
        ("baseline", "baseline", baseline_steps, 1., "metrics.json"),
        (f"augmented_{baseline_steps}", "augmented", baseline_steps, .25, f"metrics-{baseline_steps}.json"),
        ("augmented", "augmented", final, .25, "metrics.json"),
    ]


def _gates(reference: dict, candidate: dict) -> dict:
    def mpjpe_ok(a, b):
        a, b = a["mpjpe_mm"], b["mpjpe_mm"]
        return None if a is None or b is None else b <= 1.05 * a

    single, single_ref = candidate["test/original"], reference["test/original"]
    synthetic, synthetic_ref = candidate["test/synthetic"], reference["test/synthetic"]
    return dict(
        single_person_f1=single["f1"] >= single_ref["f1"] - .02,
        single_person_mpjpe=mpjpe_ok(single_ref, single),
        synthetic_f1=synthetic["f1"] >= synthetic_ref["f1"] + .05,
        synthetic_mpjpe=mpjpe_ok(synthetic_ref, synthetic),
        single_person_f1_delta=single["f1"] - single_ref["f1"],
        synthetic_f1_delta=synthetic["f1"] - synthetic_ref["f1"],
    )


def acceptance(results: dict) -> dict:
    """Primary gates compare equal-step arms; the longer augmented run is a secondary signal."""
    matched = next(name for name in results if name.startswith("augmented_"))
    return dict(
        step_matched=dict(candidate=matched, **_gates(results["baseline"], results[matched])),
        longer_augmented=dict(candidate="augmented", **_gates(results["baseline"], results["augmented"])),
        provisional_data_quality=True, real_multi_person_claim=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=r"configs\composition_experiment.yaml")
    parser.add_argument("--output", default=r"runs\composition_experiment")
    parser.add_argument("--windows-per-sequence", type=int, default=24)
    parser.add_argument("--mpi-windows-per-sequence", type=int, default=0,
                        help="MPI-only window budget; 0 uses --windows-per-sequence")
    parser.add_argument("--mpi-camera-sets", nargs="+",
                        help="Comma-separated camera IDs per set, e.g. 0,2,4 1,5,6")
    parser.add_argument("--mpi-cross-segment", action="store_true",
                        help="Pool official-mask MPI clips only when recording and exact calibration match")
    parser.add_argument("--mpi-masks", choices=("background", "prefer", "required"),
                        help="Override data.composition_mpi_masks for preparation")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare source caches and stop; never generate recipes, train, or evaluate")
    parser.add_argument("--build-only", action="store_true",
                        help="Prepare and freeze recipes, then stop without training or evaluation")
    parser.add_argument("--workers", type=int, default=1, help="Preparation processes")
    parser.add_argument("--materialize", action="store_true",
                        help="Freeze lossless recipe RGB for fast training loads (additional disk space)")
    parser.add_argument("--recipe-workers", type=int, default=1, help="Recipe generation processes")
    parser.add_argument("--storage", choices=("npz", "npy"), default="npz",
                        help="Lossless source format; npy uses lazy memory-mapped pixels")
    parser.add_argument("--recipe-multiplier", type=int, default=10)
    parser.add_argument("--recipe-attempts", type=int, default=512,
                        help="Attempt budget per capture group and person count")
    parser.add_argument("--baseline-steps", type=int, default=10000)
    args = parser.parse_args()
    if min(args.windows_per_sequence, args.recipe_multiplier, args.recipe_attempts, args.baseline_steps, args.workers, args.recipe_workers) < 1:
        parser.error("All budgets must be positive")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def status(stage, **details):
        write_json(root / "status.json", dict(
            stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **details,
        ))

    cfg = load_config(args.config)
    if args.mpi_masks:
        cfg.data.composition_mpi_masks = args.mpi_masks
    if args.mpi_windows_per_sequence < 0:
        parser.error("--mpi-windows-per-sequence cannot be negative")
    camera_sets = [item.split(",") for item in args.mpi_camera_sets] if args.mpi_camera_sets else None
    if camera_sets and (
        len({tuple(cameras) for cameras in camera_sets}) != len(camera_sets)
        or any(len(cameras) != cfg.data.num_cameras or len(set(cameras)) != len(cameras)
               or any(not c.isdecimal() or int(c) not in range(14) for c in cameras)
               for cameras in camera_sets)
    ):
        parser.error("MPI camera sets must be unique sets of num_cameras distinct IDs in 0..13")
    ffmpeg_handle = None
    if sys.platform == "win32":
        ffmpeg = os.environ.get("FFMPEG_SHARED_BIN")
        if not ffmpeg:
            raise ValueError("Set FFMPEG_SHARED_BIN to the shared FFmpeg bin directory")
        ffmpeg_handle = os.add_dll_directory(ffmpeg)
        os.environ["PATH"] = ffmpeg + os.pathsep + os.environ.get("PATH", "")
    spec = dict(arguments=vars(args), config_sha256=file_hash(Path(args.config)), protocol=PROTOCOL,
                compositor_version=VERSION, extraction_method=EXTRACTION_METHOD,
                implementation_sha256={
                    name: file_hash(Path(__file__).parents[1] / name) for name in (
                        r"data\composition.py", r"data\background.py", r"tools\composition_experiment.py",
                        r"engine\train.py", r"models\losses.py",
                        r"data\mpi_masks.py", r"tools\compose.py", r"data\transforms.py",
                    )
                })
    specification = root / "experiment-spec.json"
    if specification.exists() and read_json(specification) != spec:
        raise ValueError("Experiment configuration differs; use a fresh output directory")
    write_json(specification, spec)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    save_config(cfg, str(root / "base-config.yaml"))
    try:
        build_started = time.perf_counter()
        catalog = prepare_corpus(
            cfg, cache, args.windows_per_sequence, status,
            mpi_windows=args.mpi_windows_per_sequence, mpi_camera_sets=camera_sets,
            cross_segment=args.mpi_cross_segment,
            **({"workers": args.workers, "storage": args.storage}
               if args.workers != 1 or args.storage != "npz" else {}),
        )
        if args.prepare_only:
            status("prepared", catalog=str(cache / "catalog.json"), training_started=False)
            return
        preparation_seconds = time.perf_counter() - build_started
        recipes_started = time.perf_counter()
        manifest = freeze_recipes(catalog, cache, cfg, args.recipe_multiplier, args.recipe_attempts, status,
                                  **({"workers": args.recipe_workers} if args.recipe_workers != 1 else {}))
        write_json(root / "build-timings.json", dict(preparation_seconds=preparation_seconds,
                   recipes_seconds=time.perf_counter() - recipes_started,
                   total_seconds=time.perf_counter() - build_started))
        if args.materialize:
            started = time.perf_counter()
            materialize_recipes(manifest, status, workers=args.recipe_workers)
            timings = read_json(root / "build-timings.json")
            timings["materialization_seconds"] = time.perf_counter() - started
            timings["total_seconds"] = time.perf_counter() - build_started
            write_json(root / "build-timings.json", timings)
        if args.build_only:
            status("built", manifest=str(manifest), training_started=False)
            return
        results = {}
        for name, directory, steps, fraction, metrics_name in experiment_stages(args.baseline_steps):
            arm = root / directory
            arm.mkdir(exist_ok=True)
            cfg = load_config(args.config)
            cfg.data.composition_manifest = str(manifest)
            cfg.data.composition_original_fraction = fraction
            cfg.train.max_steps, cfg.train.output_dir = steps, str(arm)
            config = arm / f"experiment-config-{steps}.yaml"
            save_config(cfg, str(config))
            checkpoint = arm / "last.pt"
            snapshot = arm / f"step-{steps}.pt"
            import torch

            if not snapshot.exists():
                completed = torch.load(checkpoint, map_location="cpu", weights_only=False)[
                    "optimizer_steps"] if checkpoint.exists() else 0
                if completed > steps:
                    raise ValueError(f"{checkpoint} passed {steps} steps before its snapshot was saved")
                if completed < steps:
                    status("training", arm=name, optimizer_budget=steps)
                    command = [sys.executable, "-u", "-m", "unified_perception.tools.train",
                               "--config", str(config)]
                    if checkpoint.exists():
                        command.extend(["--resume", str(checkpoint)])
                    run_command(command, arm / "console.log")
                shutil.copy2(checkpoint, snapshot)
            metrics_path = arm / metrics_name
            previous = read_json(metrics_path) if metrics_path.exists() else {}
            if previous.get("optimizer_steps") == steps and set(previous.get("metrics", {})) == EVAL_SETS:
                results[name] = previous["metrics"]
            else:
                status("evaluating", arm=name)
                # Use fixed-step checkpoints, not test-selected or unmatched best-MPJPE models.
                results[name] = evaluate_arm(cfg, snapshot, metrics_path)
            torch.cuda.empty_cache()
            write_json(root / "results.json", dict(protocol=PROTOCOL, arms=results))
        write_json(root / "results.json", dict(protocol=PROTOCOL, arms=results, acceptance=acceptance(results)))
        status("completed", results=str(root / "results.json"))
    except Exception:
        status("failed", error=traceback.format_exc())
        raise
    finally:
        if ffmpeg_handle is not None:
            ffmpeg_handle.close()


if __name__ == "__main__":
    main()
