"""Append-only experiment record for training runs.

Every ``train()`` launch (fresh or resumed) writes one immutable record under
``<output_dir>/launches/``:

* ``NNN_<utc>.json`` -- the review document: the resolved config, a
  per-topic summary of the settings that shape the experiment (streaming,
  data mixture, re-identification, approximate-GT noise, tracking), content
  hashes of every input the config only names by path (golden/gym cache and
  its upstream label-set and calibration hashes, composition manifest,
  backbone, external re-ID network), what the sampler actually built (lanes,
  clips, windows per component), the code version and the environment;
* ``NNN_config_source.yaml`` -- the YAML file as written (comments kept);
* ``NNN_git.diff`` -- uncommitted changes to tracked files, when any.

Records are created exclusively and never rewritten.  ``experiment.json`` at
the run root indexes all launches and lists, per launch, which resolved
config keys changed relative to the previous launch; it is derived from the
records and can always be rebuilt from them.  ``config.yaml`` stays the
latest resolved config (other tools read it).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import Config, config_to_dict

FORMAT_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]

# Config keys grouped by the experiment topic they control (all keys stay in "config").
SECTIONS: dict[str, list[str]] = {
    "streaming": [
        "data.seq_len", "data.stride", "data.batch_size",
        "data.stream_domain_components", "data.stream_domain_chunks", "data.stream_domain_window_share",
        "data.stream_aux_chunks", "data.stream_aux_window_share",
        "data.camera_dropout_prob", "data.camera_dropout_min_views",
        "model.memory_topk", "model.memory_len",
    ],
    "mixture": [
        "data.provider", "data.gym_pseudo3d_root", "data.gym_sample_fraction",
        "data.gym_mix_aligned_dirs", "data.gym_mix_aligned_fraction", "data.train_window_cache",
        "data.composition_manifest", "data.composition_original_fraction", "data.composition_datasets",
        "data.composition_mpi_masks", "data.human_prior_aligned_dirs", "data.val_fraction", "data.split_seed",
    ],
    "tracking": [
        "data.track_max_gap", "model.track_queries", "model.max_tracks", "model.track_drop_prob",
        "model.track_fp_prob", "model.track_ref_noise", "model.track_birth_score", "model.track_keep_score",
        "model.track_max_inactive", "model.track_nms_mm",
    ],
    "reid": [
        "model.reid_dims", "model.reid_appearance", "model.reid_ema", "model.reid_loss_weight", "model.reid_query_grad_scale",
        "model.reid_temperature", "model.reid_bank_recent", "model.reid_bank_anchors", "model.reid_bank_stride",
        "model.reid_external_model", "model.reid_external_stride",
    ],
    "reid_inference_only": [
        # Not trained: tune with tools/sweep_reid (writes <checkpoint>.reid_sweep.jsonl).
        "model.reid_wait_frames", "model.reid_threshold", "model.reid_gallery_size",
    ],
    "gt_noise": [
        "loss.gt_noise_tolerance", "loss.gt_sigma_ref_mm", "loss.root_weight",
        "loss.reproj_weight", "loss.reproj_tolerance_px", "loss.reproj_scale_px",
        "loss.reproj_triangulated_weight", "model.cost_root", "model.cost_root_labeled",
    ],
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def _file_record(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    resolved = _repo_path(path)
    record: dict[str, Any] = {"path": str(path), "resolved": str(resolved.resolve())}
    if resolved.is_file():
        stat = resolved.stat()
        record.update(sha256=file_sha256(resolved), bytes=stat.st_size,
                      modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
    else:
        record["missing"] = True
    return record


def _get(tree: dict, dotted: str) -> Any:
    value: Any = tree
    for part in dotted.split("."):
        value = value[part]
    return value


def _flatten(tree: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def _guard(fn, *args, **kwargs) -> Any:
    """Provenance must never stop a training run: record failures instead."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on the machine
        return {"error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Sections                                                                    #
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True, timeout=30).stdout


def code_record() -> tuple[dict, str]:
    """(git summary, diff of tracked files).  Untracked files are listed, not copied."""
    commit = _git("rev-parse", "HEAD").strip()
    diff = _git("diff", "HEAD", "--binary")
    untracked = [line for line in _git("ls-files", "--others", "--exclude-standard").splitlines()
                 if line.endswith(".py") or line.endswith(".yaml")]
    return {"commit": commit, "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty": bool(diff.strip()), "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "untracked_source_files": untracked}, diff


def environment_record() -> dict:
    record: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                              "hostname": socket.gethostname()}
    try:
        import torch
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    for module in ("ultralytics", "numpy", "mmcv"):
        try:
            record[module] = __import__(module).__version__
        except Exception:
            pass
    return record


def gym_cache_record(root: str) -> dict:
    """Golden / pseudo-3D cache: index hash, upstream label/calibration hashes, per-split targets."""
    path = _repo_path(root)
    record: dict[str, Any] = {"path": root, "resolved": str(path.resolve())}
    index_path = path / "index.json"
    if not index_path.is_file():
        record["missing_index"] = True
        return record
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record["index"] = _file_record(str(index_path))
    for key in ("format_version", "dataset", "fps", "image_hw", "camera_ids", "camera_names", "world",
                "world_offset_mm", "sources", "config", "coverage", "rejected_outliers"):
        if key in index:
            record[key] = index[key]
    splits = {}
    for split_dir in sorted(p for p in path.iterdir() if p.is_dir() and p.name in ("train", "val", "test")):
        files = {}
        for npz in sorted(split_dir.rglob("*.npz")):
            files[str(npz.relative_to(path)).replace("\\", "/")] = file_sha256(npz)
        splits[split_dir.name] = files
    record["targets_sha256"] = splits
    return record


def composition_record(manifest: Optional[str]) -> Optional[dict]:
    if not manifest:
        return None
    record = _file_record(manifest)
    path = _repo_path(manifest)
    if record and path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        counts: dict[str, dict[str, int]] = {}
        for kind in ("recipes", "clips"):
            for entry in document.get(kind, []):
                if entry.get("approved") is True:
                    split = counts.setdefault(entry.get("split", "?"), {})
                    split[kind] = split.get(kind, 0) + 1
        record.update(version=document.get("version"), quality_approved=document.get("quality_approved"),
                      parameters=document.get("parameters"), approved_counts=counts)
    return record


def weights_record(cfg: Config) -> dict:
    record: dict[str, Any] = {}
    if cfg.model.backbone:
        from ..models.backbones import BACKBONES
        entry = BACKBONES.get(cfg.model.backbone)
        # The registry hash is verified when the model is built (models/backbones.py).
        record["backbone"] = ({"name": cfg.model.backbone, "path": entry.path, "sha256_verified": entry.sha256,
                               "description": entry.description} if entry else {"name": cfg.model.backbone})
    else:
        record["backbone"] = {"yolo_model_name": cfg.model.yolo_model_name,
                              "file": _file_record(cfg.model.yolo_model_name)}
    if cfg.model.reid_external_model:
        external = _file_record(cfg.model.reid_external_model)
        sidecar = _repo_path(cfg.model.reid_external_model).with_suffix(".json")
        if external is not None and sidecar.is_file():
            external["export_metadata"] = json.loads(sidecar.read_text(encoding="utf-8"))
        record["reid_external_model"] = external
    if cfg.train.resume:
        record["resume_checkpoint"] = {k: v for k, v in (_file_record(cfg.train.resume) or {}).items()
                                       if k != "sha256"}  # 0.5 GB: identified by size/mtime and the step
    return record


def sampler_record(loaders: dict) -> dict:
    """What the data pipeline actually built: components, clips, windows, lane plan, eval sizes."""
    record: dict[str, Any] = {}
    train = loaders.get("train")
    sampler = getattr(train, "batch_sampler", None)
    if sampler is not None and hasattr(sampler, "lane_owner"):
        components = []
        for component, policy in zip(sampler.components, sampler.policies):
            windows = sum(len(g) for g in component.groups)
            components.append({
                "name": component.name, "fraction": component.fraction, "clips": len(component.groups),
                "windows": windows, "longest_clip_windows": max((len(g) for g in component.groups), default=0),
                "chunk_windows": policy.windows, "chunk_window_share": policy.window_share,
                "streaming": policy.streaming,
            })
        names = [c.name for c in sampler.components]
        record["train"] = {
            "kind": "streaming", "batches_per_epoch": len(sampler), "components": components,
            "lane_owner": [None if o is None else names[o] for o in sampler.lane_owner],
            "shared_lane_target": (None if sampler.shared_target is None
                                   else dict(zip(names, map(float, sampler.shared_target)))),
        }
    elif train is not None:
        dataset = train.dataset
        record["train"] = {"kind": type(dataset).__name__, "samples": _guard(len, dataset)}
        for part in ("mocap", "gym", "aligned"):
            sub = getattr(dataset, part, None)
            if sub is not None:
                record["train"][f"{part}_samples"] = _guard(len, sub)
    evaluation = {}
    for name, loader in loaders.items():
        if name == "train":
            continue
        batches = _guard(len, loader)
        evaluation[name] = {"batches": batches, "dataset": type(loader.dataset).__name__}
    record["eval"] = evaluation
    return record


# --------------------------------------------------------------------------- #
# Writer                                                                      #
# --------------------------------------------------------------------------- #
def _next_index(folder: Path) -> int:
    used = [int(p.name.split("_", 1)[0]) for p in folder.glob("*_*.json") if p.name.split("_", 1)[0].isdigit()]
    return max(used, default=0) + 1


def write_launch_record(cfg: Config, loaders: dict, *, config_path: Optional[str] = None,
                        argv: Optional[list[str]] = None, resumed_optimizer_steps: int = 0) -> Path:
    """Write this launch's immutable record and refresh ``experiment.json``; returns the record path."""
    out = Path(cfg.train.output_dir)
    folder = out / "launches"
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    resolved = config_to_dict(cfg)

    result = _guard(code_record)
    code, diff = (result, "") if isinstance(result, dict) else result

    record: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_utc": now.isoformat(),
        "run": out.name,
        "output_dir": str(out.resolve()),
        "launch": {"argv": list(argv if argv is not None else sys.argv), "cwd": os.getcwd(),
                   "config_path": config_path, "resume": cfg.train.resume,
                   "resumed_at_optimizer_step": resumed_optimizer_steps},
        "sections": {name: {key: _get(resolved, key) for key in keys} for name, keys in SECTIONS.items()},
        "inputs": {
            "gym_cache": (_guard(gym_cache_record, cfg.data.gym_pseudo3d_root)
                          if cfg.data.provider == "gym_mocap_mix" else None),
            "composition_manifest": _guard(composition_record, cfg.data.composition_manifest),
            "aligned_dirs": {"gym_mix": list(cfg.data.gym_mix_aligned_dirs),
                             "human_prior": list(cfg.data.human_prior_aligned_dirs)},
            "weights": _guard(weights_record, cfg),
        },
        "sampler": _guard(sampler_record, loaders),
        "code": code,
        "environment": _guard(environment_record),
        "config": resolved,
    }

    # Exclusive create: a record is never overwritten, even by a concurrent launch.
    while True:
        index = _next_index(folder)
        stem = f"{index:03d}"
        path = folder / f"{stem}_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        try:
            with open(path, "x", encoding="utf-8") as stream:
                record["launch"]["index"] = index
                if config_path and Path(config_path).is_file():
                    source = folder / f"{stem}_config_source.yaml"
                    shutil.copyfile(config_path, source)
                    record["launch"]["config_source_copy"] = source.name
                    record["launch"]["config_source_sha256"] = file_sha256(source)
                if diff:
                    (folder / f"{stem}_git.diff").write_text(diff, encoding="utf-8")
                    record["code"] = {**record["code"], "diff_file": f"{stem}_git.diff"}
                json.dump(record, stream, indent=2, default=str)
            break
        except FileExistsError:
            continue
    rebuild_experiment_index(out)
    return path


def rebuild_experiment_index(out: Path) -> dict:
    """Derive ``experiment.json`` (launch list + config changes between launches) from the records."""
    folder = Path(out) / "launches"
    records = sorted(p for p in folder.glob("[0-9][0-9][0-9]_*.json"))
    launches, previous = [], None
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        flat = _flatten(record.get("config", {}))
        changed = None
        if previous is not None:
            changed = {key: {"before": previous.get(key), "after": flat.get(key)}
                       for key in sorted(set(previous) | set(flat))
                       if previous.get(key) != flat.get(key) and key != "train.resume"}
        code = record.get("code") or {}
        launches.append({"record": path.name, "created_utc": record.get("created_utc"),
                         "resume": record.get("launch", {}).get("resume"),
                         "resumed_at_optimizer_step": record.get("launch", {}).get("resumed_at_optimizer_step"),
                         "commit": code.get("commit"), "dirty": code.get("dirty"),
                         "config_changes_vs_previous": changed})
        previous = flat
    index = {"format_version": FORMAT_VERSION, "run": Path(out).name, "launches": launches,
             "note": "Derived from launches/*.json (immutable); rebuild with "
                     "unified_perception.engine.run_manifest.rebuild_experiment_index."}
    tmp = Path(out) / "experiment.json.tmp"
    tmp.write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, Path(out) / "experiment.json")
    return index
