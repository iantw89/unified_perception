"""Refresh a training report from last.pt, with full gym evaluation and fixed examples."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import tempfile
import warnings
import webbrowser

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent

from ..config import Config, load_config
from ..constants import ROOT_JOINTS, SKELETON_EDGES
from ..data import MultiViewPoseDataset, collate_batch
from ..data.gym_pseudo3d import GymPseudo3DDataset
from ..engine.evaluate import run_eval
from ..engine.metrics import MetricsAccumulator
from ..models import UnifiedPerceptionModel
from ..engine.train_viz import _draw_skeleton, _project
from .visualize_predictions import _draw_dashed_skeleton, _label


DEFAULT_RUN = Path("runs") / "gym_sixcam_yolov8x_c7"
COLORS = [tuple(int(c * 255) for c in plt.get_cmap("tab20")(i)[:3][::-1])
          for i in range(20)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                        help="training output directory (default: %(default)s)")
    parser.add_argument("--checkpoint", type=Path, default=Path("last.pt"),
                        help="filename relative to run-dir, or absolute path (default: last.pt)")
    parser.add_argument("--config", type=Path, help="config override (default: run-dir/config.yaml)")
    parser.add_argument("--output", type=Path, help="report directory (default: run-dir/visualizations)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto uses CUDA when available; inference batch size is always 1")
    parser.add_argument("--examples", type=int, default=4, help="evenly spaced gym examples (default: 4)")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--gym-only", action="store_true", help="skip composed mocap examples")
    parser.add_argument("--no-animations", action="store_true", help="save still montages only")
    parser.add_argument("--open", action="store_true", help="open the completed HTML report")
    return parser


def load_snapshot(path: Path) -> tuple[dict, str]:
    """Copy one opened checkpoint version; atomic trainer replacements cannot mix versions."""
    with path.open("rb") as source, tempfile.TemporaryDirectory() as temporary:
        modified = datetime.fromtimestamp(os.fstat(source.fileno()).st_mtime, timezone.utc).isoformat()
        snapshot = Path(temporary) / "checkpoint.pt"
        with snapshot.open("wb") as target:
            shutil.copyfileobj(source, target)
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
    return state, modified


def read_scalars(run: Path) -> dict[str, dict[int, ScalarEvent]]:
    tags: dict[str, dict[int, ScalarEvent]] = defaultdict(dict)
    for file in sorted(run.glob("events.out.tfevents.*")):
        events = EventAccumulator(str(file), size_guidance={"scalars": 0}).Reload()
        for tag in events.Tags()["scalars"]:
            for event in events.Scalars(tag):
                old = tags[tag].get(event.step)
                if old is None or old.wall_time < event.wall_time:
                    tags[tag][event.step] = event
    return dict(tags)


def even_indices(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("Dataset length and example count must be positive")
    return np.linspace(0, length - 1, min(count, length), dtype=int).tolist()


def compose_indices(dataset: MultiViewPoseDataset) -> list[int]:
    composed = dataset.composed
    if composed is None:
        raise ValueError("Mocap examples require a composition manifest")
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, recipe in enumerate(composed.recipes):
        source = composed.entries[recipe["sources"][0]]["dataset"]
        strata[(source, len(recipe["sources"]))].append(index)
    indices = []
    for source in sorted({key[0] for key in strata}):
        counts = sorted(key[1] for key in strata if key[0] == source)
        for count in sorted({counts[0], counts[-1]}):
            candidates = strata[(source, count)]
            indices.append(candidates[len(candidates) // 2])
    return indices


def finite_metrics(metrics: dict) -> dict:
    return {key: value if math.isfinite(value) else None for key, value in metrics.items()}


def metric_text(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def sample_tensor(sample, key: str) -> torch.Tensor:
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Sample field {key!r} must be a Tensor, got {type(value).__name__}")
    return value


def save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training(tags: dict[str, dict[int, ScalarEvent]], run: Path, output: Path,
                  checkpoint_step: int) -> tuple[dict, list[str]]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], ("train/loss", "train/loss_kpt", "train/loss_root", "train/loss_cls"),
         "Training losses (EMA 0.9)", "Loss (log scale)"),
        (axes[0, 1], ("train/loss_ema",), "Trainer loss EMA", "Loss"),
        (axes[1, 0], ("train/grad_norm",), "Gradient norm before clipping", "Norm"),
        (axes[1, 1], ("sys/gpu_alloc_gb", "sys/gpu_rsvd_gb"), "Training GPU memory", "GB"),
    )
    for ax, keys, title, ylabel in panels:
        plotted = False
        for key in keys:
            series = sorted(tags.get(key, {}).items())
            if not series:
                continue
            values = [event.value for _, event in series]
            if key in panels[0][1]:
                previous = values[0]
                smoothed = []
                for value in values:
                    previous = previous * 0.9 + value * 0.1
                    smoothed.append(previous)
                values = smoothed
            ax.plot([step for step, _ in series], values, label=key.split("/")[-1], linewidth=1.3)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No saved scalars for this panel", ha="center", transform=ax.transAxes)
        ax.axvline(checkpoint_step, color="#333", linestyle="--", label="Checkpoint")
        ax.set(title=title, xlabel="Logged global step", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    if tags.get("train/loss"):
        axes[0, 0].set_yscale("log")
    if tags.get("train/grad_norm"):
        axes[1, 0].set_yscale("log")
    loss = tags.get("train/loss", {})
    latest = max(loss) if loss else None
    if not loss:
        warnings.warn(f"No saved training loss scalars in {run}", stacklevel=2)
    fig.suptitle(f"{run.name} | checkpoint {checkpoint_step:,} | latest logged step {latest}",
                 fontsize=15)
    save_figure(fig, output, "learning_curves.png")
    assets = ["learning_curves.png"]
    history = {}
    for split in ("val", "val_gym", "val_aligned"):
        values = tags.get(f"{split}/mpjpe_mm", {})
        if values:
            step = max(values)
            history[split] = {"global_step": step, "metrics": finite_metrics({
                key.split("/", 1)[1]: series[step].value
                for key, series in tags.items()
                if key.startswith(split + "/") and key.count("/") == 1 and step in series
            })}
    if history:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, key, title in zip(axes, ("mpjpe_mm", "f1", "root_error_mm"),
                                  ("Validation pose error (mm)", "Detection F1", "Root error (mm)")):
            for split, label in (("val", "Mocap"), ("val_gym", "Gym"), ("val_aligned", "MOYO")):
                series = sorted(tags.get(f"{split}/{key}", {}).items())
                if series:
                    ax.plot([step for step, _ in series], [event.value for _, event in series],
                            ".-", label=label)
            ax.axvline(checkpoint_step, color="#333", linestyle="--", label="Checkpoint")
            ax.set(title=title, xlabel="Logged global step")
            ax.legend(fontsize=8)
        axes[1].set_ylim(0, 1)
        fig.suptitle("Saved epoch-end validation history (not fresh checkpoint evaluation)")
        save_figure(fig, output, "validation_history.png")
        assets.append("validation_history.png")
    return {
        "latest_logged_step": latest,
        "loss_at_latest_log": metric_text(loss[latest].value, 5) if latest is not None else None,
        "epoch_validation_logged": bool(history),
        "latest_epoch_validation": history,
        "event_files": [file.name for file in sorted(run.glob("events.out.tfevents.*"))],
    }, assets


def plot_validation(metrics: dict, cfg: Config, windows: int, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, keys, title, ylabel in (
        (axes[0], ("mpjpe_mm", "root_error_mm"), "Pose / root error", "mm"),
        (axes[1], ("precision", "recall", "f1"), "Person detection", "Fraction"),
        (axes[2], tuple(f"pck@{int(t)}mm" for t in cfg.eval.pck_thresholds_mm),
         "Valid-joint PCK", "Fraction"),
    ):
        raw = [metrics[key] for key in keys]
        values = [0 if value is None else value for value in raw]
        bars = ax.bar(keys, values, color="#2878b5")
        ax.bar_label(bars, labels=[metric_text(value, 1 if ylabel == "mm" else 3) for value in raw],
                     padding=4)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=15)
        ax.set_ylim(0, max(1, max(values, default=0) * 1.2) if ylabel == "mm" else 1.08)
    fig.suptitle(f"Fresh checkpoint gym evaluation | {windows} windows / "
                 f"{windows * cfg.data.seq_len} window-frames")
    save_figure(fig, output, "gym_validation.png")


def floor_tile(gt, masks, roots, root_valid, pred, matches, ids, position_range):
    size = 320
    tile = np.full((size, size, 3), 26, np.uint8)
    lower = np.asarray(position_range[:2], dtype=float) / 1000
    upper = np.asarray(position_range[3:5], dtype=float) / 1000
    centre = (lower + upper) / 2
    half = max(upper - lower) / 2 + 0.5

    def pixels(xyz):
        xy = (np.asarray(xyz)[..., :2] / 1000 - centre) / (2 * half)
        xy = xy * [1, -1] + 0.5
        return np.round(12 + xy * (size - 24)).astype(int)

    for meter in np.arange(np.ceil((centre[0] - half) / 2) * 2, centre[0] + half, 2):
        x = pixels([meter * 1000, 0, 0])[0]
        cv2.line(tile, (x, 0), (x, size), (48, 48, 48), 1)
    for meter in np.arange(np.ceil((centre[1] - half) / 2) * 2, centre[1] + half, 2):
        y = pixels([0, meter * 1000, 0])[1]
        cv2.line(tile, (0, y), (size, y), (48, 48, 48), 1)
    for g, person in enumerate(gt):
        color = COLORS[int(ids[g]) % len(COLORS)]
        _draw_dashed_skeleton(tile, pixels(person).astype(np.float32), masks[g], color)
        if root_valid[g]:
            point = tuple(pixels(roots[g]))
            cv2.circle(tile, point, 6, color, 1, cv2.LINE_AA)
            _label(tile, str(int(ids[g])), (point[0] + 7, point[1]), color, 0.32)
    for p, person in enumerate(pred):
        color = COLORS[int(ids[matches[p]]) % len(COLORS)] if p in matches else (40, 40, 255)
        _draw_skeleton(tile, pixels(person), np.ones(len(person), bool), color, 1, 1)
    _label(tile, "Floor: x,y | 2 m grid | circles=GT roots", (5, 16), (255, 255, 255), 0.37)
    return tile


def render_clip(sample, scores, poses, cfg: Config, cameras, title: str, threshold: float = 0.3):
    acc = MetricsAccumulator(cfg.eval.pck_thresholds_mm, cfg.eval.match_threshold_mm)
    frames, details = [], []
    for t, frame_id in enumerate(sample["frame_ids"].tolist()):
        valid = sample["gt_valid"][t].numpy()
        gt = sample["gt_keypoints"][t].numpy()[valid]
        supervised = bool(sample["gt_keypoint_frame_valid"][t])
        masks = sample["gt_keypoint_valid"][t].numpy()[valid] & supervised
        roots = sample["gt_root"][t].numpy()[valid]
        root_valid = sample["gt_root_valid"][t].numpy()[valid]
        ids = sample.get("instance_ids")
        ids = ids[t].numpy()[valid] if ids is not None else np.arange(len(gt))
        pred = poses[t, scores[t] > threshold]
        associations = acc.update_frame(
            pred, gt, keypoints_supervised=supervised,
            pred_root_mm=pred[:, list(ROOT_JOINTS)].mean(1),
            gt_root_mm=roots, gt_root_valid=root_valid, gt_keypoint_valid=masks,
        )
        matches = dict(associations)
        tiles = []
        for c, camera in enumerate(cameras):
            tile = cv2.cvtColor(sample["imgs"][t, c].numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            projection = sample["world2img"][t, c].numpy()
            for g, person in enumerate(gt):
                color = COLORS[int(ids[g]) % len(COLORS)]
                uv, ok = _project(projection, person)
                _draw_dashed_skeleton(tile, uv, ok & masks[g], color)
                if root_valid[g]:
                    uv_root, ok_root = _project(projection, roots[g:g+1])
                    height, width = tile.shape[:2]
                    if ok_root[0] and np.all((uv_root[0] >= 0) & (uv_root[0] < [width, height])):
                        point = tuple(np.round(uv_root[0]).astype(int))
                        cv2.circle(tile, point, 7, color, 1, cv2.LINE_AA)
                        _label(tile, f"GT{ids[g]}", (point[0] + 8, point[1]), color, 0.32)
            for p, person in enumerate(pred):
                color = COLORS[int(ids[matches[p]]) % len(COLORS)] if p in matches else (40, 40, 255)
                uv, ok = _project(projection, person)
                _draw_skeleton(tile, uv, ok, color, 2, 1)
            _label(tile, f"Camera {camera} | frame {frame_id}", (6, 16), (255, 255, 255))
            tiles.append(tile)
        floor = floor_tile(gt, masks, roots, root_valid, pred, matches, ids, cfg.model.position_range)
        tiles.append(cv2.resize(floor, (tiles[0].shape[1], tiles[0].shape[0])))
        row = np.concatenate(tiles, axis=1)
        bar = np.full((50, row.shape[1], 3), 18, np.uint8)
        _label(bar, title, (6, 18), (255, 255, 255), 0.46)
        label = "labeled joints" if supervised else "roots only; no GT pose labels"
        _label(bar, f"F{frame_id} | {label} | GT {len(gt)} | pred {len(pred)} | "
               f"matched {len(matches)} | FP {len(pred)-len(matches)} | missed {len(gt)-len(matches)}",
               (6, 39), (255, 255, 255), 0.43)
        frames.append(np.vstack([bar, row]))
        details.append({"frame": frame_id, "pose_labeled": supervised, "gt": len(gt),
                        "predicted": len(pred), "matched": len(matches),
                        "false_positives": len(pred)-len(matches), "missed": len(gt)-len(matches)})
    return frames, details, acc.compute()


def plot_poses(samples, scores, poses, cfg: Config, title: str, output: Path, threshold: float = 0.3) -> None:
    columns = min(2, len(samples))
    rows = math.ceil(len(samples) / columns)
    fig = plt.figure(figsize=(7 * columns, 5 * rows), constrained_layout=True)
    lower, upper = np.asarray(cfg.model.position_range[:3]) / 1000, np.asarray(cfg.model.position_range[3:]) / 1000
    for number, (sample, clip_scores, clip_poses) in enumerate(zip(samples, scores, poses)):
        ax = fig.add_subplot(rows, columns, number + 1, projection="3d")
        labeled = torch.nonzero(sample["gt_keypoint_frame_valid"]).flatten()
        t = int(labeled[-1]) if len(labeled) else 0
        valid = sample["gt_valid"][t].numpy()
        gt = sample["gt_keypoints"][t].numpy()[valid] / 1000
        masks = sample["gt_keypoint_valid"][t].numpy()[valid] & bool(sample["gt_keypoint_frame_valid"][t])
        pred = clip_poses[t, clip_scores[t] > threshold] / 1000
        for people, joint_masks, color, style, label in (
            (gt, masks, "#2878b5", "--", "GT (valid joints only)"),
            (pred, np.ones(pred.shape[:2], bool), "#e1812c", "-", "Predictions"),
        ):
            first = True
            for person, mask in zip(people, joint_masks):
                ax.scatter(*person[mask].T, color=color, s=4)
                for a, b in SKELETON_EDGES:
                    if mask[a] and mask[b]:
                        ax.plot(*person[[a, b]].T, color=color, linestyle=style, linewidth=1,
                                label=label if first else None)
                        first = False
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
               xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
               title=f"Frame {int(sample['frame_ids'][t])} | {sample['seq'].split('/')[-1]}")
        ax.set_box_aspect((upper[0] - lower[0], upper[1] - lower[1], 6))
        ax.view_init(elev=24, azim=-65)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=15)
    save_figure(fig, output, "gym_3d_poses.png")


def write_report(output: Path, summary: dict, assets: list[str]) -> None:
    checkpoint = summary["checkpoint"]
    metadata = summary["checkpoint_metadata"]
    steps = metadata.get("optimizer_steps")
    target = summary["target_optimizer_steps"]
    progress = f"{steps:,} / {target:,} updates ({steps / target:.1%})" if steps is not None and target else f"step {steps}"
    epoch = metadata.get("epoch")
    complete = metadata.get("epoch_complete")
    epoch_text = f"epoch {epoch}" + (" complete" if complete is True else " incomplete" if complete is False else "")
    logs = summary["training_logs"]
    latest = logs["latest_logged_step"]
    history_rows = []
    for split, record in logs["latest_epoch_validation"].items():
        metrics = record["metrics"]
        history_rows.append(f"<tr><td>{html.escape(split)}</td><td>{record['global_step']}</td>"
                            f"<td>{metric_text(metrics.get('mpjpe_mm'), 1)}</td>"
                            f"<td>{metric_text(metrics.get('f1'))}</td></tr>")
    history = (
        "<p>No epoch-end validation metrics were saved yet.</p>" if not history_rows else
        "<p>Latest saved epoch-end metrics below are historical, not evaluations of this checkpoint.</p>"
        "<table><tr><th>Split</th><th>Logged step</th><th>MPJPE (mm)</th><th>F1</th></tr>"
        + "".join(history_rows) + "</table>"
    )
    plots = "".join(f"<img src='{name}'>" for name in assets)
    metric_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{metric_text(value)}</td></tr>"
        for key, value in summary["gym_validation"].items() if "/" not in key
    )
    cards = []
    for example in summary["examples"]:
        metrics = example["metrics"]
        image = example["animation"] or example["file"]
        cards.append(f"<section><h3>{example['group']} | {html.escape(example['sequence'])}</h3>"
                     f"<p>Example F1 {metric_text(metrics['f1'])}; "
                     f"MPJPE {metric_text(metrics['mpjpe_mm'], 1)} mm; "
                     f"root error {metric_text(metrics['root_error_mm'], 1)} mm</p>"
                     f"<a href='{example['file']}'>Full-resolution still montage</a>"
                     f"<br><img loading='lazy' src='{image}'></section>")
    config = summary["data_shape"]
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Training checkpoint report</title>"
        "<style>body{max-width:1600px;margin:32px auto;padding:0 20px;font-family:system-ui;"
        "background:#141821;color:#e5e9f0}img{max-width:100%;margin:16px 0}a{color:#80bfff}"
        "td,th{padding:8px;border:1px solid #555}table{border-collapse:collapse}"
        "section{margin:32px 0;border-top:1px solid #555}</style></head><body>"
        f"<h1>{html.escape(Path(summary['run']).name)} | checkpoint report</h1>"
        f"<p><b>{html.escape(Path(checkpoint).name)}: {progress}; {epoch_text}.</b> "
        f"Latest saved training loss log: {latest if latest is not None else 'unavailable'}. "
        "Predictions use only the snapshotted checkpoint, never later unsaved weights.</p>"
        f"<p>Generated {summary['generated_utc']}; checkpoint modified "
        f"{metadata['mtime_utc']}. Image size {config['height']} x {config['width']}; "
        f"{config['cameras']} cameras; {config['frames']} frames; "
        f"inference device {summary['device']}. Eager FP32 evaluation, batch size 1. "
        "This tool does not pause, resume, or modify training.</p>"
        f"<h2>Saved training history</h2>{plots}{history}"
        "<h2>Fresh held-out gym evaluation</h2><img src='gym_validation.png'>"
        f"<p>All {summary['gym_validation_windows']} validation windows; "
        f"{summary['gym_validation_window_frames']} window-frames. Overlaps count repeated source "
        "frames, as in training validation. MPJPE/PCK use matched valid labeled joints only; "
        "missed people are reflected by F1, not MPJPE. Unlabeled frames use rough hip roots "
        "for matching. Gym pseudo-GT is sparse, not dense motion-capture ground truth. "
        f"Score &gt;{summary['score_threshold']}; Hungarian match gate "
        f"{summary['match_threshold_mm']} mm.</p>"
        f"<table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>"
        "<h2>3D gym poses</h2><img src='gym_3d_poses.png'>"
        "<h2>Prediction examples</h2><p>Gym: evenly spaced validation indices. "
        "Mocap (when enabled): middle recipe at smallest/largest person count per source. "
        "No quality-based selection. Example metrics are not full mocap validation; "
        "MOYO is only represented by saved training validation history.</p>"
        "<p>Dashed lines: valid GT joints; circles: GT hip roots. Solid lines: predictions "
        "colored by matched person; red: false positives. Unlabeled joints are never drawn "
        "as GT. Mocap camera numbers are view slots. Animations, when enabled, play at 4 fps.</p>"
        + "".join(cards) +
        "<h2>Refresh</h2><p>From the repository root (pause GPU training first if needed):</p>"
        "<pre>" + html.escape(summary["refresh_command"]) + "</pre>"
        "<p><a href='summary.json'>Machine-readable metrics and snapshot provenance</a></p>"
        "</body></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")


@torch.inference_mode()
def _generate_report(args: argparse.Namespace, output: Path) -> None:
    run = args.run_dir.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else run / args.checkpoint
    checkpoint = checkpoint.resolve()
    config = args.config.resolve() if args.config else run / "config.yaml"
    cfg = load_config(str(config))
    if not args.gym_only and not cfg.data.composition_manifest:
        raise ValueError("This config has no composition manifest; use --gym-only")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    device = torch.device(device_name)
    state, modified = load_snapshot(checkpoint)
    metadata = {key: state[key] for key in ("epoch", "epoch_complete", "optimizer_steps", "global_step") if key in state}
    metadata["mtime_utc"] = modified
    step = metadata.get("global_step", metadata.get("optimizer_steps", 0))
    print(f"Snapshot: {checkpoint} | optimizer step {metadata.get('optimizer_steps')} | {device}", flush=True)
    model = UnifiedPerceptionModel(cfg).to(device).eval()
    model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in state["model"].items()})
    del state
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    logs, assets = plot_training(read_scalars(run), run, output, step)
    gym = GymPseudo3DDataset(cfg, "val")
    loader = DataLoader(gym, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)
    metrics = finite_metrics(run_eval(model, loader, cfg, device, score_threshold=args.score_threshold))
    plot_validation(metrics, cfg, len(gym), output)
    print(f"Full gym validation: MPJPE {metric_text(metrics['mpjpe_mm'], 1)} mm; "
          f"F1 {metric_text(metrics['f1'])}", flush=True)
    selections: list[tuple[str, GymPseudo3DDataset | MultiViewPoseDataset, list[int]]] = [
        ("gym", gym, even_indices(len(gym), args.examples)),
    ]
    if not args.gym_only:
        composed = MultiViewPoseDataset(cfg, "val")
        selections.append(("mocap", composed, compose_indices(composed)))
    mean = torch.tensor(cfg.data.pixel_mean, device=device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(cfg.data.pixel_std, device=device).view(1, 1, 1, 3, 1, 1)
    examples, gym_samples, gym_scores, gym_poses = [], [], [], []
    for group, dataset, indices in selections:
        for number, index in enumerate(indices):
            sample = dataset[index]
            sample.setdefault("frame_ids", torch.arange(cfg.data.seq_len) + int(sample["start"]))
            images = (sample_tensor(sample, "imgs")[None].to(device, torch.float32) - mean) / std
            cls, kpts, _, _ = model(images, sample_tensor(sample, "world2img")[None].to(device),
                                    sample_tensor(sample, "intrinsics")[None].to(device))
            scores = cls[-1, 0, ..., 0].sigmoid().cpu().numpy()
            poses = model.denormalize_keypoints(kpts[-1])[0].cpu().numpy()
            cameras = ([gym.provider.camera_ids[c] for c in gym.samples[index][2]]
                       if group == "gym" else list(range(sample_tensor(sample, "imgs").shape[1])))
            title = f"{checkpoint.name} | step {step} | {group} validation | " \
                    "Dashed=GT joints; circles=GT roots; solid=prediction; red=false positive"
            frames, details, clip_metrics = render_clip(sample, scores, poses, cfg, cameras, title,
                                                        args.score_threshold)
            still = f"{group}_example_{number+1:02d}.png"
            selected = {0, cfg.data.seq_len // 2, cfg.data.seq_len - 1}
            if group == "gym":
                selected.update(torch.nonzero(sample_tensor(sample, "gt_keypoint_frame_valid")).flatten().tolist())
            if not cv2.imwrite(str(output / still), np.vstack([frames[t] for t in sorted(selected)])):
                raise OSError(f"Failed to save {output / still}")
            gif = None
            if not args.no_animations:
                gif = f"{group}_example_{number+1:02d}.gif"
                animation = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
                animation[0].save(output / gif, save_all=True, append_images=animation[1:],
                                  duration=250, loop=0)
            examples.append({"group": group, "dataset_index": index, "sequence": sample["seq"],
                             "cameras": cameras, "file": still, "animation": gif, "frames": details,
                             "metrics": finite_metrics(clip_metrics)})
            if group == "gym":
                gym_samples.append(sample)
                gym_scores.append(scores)
                gym_poses.append(poses)
            print(f"{still}: F1 {metric_text(clip_metrics['f1'])}", flush=True)
    plot_poses(gym_samples, gym_scores, gym_poses, cfg, f"Checkpoint step {step:,} | z-up gym poses",
               output, args.score_threshold)
    refresh = ["python", "-m", "unified_perception.tools.training_report",
               "--run-dir", run.as_posix(), "--checkpoint", checkpoint.as_posix(),
               "--examples", str(args.examples), "--score-threshold", str(args.score_threshold),
               "--device", args.device]
    if args.config:
        refresh.extend(["--config", config.as_posix()])
    if args.output:
        refresh.extend(["--output", args.output.resolve().as_posix()])
    if args.gym_only:
        refresh.append("--gym-only")
    if args.no_animations:
        refresh.append("--no-animations")
    summary = {
        "run": str(run), "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_metadata": metadata,
        "config": str(config), "device": str(device), "training_logs": logs,
        "target_optimizer_steps": cfg.train.max_steps,
        "data_shape": {"height": cfg.data.image_height, "width": cfg.data.image_width,
                       "cameras": cfg.data.num_cameras, "frames": cfg.data.seq_len},
        "gym_validation_windows": len(gym),
        "gym_validation_window_frames": len(gym) * cfg.data.seq_len,
        "gym_validation": metrics, "score_threshold": args.score_threshold,
        "match_threshold_mm": cfg.eval.match_threshold_mm, "examples": examples,
        "refresh_command": shlex.join(refresh),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_report(output, summary, assets)


def generate_report(args: argparse.Namespace) -> Path:
    output = args.output.resolve() if args.output else args.run_dir.resolve() / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    # A failed refresh must not overwrite assets belonging to the previous report.
    with tempfile.TemporaryDirectory(prefix=".report-", dir=output) as temporary:
        staged = Path(temporary)
        _generate_report(args, staged)
        for file in sorted(staged.iterdir(), key=lambda path: path.name == "index.html"):
            file.replace(output / file.name)
    report = output / "index.html"
    print(f"Report: {report}", flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.examples < 1:
        parser.error("--examples must be positive")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    report = generate_report(args)
    if args.open and not webbrowser.open(report.as_uri()):
        warnings.warn(f"Could not open browser; open {report} manually", stacklevel=2)


if __name__ == "__main__":
    main()
