"""Render multi-person prediction examples from a checkpoint on composed clips.

Each example shows selected frames for every camera plus a top-down view.
Predictions above the score threshold are matched to ground truth with the
evaluator's Hungarian matching, so colors agree with reported metrics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import random
import shutil
import tempfile

import cv2
import numpy as np
import torch

from ..config import load_config
from ..constants import SKELETON_EDGES
from ..data import MultiViewPoseDataset
from ..engine.metrics import MetricsAccumulator
from ..engine.train_viz import _draw_skeleton, _project
from ..models import UnifiedPerceptionModel

# BGR, one per ground-truth person.
PALETTE = [(255, 160, 0), (0, 200, 255), (200, 80, 255), (80, 255, 80),
           (255, 255, 0), (0, 120, 255), (255, 100, 180), (160, 255, 200)]
FALSE_POSITIVE = (0, 0, 255)
UNMATCHED_QUERY = (110, 110, 110)
TILE_TEXT = (255, 255, 255)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    # The checkpoint may be rewritten by a running trainer; read a private copy.
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / path.name
        shutil.copy2(path, copy)
        return torch.load(copy, map_location=device, weights_only=False)


def choose_examples(dataset, per_count: int, seed: int) -> list[int]:
    """Sample up to ``per_count`` recipes for every (source dataset, person count)."""
    composed = dataset.composed
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, recipe in enumerate(composed.recipes):
        source = composed.entries[recipe["sources"][0]]
        strata[(source["dataset"], len(recipe["sources"]))].append(index)
    rng = random.Random(seed)
    chosen = []
    for key in sorted(strata):
        indices = strata[key]
        chosen.extend(sorted(rng.sample(indices, min(per_count, len(indices)))))
    return chosen


def _label(img, text, origin, color, scale=0.42):
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_dashed_skeleton(img, uv, vis, color):
    ih, iw = img.shape[:2]
    for a, b in SKELETON_EDGES:
        if not (vis[a] and vis[b]):
            continue
        start, end = uv[a], uv[b]
        length = float(np.linalg.norm(end - start))
        for s in np.arange(0, length, 8.0):
            p = start + (end - start) * (s / max(length, 1e-6))
            q = start + (end - start) * (min(s + 4.0, length) / max(length, 1e-6))
            pa, pb = tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int))
            if 0 <= pa[0] < iw and 0 <= pa[1] < ih and 0 <= pb[0] < iw and 0 <= pb[1] < ih:
                cv2.line(img, pa, pb, color, 1, cv2.LINE_AA)


def _top_down(size: int, gt, preds, matches, scores, missed):
    """Floor-plane (x, y) view of GT (dashed) and predictions, framed on the people."""
    tile = np.full((size, size, 3), 32, np.uint8)
    points = np.concatenate([gt[..., :2].reshape(-1, 2), preds[..., :2].reshape(-1, 2)])
    centre_xy = (points.min(0) + points.max(0)) / 2
    half = max(float((points.max(0) - points.min(0)).max()) / 2 + 400., 1000.)
    lo, hi = centre_xy - half, centre_xy + half
    margin = 24

    def to_px(xy):
        return np.round(margin + (xy - lo) / (hi - lo) * (size - 1 - 2 * margin)).astype(int)

    # 1 m grid for scale.
    for value in np.arange(np.ceil(lo[0] / 1000) * 1000, hi[0], 1000):
        x = to_px(np.array([value, lo[1]]))[0]
        cv2.line(tile, (x, 0), (x, size - 1), (60, 60, 60), 1)
    for value in np.arange(np.ceil(lo[1] / 1000) * 1000, hi[1], 1000):
        y = to_px(np.array([lo[0], value]))[1]
        cv2.line(tile, (0, y), (size - 1, y), (60, 60, 60), 1)
    for g, person in enumerate(gt):
        color = PALETTE[g % len(PALETTE)]
        uv = to_px(person[:, :2])
        _draw_dashed_skeleton(tile, uv.astype(np.float32), np.ones(len(uv), bool), color)
        centre = tuple(to_px(person[:, :2].mean(0)))
        cv2.circle(tile, centre, 9, color, 1, cv2.LINE_AA)
        if g in missed:
            _label(tile, "MISS", (centre[0] + 10, centre[1]), color)
    for p, person in enumerate(preds):
        color = PALETTE[matches[p] % len(PALETTE)] if p in matches else FALSE_POSITIVE
        uv = to_px(person[:, :2])
        for a, b in SKELETON_EDGES:
            cv2.line(tile, tuple(uv[a]), tuple(uv[b]), color, 2, cv2.LINE_AA)
        centre = tuple(to_px(person[:, :2].mean(0)))
        _label(tile, f"{scores[p]:.2f}", (centre[0] - 14, centre[1] - 12), color, 0.38)
    _label(tile, "top-down (x,y), 1 m grid", (6, 16), TILE_TEXT)
    return tile


def render_example(sample, cls_scores, kpts_mm, cfg, threshold, frames):
    imgs = sample["imgs"].numpy()                 # (T, N, 3, H, W) uint8 RGB
    world2img = sample["world2img"].numpy()       # (T, N, 4, 4)
    gt_all, valid = sample["gt_keypoints"].numpy(), sample["gt_valid"].numpy()
    vis_frac = sample["visibility"].numpy()       # (T, N, P)
    scores = cls_scores[-1, 0, ..., 0].sigmoid().cpu().numpy()   # (T, Q)
    kpts = kpts_mm[0].cpu().numpy()                               # (T, Q, J, 3)
    accumulator = MetricsAccumulator(cfg.eval.pck_thresholds_mm, cfg.eval.match_threshold_mm)
    summary = []
    rows = []
    for t in range(imgs.shape[0]):
        keep = np.flatnonzero(scores[t] > threshold)
        preds, gt = kpts[t, keep], gt_all[t, valid[t]]
        associations = accumulator.update_frame(preds, gt)
        matches = dict(associations)
        missed = set(range(len(gt))) - set(matches.values())
        errors = {g: float(np.linalg.norm(preds[p] - gt[g], axis=-1).mean()) for p, g in associations}
        summary.append(dict(frame=t, gt=len(gt), predicted=len(keep), matched=len(associations),
                            false_positive=len(keep) - len(associations), missed=len(missed),
                            mpjpe_mm={str(g): round(e, 1) for g, e in errors.items()}))
        if t not in frames:
            continue
        tiles = []
        for c in range(imgs.shape[1]):
            tile = cv2.cvtColor(np.ascontiguousarray(imgs[t, c].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
            for q in np.flatnonzero(scores[t] > 0.1):
                if q not in keep:
                    uv, ok = _project(world2img[t, c], kpts[t, q])
                    _draw_skeleton(tile, uv, ok, UNMATCHED_QUERY, radius=1, thickness=1)
            for g, person in enumerate(gt):
                uv, ok = _project(world2img[t, c], person)
                _draw_dashed_skeleton(tile, uv, ok, PALETTE[g % len(PALETTE)])
                if g in missed and ok.any():
                    head = uv[ok][0].astype(int)
                    _label(tile, f"MISS v={vis_frac[t, c, g]:.2f}", (head[0], head[1] - 6),
                           PALETTE[g % len(PALETTE)])
            for p, person in enumerate(preds):
                color = PALETTE[matches[p] % len(PALETTE)] if p in matches else FALSE_POSITIVE
                uv, ok = _project(world2img[t, c], person)
                _draw_skeleton(tile, uv, ok, color, radius=3, thickness=2)
            _label(tile, f"T{t} C{c}", (6, 16), TILE_TEXT)
            tiles.append(tile)
        size = tiles[0].shape[0]
        tiles.append(_top_down(size, gt, preds, matches, scores[t, keep], missed))
        row = np.concatenate(tiles, axis=1)
        frame = summary[-1]
        bar = np.full((24, row.shape[1], 3), 20, np.uint8)
        errors = ", ".join(f"P{g}:{e:.0f}mm" for g, e in sorted(errors.items()))
        _label(bar, f"frame {t}: GT {frame['gt']}  pred {frame['predicted']}  matched {frame['matched']}  "
                    f"FP {frame['false_positive']}  miss {frame['missed']}   {errors}", (6, 17), TILE_TEXT, 0.45)
        rows.append(np.vstack([bar, row]))
    metrics = accumulator.compute()
    return np.vstack(rows), summary, metrics


def legend(width: int) -> np.ndarray:
    bar = np.full((30, width, 3), 20, np.uint8)
    _label(bar, "Dashed = GT person (color per person)   Solid = prediction matched to that person   "
                "Red = false positive   Grey = queries 0.1-threshold   MISS v = visible fraction",
           (6, 20), TILE_TEXT, 0.45)
    return bar


def write_index(output: Path, header: dict, examples: list[dict]) -> None:
    cards = []
    for example in examples:
        m = example["metrics"]
        mpjpe = "n/a" if m.get("mpjpe_mm") is None or not np.isfinite(m["mpjpe_mm"]) else f"{m['mpjpe_mm']:.0f} mm"
        cards.append(
            f"<section><h2>{html.escape(example['file'])}: {example['people']} people, "
            f"{html.escape(example['capture_group'])}</h2>"
            f"<p>clip F1 {m['f1']:.2f} &middot; precision {m['precision']:.2f} &middot; recall {m['recall']:.2f}"
            f" &middot; MPJPE {mpjpe}</p><img src='{html.escape(example['file'])}' loading='lazy'></section>")
    page = (
        "<!doctype html><meta charset='utf-8'><title>Augmented checkpoint predictions</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#ddd;margin:20px}"
        "img{max-width:100%;border:1px solid #444}section{margin-bottom:36px}</style>"
        f"<h1>Augmented checkpoint, step {header['optimizer_steps']}, {header['split']} synthetic composites</h1>"
        f"<p>Score threshold {header['score_threshold']}; match threshold {header['match_threshold_mm']} mm. "
        f"Mean clip F1 across shown examples: {header['mean_f1']:.3f}. Examples are stratified by source "
        f"dataset and person count with seed {header['seed']}, not selected by quality.</p>" + "".join(cards))
    (output / "index.html").write_text(page, encoding="utf-8")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--per-count", type=int, default=3)
    parser.add_argument("--frames", default="0,4,7")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.data.composition_original_fraction = 0.
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(Path(args.checkpoint), device)
    model = UnifiedPerceptionModel(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    steps = state.get("optimizer_steps")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = MultiViewPoseDataset(cfg, split=args.split)
    frames = {int(f) for f in args.frames.split(",")}
    mean = torch.tensor(cfg.data.pixel_mean, device=device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(cfg.data.pixel_std, device=device).view(1, 1, 1, 3, 1, 1)
    examples = []
    for number, index in enumerate(choose_examples(dataset, args.per_count, args.seed)):
        sample = dataset[index]
        imgs = (sample["imgs"][None].to(device, torch.float32) - mean) / std
        cls_scores, kpts_norm, _, _ = model(imgs, sample["world2img"][None].to(device),
                                            sample["intrinsics"][None].to(device))
        kpts_mm = model.denormalize_keypoints(kpts_norm[-1])
        image, summary, metrics = render_example(sample, cls_scores, kpts_mm, cfg, args.score_threshold, frames)
        canvas = np.vstack([image, legend(image.shape[1])])
        people = int(sample["gt_valid"][0].sum())
        source = dataset.composed.entries[dataset.composed.recipes[index]["sources"][0]]["dataset"]
        name = f"{number:02d}_{source}_people{people}.png"
        cv2.imwrite(str(output / name), canvas)
        examples.append(dict(file=name, index=index, recipe=sample["seq"], dataset=source, people=people,
                             capture_group=sample["capture_group"], frames=summary,
                             metrics={k: float(v) for k, v in metrics.items()}))
        print(f"{name}: F1 {metrics['f1']:.2f} recall {metrics['recall']:.2f}")
    header = dict(checkpoint=str(Path(args.checkpoint).resolve()), optimizer_steps=steps, split=args.split,
                  score_threshold=args.score_threshold, match_threshold_mm=cfg.eval.match_threshold_mm,
                  seed=args.seed, mean_f1=float(np.mean([e["metrics"]["f1"] for e in examples])))
    (output / "examples.json").write_text(json.dumps(dict(**header, examples=examples), indent=2))
    write_index(output, header, examples)


if __name__ == "__main__":
    main()
