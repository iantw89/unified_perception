#!/usr/bin/env python
"""Visualise dataset ground truth and/or model predictions.

For a chosen sequence, overlays ground-truth (green) and predicted (red)
skeletons projected into each calibrated camera view, writing per-frame montage
PNGs (and optionally an MP4) to an output directory.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import cv2
import torch

from unified_perception.engine import resolve_device
from unified_perception.models import UnifiedPerceptionModel
from unified_perception.config import load_config
from unified_perception.data import MultiViewPoseDataset, collate_batch
from unified_perception.data.render import render_skeleton


def _denormalize_image(img_chw, mean, std):
    img = np.transpose(img_chw, (1, 2, 0))
    img = (
        img * np.asarray(std, dtype=np.float32)
        + np.asarray(mean, dtype=np.float32)
    )
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def _project_world2img(world2img, keypoints_mm):
    pts = np.asarray(keypoints_mm, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    proj = (np.asarray(world2img, dtype=np.float32) @ pts_h.T).T
    z = proj[:, 2]
    visible = z > 1e-6
    uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
    uv[visible] = proj[visible, :2] / z[visible, None]
    return uv, visible


def draw_people(image, world2img, people_kpts_mm, color):
    for kp3d in people_kpts_mm:
        uv, visible = _project_world2img(world2img, kp3d)
        render_skeleton(
            image,
            uv,
            visible=visible,
            bone_color=color,
            radius=3,
            thickness=2,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="optional; if omitted, GT only",
    )
    ap.add_argument("--split", default="val")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--out", default="runs/vis")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    os.makedirs(args.out, exist_ok=True)

    dataset = MultiViewPoseDataset(cfg, split=args.split)
    sample = dataset[args.sample_index]
    start = sample["start"]
    frame_ids = sample.get("frame_ids")
    if frame_ids is not None:
        frame_ids = frame_ids.numpy()
    imgs = sample["imgs"].numpy()  # (T, N, 3, H, W), normalized RGB
    world2img = sample["world2img"].numpy()  # (T, N, 4, 4)
    T, N = imgs.shape[0], imgs.shape[1]

    pred_people_per_frame = None
    if args.checkpoint:
        device = resolve_device(cfg.train.device)
        model = UnifiedPerceptionModel(cfg).to(device)
        ckpt = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        batch = collate_batch([sample])
        with torch.no_grad():
            cls_scores, kpts_norm, _ = model(
                batch["imgs"].to(device),
                batch["world2img"].to(device),
                batch["intrinsics"].to(device),
            )
        scores = cls_scores[-1, 0, :, :, 0].sigmoid()       # (T, Q)
        kpts_mm = model.denormalize_keypoints(kpts_norm[-1, 0]).cpu().numpy()
        pred_people_per_frame = []
        for t in range(scores.shape[0]):
            keep = (scores[t] > args.score_threshold).cpu().numpy()
            pred_people_per_frame.append(
                [kpts_mm[t, q] for q in np.where(keep)[0]]
            )

    gt_kpts = sample["gt_keypoints"].numpy()   # (T,P,J,3) mm
    gt_valid = sample["gt_valid"].numpy()
    writer = None
    for t in range(T):
        tiles = []
        for c in range(N):
            img = _denormalize_image(
                imgs[t, c], cfg.data.pixel_mean, cfg.data.pixel_std
            )

            gt_people = [gt_kpts[t, p] for p in np.where(gt_valid[t])[0]]
            draw_people(
                img, world2img[t, c], gt_people, color=(40, 220, 40)
            )
            if pred_people_per_frame is not None:
                draw_people(
                    img,
                    world2img[t, c],
                    pred_people_per_frame[t],
                    color=(230, 40, 40),
                )

            cv2.putText(
                img,
                f"frame {int(frame_ids[t]) if frame_ids is not None else start + t} cam{c}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            tiles.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        h, w = tiles[0].shape[:2]
        montage = np.full((rows * h, cols * w, 3), 0, np.uint8)
        for i, tile in enumerate(tiles):
            r, cc = divmod(i, cols)
            montage[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = tile

        out_path = os.path.join(args.out, f"frame_{t:03d}.png")
        cv2.imwrite(out_path, montage)
        if args.video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(  # type: ignore[attr-defined]
                    *"mp4v"
                )
                writer = cv2.VideoWriter(
                    os.path.join(args.out, "vis.mp4"), fourcc, 5.0,
                    (montage.shape[1], montage.shape[0]),
                )
            writer.write(montage)

    if writer is not None:
        writer.release()
    legend = "GT=green" + (
        ", pred=red" if pred_people_per_frame is not None else ""
    )
    print(f"Wrote {T} frames to {args.out} ({legend})")


if __name__ == "__main__":
    main()
