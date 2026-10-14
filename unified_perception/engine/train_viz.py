"""Training-time visualisation helpers.

Saves a PNG montage for each batch sample laid out as:

    ┌──────────────────────────────────────────────────────┐
    │  EPOCH N  |  loss=X.XXX  |  step=NNNN               │  ← header row
    ├─────────────────────────────────────────────────────-┤
    │  Sample 0                                            │  ← separator (if B>1)
    ├──────────┬──────────┬──────────┬──────────┤
    │  T0  C0  │  T0  C1  │  T0  C2  │  T0  C3  │
    ├──────────┼──────────┼──────────┼──────────┤
    │  T1  C0  │  ...                            │
    ├─────────────────────────────────────────────────────-┤
    │  Sample 1                                            │  ← separator
    ├──────────┬──────────┬──────────┬──────────┤
    │  ...                                                  │
    └──────────────────────────────────────────────────────┘
    │ LEGEND: ■ GT skeleton  ■ Best prediction  ■ All queries │  ← legend row

Colour scheme (BGR):
  (40, 220, 40)   bright green   GT keypoints / skeleton
  (40,  40, 220)  bright red     Best-scoring predicted query (top-1 per frame)
  (120, 120, 120) mid grey       All other predicted queries (background)
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch

from ..constants import SKELETON_EDGES

# Tile size for the legend / header bars
_HEADER_H = 28
_LEGEND_H = 28

# BGR colours
_COL_GT   = (40, 220,  40)   # bright green  – ground truth
_COL_BEST = (40,  40, 220)   # bright red    – top-scoring prediction
_COL_REST = (140, 140, 140)  # grey          – other queries
_COL_YOLO = (0,  165, 255)   # orange        – YOLO 2-D auxiliary head


def _denorm_img(chw: np.ndarray, mean, std) -> np.ndarray:
    img = chw.transpose(1, 2, 0) * np.array(std) + np.array(mean)
    return np.clip(img, 0, 255).astype(np.uint8)


def _project(world2img_4x4: np.ndarray, pts_mm: np.ndarray):
    """Project world points into camera image.

    Parameters
    ----------
    world2img_4x4 : (4, 4)
    pts_mm        : (J, 3)

    Returns
    -------
    uv  : (J, 2)  pixel coords
    vis : (J,)    bool, True when z > 1 mm
    """
    ones = np.ones((pts_mm.shape[0], 1), dtype=np.float32)
    h = np.concatenate([pts_mm, ones], axis=1)          # (J, 4)
    p = (world2img_4x4.astype(np.float32) @ h.T).T      # (J, 4)
    z = p[:, 2]
    vis = z > 1.0
    uv = np.zeros((pts_mm.shape[0], 2), dtype=np.float32)
    uv[vis] = p[vis, :2] / z[vis, None]
    return uv, vis


def _draw_yolo_det(img: np.ndarray, boxes: np.ndarray, kpts: np.ndarray) -> None:
    """Draw YOLO 2-D bounding boxes (thin rectangle) and keypoints (dots)."""
    ih, iw = img.shape[:2]
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (
            int(np.clip(boxes[i, 0], 0, iw - 1)),
            int(np.clip(boxes[i, 1], 0, ih - 1)),
            int(np.clip(boxes[i, 2], 0, iw - 1)),
            int(np.clip(boxes[i, 3], 0, ih - 1)),
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), _COL_YOLO, 1)
        for j in range(kpts.shape[1]):
            x, y = int(round(kpts[i, j, 0])), int(round(kpts[i, j, 1]))
            if 0 <= x < iw and 0 <= y < ih:
                cv2.circle(img, (x, y), 2, _COL_YOLO, -1)


def _draw_skeleton(img, uv, vis, color, radius=3, thickness=2):
    ih, iw = img.shape[:2]
    for j in range(len(uv)):
        if not vis[j]:
            continue
        x, y = int(round(uv[j, 0])), int(round(uv[j, 1]))
        if 0 <= x < iw and 0 <= y < ih:
            cv2.circle(img, (x, y), radius, color, -1)
    for a, b in SKELETON_EDGES:
        if vis[a] and vis[b]:
            xa, ya = int(round(uv[a, 0])), int(round(uv[a, 1]))
            xb, yb = int(round(uv[b, 0])), int(round(uv[b, 1]))
            if 0 <= xa < iw and 0 <= ya < ih and 0 <= xb < iw and 0 <= yb < ih:
                cv2.line(img, (xa, ya), (xb, yb), color, thickness)


def _make_header(width: int, epoch: int, extra_text: str = "") -> np.ndarray:
    bar = np.zeros((_HEADER_H, width, 3), np.uint8)
    bar[:] = (30, 30, 30)
    label = f"Epoch {epoch:04d}"
    if extra_text:
        label += f"   {extra_text}"
    cv2.putText(bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def _make_sample_bar(width: int, sample_idx: int) -> np.ndarray:
    """Thin separator bar labelled 'Sample N' shown between batch elements."""
    _SEP_H = 20
    bar = np.zeros((_SEP_H, width, 3), np.uint8)
    bar[:] = (55, 45, 25)   # dark blue-grey
    cv2.putText(bar, f"Sample {sample_idx}", (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 200, 255), 1, cv2.LINE_AA)
    return bar


def _make_legend(width: int) -> np.ndarray:
    bar = np.zeros((_LEGEND_H, width, 3), np.uint8)
    bar[:] = (20, 20, 20)
    items = [
        (_COL_GT,   "GT skeleton"),
        (_COL_BEST, "Best predicted query (top-1 score)"),
        (_COL_REST, "Other predicted queries"),
        (_COL_YOLO, "YOLO 2-D aux det"),
    ]
    x = 8
    for color, text in items:
        cv2.rectangle(bar, (x, 7), (x + 14, 21), color, -1)
        x += 18
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(bar, text, (x, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
        x += tw + 24
    return bar


def save_train_viz(
    out_dir: str,
    viz_examples: list,
    model,
    epoch: int = -1,
    extra_text: str = "",
) -> None:
    """Render one PNG per viz example and write them to out_dir.

    Parameters
    ----------
    out_dir    : directory to write into; files are named ``{sample_idx}.png``.
    viz_examples : list of (batch_cpu, kpts_norm, cls_scores, yolo_preds) tuples.
        batch_cpu  : dict with 'imgs' (1,T,N,3,H,W), 'world2img', etc.  (B=1 slice)
        kpts_norm  : (L, 1, T, Q, J, 3) normalised keypoint predictions for this sample.
        cls_scores : (L, 1, T, Q, 1) logits for this sample, or None.
        yolo_preds : flat list of T*N dicts for this sample, or None.
    epoch      : epoch number shown in the header.
    extra_text : appended to the header line (e.g. "loss=0.312").
    """
    os.makedirs(out_dir, exist_ok=True)

    pr   = model.position_range.cpu()

    for sample_idx, (batch, kpts_norm, cls_scores, yolo_preds) in enumerate(viz_examples):
        imgs_np  = batch["imgs"][0].numpy()          # (T, N, 3, H, W)
        w2i_np   = batch["world2img"][0].numpy()     # (T, N, 4, 4)
        gt_kpts  = batch["gt_keypoints"][0].numpy()  # (T, P, J, 3)  mm
        gt_valid = batch["gt_valid"][0].numpy()       # (T, P)

        T, N = imgs_np.shape[:2]

        kpts_last = kpts_norm[-1, 0].detach().cpu()              # (T, Q, J, 3) norm
        kpts_mm   = kpts_last * (pr[3:6] - pr[:3]) + pr[:3]     # mm

        if cls_scores is not None:
            scores = cls_scores[-1, 0].detach().cpu().float().sigmoid().max(dim=-1).values  # (T, Q)
        else:
            scores = torch.ones(T, kpts_mm.shape[1])

        Q = kpts_mm.shape[1]

        # One row per temporal frame, all cameras side-by-side.
        t_rows = []
        for t in range(T):
            best_q = int(scores[t].argmax().item())
            tiles = []
            for c in range(N):
                bgr = cv2.cvtColor(
                    imgs_np[t, c].transpose(1, 2, 0), cv2.COLOR_RGB2BGR
                )

                # --- YOLO 2-D auxiliary detections (orange, drawn first/bottom) ---
                if yolo_preds is not None:
                    det = yolo_preds[t * N + c]
                    if len(det["boxes"]) > 0:
                        _draw_yolo_det(bgr, det["boxes"], det["kpts"])

                # --- All queries (grey, thin) ---
                for q in range(Q):
                    if q == best_q:
                        continue
                    uv, vis = _project(w2i_np[t, c], kpts_mm[t, q].numpy())
                    _draw_skeleton(bgr, uv, vis, color=_COL_REST, radius=2, thickness=1)

                # --- Best-scoring prediction (red) ---
                uv, vis = _project(w2i_np[t, c], kpts_mm[t, best_q].numpy())
                _draw_skeleton(bgr, uv, vis, color=_COL_BEST, radius=4, thickness=2)

                # --- GT (green, thick on top) ---
                for p in range(gt_kpts.shape[1]):
                    if gt_valid[t, p]:
                        uv, vis = _project(w2i_np[t, c], gt_kpts[t, p])
                        _draw_skeleton(bgr, uv, vis, color=_COL_GT, radius=5, thickness=2)

                sc = scores[t, best_q].item()
                cv2.putText(bgr, f"T{t} C{c}  score={sc:.2f}", (5, bgr.shape[0] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(bgr)
            t_rows.append(np.concatenate(tiles, axis=1))

        example_block = np.concatenate(t_rows, axis=0)
        canvas_w = example_block.shape[1]
        header = _make_header(canvas_w, epoch, extra_text)
        legend = _make_legend(canvas_w)
        canvas = np.vstack([header, example_block, legend])
        cv2.imwrite(os.path.join(out_dir, f"{sample_idx}.png"), canvas)
