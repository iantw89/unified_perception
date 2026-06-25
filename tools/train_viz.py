"""Training-time visualisation helpers.

Saves a PNG montage for the first batch sample laid out as:

    ┌──────────────────────────────────────────────────────┐
    │  EPOCH N  |  loss=X.XXX  |  frame 0 cam 0  |  ...   │  ← header row
    ├──────────┬──────────┬──────────┬──────────┤
    │  T0  C0  │  T0  C1  │  T0  C2  │  T0  C3  │
    ├──────────┼──────────┼──────────┼──────────┤
    │  T1  C0  │  ...                            │
    │  ...                                       │
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

from unified_perception.constants import SKELETON_EDGES

# Tile size for the legend / header bars
_HEADER_H = 28
_LEGEND_H = 28

# BGR colours
_COL_GT   = (40, 220,  40)   # bright green  – ground truth
_COL_BEST = (40,  40, 220)   # bright red    – top-scoring prediction
_COL_REST = (140, 140, 140)  # grey          – other queries


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


def _make_legend(width: int) -> np.ndarray:
    bar = np.zeros((_LEGEND_H, width, 3), np.uint8)
    bar[:] = (20, 20, 20)
    items = [
        (_COL_GT,   "GT skeleton"),
        (_COL_BEST, "Best predicted query (top-1 score)"),
        (_COL_REST, "Other predicted queries"),
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
    out_path: str,
    batch: dict,
    kpts_norm: torch.Tensor,
    model,
    cls_scores: Optional[torch.Tensor] = None,
    epoch: int = -1,
    extra_text: str = "",
) -> np.ndarray:
    """Render the first sample from the batch and save a montage PNG.

    Parameters
    ----------
    kpts_norm  : (L, B, T, Q, J, 3)   normalised keypoint predictions.
    cls_scores : (L, B, T, Q, C)       classification logits (optional).
                 Uses last layer [-1]; sigmoid applied; max over classes used.
    epoch      : int  epoch number shown in the header.
    extra_text : str  appended to the header line (e.g. "loss=0.312").

    Returns
    -------
    canvas : (H, W, 3) uint8 BGR  — also written to out_path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    imgs_np  = batch["imgs"][0].numpy()          # (T, N, 3, H, W)
    w2i_np   = batch["world2img"][0].numpy()     # (T, N, 4, 4)
    gt_kpts  = batch["gt_keypoints"][0].numpy()  # (T, P, J, 3)  mm
    gt_valid = batch["gt_valid"][0].numpy()      # (T, P)

    # Use last decoder layer predictions for sample 0.
    kpts_last = kpts_norm[-1, 0].detach().cpu()  # (T, Q, J, 3) norm
    pr = model.position_range.cpu()
    kpts_mm = kpts_last * (pr[3:6] - pr[:3]) + pr[:3]   # (T, Q, J, 3) mm

    # Objectness scores: (T, Q) in [0, 1]; fall back to uniform if not given.
    if cls_scores is not None:
        scores = cls_scores[-1, 0].detach().cpu().float()  # (T, Q, C)
        scores = scores.sigmoid().max(dim=-1).values        # (T, Q)
    else:
        T, Q = kpts_mm.shape[:2]
        scores = torch.ones(T, Q)

    mean = getattr(model.cfg.data, "pixel_mean", [123.675, 116.28, 103.53])
    std  = getattr(model.cfg.data, "pixel_std",  [58.395,  57.12,  57.375])

    T, N = imgs_np.shape[:2]
    Q    = kpts_mm.shape[1]

    tiles = []
    for t in range(T):
        best_q = int(scores[t].argmax().item())   # top-scoring query for this frame

        for c in range(N):
            bgr = cv2.cvtColor(
                _denorm_img(imgs_np[t, c], mean, std), cv2.COLOR_RGB2BGR
            )

            # --- All queries (grey, thin) drawn first so GT sits on top ---
            for q in range(Q):
                if q == best_q:
                    continue
                uv, vis = _project(w2i_np[t, c], kpts_mm[t, q].numpy())
                _draw_skeleton(bgr, uv, vis, color=_COL_REST, radius=2, thickness=1)

            # --- Best-scoring prediction (red, medium) ---
            uv, vis = _project(w2i_np[t, c], kpts_mm[t, best_q].numpy())
            _draw_skeleton(bgr, uv, vis, color=_COL_BEST, radius=4, thickness=2)

            # --- GT (green, thick) on top ---
            for p in range(gt_kpts.shape[1]):
                if gt_valid[t, p]:
                    uv, vis = _project(w2i_np[t, c], gt_kpts[t, p])
                    _draw_skeleton(bgr, uv, vis, color=_COL_GT, radius=5, thickness=2)

            # Tile label (frame / cam)
            sc = scores[t, best_q].item()
            label = f"F{t} C{c}  score={sc:.2f}"
            cv2.putText(bgr, label, (5, bgr.shape[0] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(bgr)

    cols = N
    rows = T
    th, tw = tiles[0].shape[:2]
    canvas_w = cols * tw
    # Image grid
    grid = np.zeros((rows * th, canvas_w, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c_ = divmod(i, cols)
        grid[r * th:(r + 1) * th, c_ * tw:(c_ + 1) * tw] = tile

    header = _make_header(canvas_w, epoch, extra_text)
    legend = _make_legend(canvas_w)
    canvas = np.vstack([header, grid, legend])

    cv2.imwrite(out_path, canvas)
    return canvas