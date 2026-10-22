"""Rendering helpers: draw 2D/projected skeletons onto images."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import cv2

from ..constants import JOINT_COLORS, SKELETON_EDGES


def render_skeleton(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    visible: Optional[np.ndarray] = None,
    bone_color: Sequence[int] = (200, 200, 200),
    radius: int = 4,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a single skeleton (J, 2) onto ``image`` (H, W, 3) RGB in place.

    ``visible`` is an optional (J,) boolean mask of joints to draw.
    """
    J = keypoints_2d.shape[0]
    if visible is None:
        visible = np.ones(J, dtype=bool)

    for a, b in SKELETON_EDGES:
        if a < J and b < J and visible[a] and visible[b]:
            pa = tuple(np.round(keypoints_2d[a]).astype(int))
            pb = tuple(np.round(keypoints_2d[b]).astype(int))
            cv2.line(image, pa, pb, bone_color, thickness, lineType=cv2.LINE_AA)

    for j in range(J):
        if visible[j]:
            c = JOINT_COLORS[j % len(JOINT_COLORS)]
            p = tuple(np.round(keypoints_2d[j]).astype(int))
            cv2.circle(image, p, radius, c, -1, lineType=cv2.LINE_AA)
    return image


def render_people(
    height: int,
    width: int,
    people_kpts_2d: Sequence[np.ndarray],
    people_visible: Optional[Sequence[np.ndarray]] = None,
    background: int = 30,
) -> np.ndarray:
    """Render a list of people skeletons onto a fresh RGB image."""
    image = np.full((height, width, 3), background, dtype=np.uint8)
    for idx, kpts in enumerate(people_kpts_2d):
        vis = None if people_visible is None else people_visible[idx]
        render_skeleton(image, kpts, visible=vis)
    return image
