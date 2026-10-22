"""Camera geometry helpers (numpy).

Calibration convention (matches the human_prior project):
    X_cam = R @ X_world + T          (R: world->cam rotation, T: translation)
    x_img = K @ X_cam ; u,v = x_img[:2] / x_img[2]

All world coordinates are in millimetres.

We also build the 4x4 ``world2img`` projection matrix used by the model's 3D
position encoder:

    world2img = K4 @ [R | T]
                     [0 | 1]

where ``K4`` is the 3x3 intrinsics padded to 4x4.  ``img2world`` is its inverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    K: np.ndarray  # (3, 3) intrinsics
    R: np.ndarray  # (3, 3) world->cam rotation
    T: np.ndarray  # (3,)   world->cam translation
    width: int
    height: int

    def world2img_matrix(self) -> np.ndarray:
        """Return the 4x4 world->image projection matrix."""
        extr = np.eye(4, dtype=np.float64)
        extr[:3, :3] = self.R
        extr[:3, 3] = self.T
        k4 = np.eye(4, dtype=np.float64)
        k4[:3, :3] = self.K
        return k4 @ extr

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project (N, 3) world points to (N, 2) pixel coordinates."""
        pts = np.asarray(points_world, dtype=np.float64)
        cam = pts @ self.R.T + self.T[None, :]
        z = np.clip(cam[:, 2:3], 1e-6, None)
        uv = (self.K @ (cam / z).T).T[:, :2]
        return uv

    def in_front(self, points_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_world, dtype=np.float64)
        cam = pts @ self.R.T + self.T[None, :]
        return cam[:, 2] > 1e-6


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale a 3x3 intrinsics matrix for an image resized by (sx, sy)."""
    K = K.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K
