"""Image transforms that keep camera intrinsics consistent.

Mirrors the StreamPETR idea that any geometric image transform (resize/crop)
must be reflected in the camera intrinsics so that the 3D position encoding
stays correct.  Here we only need resize + normalise, but intrinsics are scaled
accordingly.
"""

from __future__ import annotations

from typing import Tuple, Union, overload

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_Array = Union[np.ndarray, torch.Tensor]


def fit_long_side(height: int, width: int, long_side: int) -> Tuple[int, int]:
    """(h, w) of a ``height`` x ``width`` image scaled so its longer side is ``long_side``
    (aspect kept; ``resize_image_and_intrinsics`` to this size adds no padding)."""
    s = long_side / max(height, width)
    return int(round(height * s)), int(round(width * s))


@overload
def resize_image_and_intrinsics(image: torch.Tensor, K: np.ndarray, out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, np.ndarray]: ...


@overload
def resize_image_and_intrinsics(image: torch.Tensor, K: torch.Tensor, out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]: ...


@overload
def resize_image_and_intrinsics(image: np.ndarray, K: np.ndarray, out_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]: ...


@overload
def resize_image_and_intrinsics(image: np.ndarray, K: torch.Tensor, out_hw: Tuple[int, int]) -> Tuple[np.ndarray, torch.Tensor]: ...


def resize_image_and_intrinsics(
    image: _Array, K: _Array, out_hw: Tuple[int, int]
) -> Tuple[_Array, _Array]:
    """Resize ``image`` to ``out_hw`` preserving aspect ratio with zero-padding.

    The image is scaled uniformly so it fits within the output canvas, then
    centered with zero-padding.  K is updated to reflect both the uniform scale
    and the translation introduced by the padding offset.

    Accepts ``np.ndarray`` (HWC) or ``torch.Tensor`` (HWC); returns the same
    type.  No cross-framework conversion is performed.
    """
    h, w = image.shape[:2]
    out_h, out_w = out_hw
    s = min(out_w / w, out_h / h)
    new_w, new_h = int(round(w * s)), int(round(h * s))
    pad_top  = (out_h - new_h) // 2
    pad_left = (out_w - new_w) // 2

    canvas = image
    if isinstance(image, torch.Tensor):
        t = image
        if new_h != h or new_w != w:
            # (H, W, C) → (1, C, H, W) for interpolate, then back
            t = image.permute(2, 0, 1).unsqueeze(0).float()
            # Use area interpolation for downsampling (matches cv2.INTER_AREA);
            # bilinear for upsampling.  area mode has no align_corners parameter.
            if new_h < h or new_w < w:
                t = F.interpolate(t, size=(new_h, new_w), mode="area")
            else:
                t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            t = t.squeeze(0).permute(1, 2, 0)
            if not image.is_floating_point():
                t = t.round()
            t = t.to(image.dtype)
        canvas = t
        if new_h != out_h or new_w != out_w:
            canvas = torch.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype, device=image.device)
            canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = t
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    # Intrinsics arithmetic is identical for both array types.
    if isinstance(K, np.ndarray):
        K2 = K.copy()
        K2[0, :] *= s
        K2[1, :] *= s
        K2[0, 2] += pad_left
        K2[1, 2] += pad_top
    else:
        K2 = K.clone()
        K2[0, :] = K2[0, :] * s
        K2[1, :] = K2[1, :] * s
        K2[0, 2] = K2[0, 2] + pad_left
        K2[1, 2] = K2[1, 2] + pad_top
    return canvas, K2


def normalize_image(image: _Array, mean, std) -> _Array:
    """RGB ``(H, W, 3)`` → normalised float32 ``(3, H, W)``.

    Accepts ``np.ndarray`` or ``torch.Tensor`` (both HWC); returns the same
    type in CHW layout.  ``np.transpose`` and ``Tensor.permute`` both return
    views — no data is copied.
    """
    if isinstance(image, torch.Tensor):
        img = image.float()
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=image.device)
        std_t  = torch.as_tensor(std,  dtype=torch.float32, device=image.device)
        return ((img - mean_t) / std_t).permute(2, 0, 1).contiguous()
    else:
        img  = image.astype(np.float32)
        mean = np.asarray(mean, dtype=np.float32)
        std  = np.asarray(std,  dtype=np.float32)
        img  = (img - mean[None, None, :]) / std[None, None, :]
        return np.transpose(img, (2, 0, 1))
