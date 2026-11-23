"""Label-gated background means and geometry-constrained foreground masks."""

from __future__ import annotations

import cv2
import numpy as np

from ..constants import SKELETON_EDGES

EXTRACTION_METHOD = "label-gated-mean-v2"


def person_region(joints: np.ndarray, projection: np.ndarray, image_hw: tuple[int, int],
                  margin_mm: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """Conservative image footprint, including an inferred head above shoulders."""
    if not np.isfinite(margin_mm) or margin_mm <= 0:
        raise ValueError("Background exclusion margin must be positive")
    shoulders, hips = joints[2:4].mean(axis=0), joints[:2].mean(axis=0)
    points = np.concatenate((joints, (shoulders + 0.6 * (shoulders - hips))[None]))
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    projected = homogeneous @ projection.T
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= margin_mm):
        raise ValueError("Cannot establish a safe projected person footprint")
    pixels = projected[:, :2] / projected[:, 2:3]
    # Project a world-axis-aligned cube around each point. This remains
    # conservative for any camera rotation, without assuming a world up axis.
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    expanded = (points[:, None] + margin_mm * corners).reshape(-1, 3)
    expanded = np.concatenate((expanded, np.ones((len(expanded), 1))), axis=1) @ projection.T
    if np.any(expanded[:, 2] <= 0):
        raise ValueError("Person exclusion volume intersects the camera plane")
    contour = np.rint(expanded[:, :2] / expanded[:, 2:3]).astype(np.int32)
    region = np.zeros(image_hw, dtype=np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(contour), 1)
    return region.astype(bool), np.rint(pixels[:len(joints)]).astype(int)


class BackgroundMean:
    """Streaming mean/variance using only label-certified background observations."""

    def __init__(self, shape: tuple[int, int, int]):
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sum_squares = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape[:2], dtype=np.uint32)

    def update(self, rgb: np.ndarray, excluded: np.ndarray) -> None:
        if rgb.shape != self.sum.shape or excluded.shape != self.count.shape or excluded.dtype != bool or rgb.dtype != np.uint8:
            raise ValueError("Background observations must share camera/image geometry")
        clean = (~excluded).astype(np.uint8)
        cv2.accumulate(rgb, self.sum, mask=clean)
        cv2.accumulateSquare(rgb, self.sum_squares, mask=clean)
        self.count += clean


    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        denominator = np.maximum(self.count, 1)[..., None]
        mean = self.sum / denominator
        variance = np.maximum(self.sum_squares / denominator - mean * mean, 0)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), self.count.copy()


def _limb_regions(joints: np.ndarray, projection: np.ndarray,
                  image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    homogeneous = np.concatenate((joints, np.ones((len(joints), 1))), axis=1)
    projected = homogeneous @ projection.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
    uv = projected[:, :2] / projected[:, 2:3]
    jacobian = (
        projection[None, :2, :3] - uv[:, :, None] * projection[None, 2:3, :3]
    ) / projected[:, 2:3, None]
    scale = np.linalg.norm(jacobian, axis=-1).max(axis=1)
    broad = np.zeros(image_hw, dtype=np.uint8)
    core = np.zeros(image_hw, dtype=np.uint8)
    near_feet = np.zeros(image_hw, dtype=np.uint8)
    for a, b in SKELETON_EDGES:
        radius = max(scale[a], scale[b])
        for canvas, millimetres in ((broad, 80.), (core, 40.)):
            thickness = max(1, int(np.ceil(2 * radius * millimetres)))
            cv2.line(canvas, tuple(pixels[a]), tuple(pixels[b]), 1, thickness)
    # Mocap ends at the ankle, not the toe. Protect a small shoe neighbourhood.
    for joint in (10, 11):
        radius = max(1, int(np.ceil(scale[joint] * 75.)))
        cv2.circle(core, tuple(pixels[joint]), radius, 1, -1)
        radius = max(1, int(np.ceil(scale[joint] * 250.)))
        cv2.circle(near_feet, tuple(pixels[joint]), radius, 1, -1)
    return broad.astype(bool), core.astype(bool), near_feet.astype(bool)


def _foreground_candidates(rgb: np.ndarray, joints: np.ndarray, projection: np.ndarray,
                           mean: np.ndarray, std: np.ndarray, region: np.ndarray,
                           pixel_threshold: float, noise_scale: float) -> np.ndarray:
    values = rgb.astype(np.float32)
    difference = np.abs(values - mean)
    strong = (difference > np.maximum(pixel_threshold, noise_scale * std)).any(axis=-1)
    limbs, core, near_feet = _limb_regions(joints, projection, rgb.shape[:2])
    # Weak, noise-significant changes can recover similarly coloured calves,
    # but only near actual bones and connected to strong foreground evidence.
    weak = (difference > np.maximum(min(6., pixel_threshold), noise_scale * std)).any(axis=-1)
    candidates = (strong | (weak & limbs)) & region
    brightness = (values * mean).sum(axis=-1) / np.maximum((mean * mean).sum(axis=-1), 1.)
    color_residual = np.linalg.norm(values - brightness[..., None] * mean, axis=-1)
    shadow = ((brightness >= 0.35) & (brightness < 0.98) & (color_residual < 6.)
              & (np.linalg.norm(mean, axis=-1) >= 30.) & near_feet & ~core)
    candidates &= ~shadow
    # Closing may connect shoes to shadowed carpet. Permit gap filling only
    # near bones, with measured weak foreground evidence, never on a shadow.
    closed = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    candidates |= closed.astype(bool) & weak & limbs & region & ~shadow
    _, labels = cv2.connectedComponents(candidates.astype(np.uint8))
    seeded = np.unique(labels[strong & candidates])
    seeded = seeded[seeded != 0]
    return np.isin(labels, seeded)


def foreground_mask(rgb: np.ndarray, joints: np.ndarray, projection: np.ndarray,
                    mean: np.ndarray, std: np.ndarray, count: np.ndarray,
                    margin_mm: float = 200.0, min_count: int = 3,
                    pixel_threshold: float = 20.0, noise_scale: float = 3.0) -> np.ndarray:
    if min_count < 1 or pixel_threshold <= 0 or noise_scale < 0:
        raise ValueError("Invalid background subtraction thresholds")
    region, pixels = person_region(joints, projection, rgb.shape[:2], margin_mm)
    supported = count >= min_count
    if not region.any() or np.any(region & ~supported):
        raise ValueError("Insufficient clean-background support near the person; add diverse reference frames")
    mask = _foreground_candidates(rgb, joints, projection, mean, std, region, pixel_threshold, noise_scale)
    # Drop small disconnected optical/noise artifacts, retaining separate body
    # components (e.g. shoes) rather than only the largest connected component.
    labels, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    cleaned = np.zeros_like(mask)
    for component in range(1, labels):
        if stats[component, cv2.CC_STAT_AREA] >= 8:
            cleaned |= components == component
    near_mask = cv2.dilate(cleaned.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    inside = ((pixels >= 0).all(axis=1) & (pixels[:, 0] < mask.shape[1])
              & (pixels[:, 1] < mask.shape[0]))
    covered = np.zeros(len(pixels), dtype=bool)
    covered[inside] = near_mask[pixels[inside, 1], pixels[inside, 0]]
    if cleaned.sum() < 32 or covered.mean() < 0.75:
        raise ValueError("Background-difference mask is incomplete at projected joints")
    return cleaned
