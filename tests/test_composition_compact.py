"""Compact composition pixel caches: lossless masks, near-lossless JPEG frames, loader preference."""

import numpy as np

from unified_perception.data.composition import compact_path, load_pixels
from unified_perception.tools.compact_composition_cache import convert


def test_compact_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    masks = rng.random((2, 3, 40, 37)) > 0.5
    imgs = np.zeros((2, 3, 3, 40, 48), dtype=np.uint8)
    imgs[..., 10:30, 12:36] = rng.integers(60, 200, size=(2, 3, 3, 1, 1), dtype=np.uint8)
    np.save(tmp_path / "c-000-masks.npy", masks)
    np.save(tmp_path / "c-000-imgs.npy", imgs)
    for name in ("c-000-masks.npy", "c-000-imgs.npy"):
        src, dst = convert(tmp_path / name)
        assert dst > 0 and compact_path(tmp_path / name).exists()
        assert convert(tmp_path / name) == (0, 0)                  # idempotent
    (tmp_path / "c-000-masks.npy").unlink()
    (tmp_path / "c-000-imgs.npy").unlink()                        # loader must use the siblings
    back_masks = load_pixels(tmp_path / "c-000-masks.npy")
    back_imgs = load_pixels(tmp_path / "c-000-imgs.npy")
    assert back_masks.dtype == np.bool_ and np.array_equal(back_masks, masks)
    err = np.abs(back_imgs.astype(int) - imgs)
    assert back_imgs.shape == imgs.shape and err.mean() < 2.0      # (chroma subsampling peaks at edges)
