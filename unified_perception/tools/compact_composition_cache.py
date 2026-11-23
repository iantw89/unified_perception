"""Write compact siblings of a composition cache's NPY pixel files (data/composition.py).

    python -m unified_perception.tools.compact_composition_cache data/composition_640_c4_s8/cache --workers 12

RGB caches (``*-imgs.npy`` and ``renders/*.npy``, uint8 (..., 3, H, W)) become JPEG quality-95
packets (``.jpg.pt``, ~15x smaller); bool masks (``*-masks.npy``) become bit-packed ``.bits.npz``
(lossless, 8x smaller).  The loader prefers a compact sibling when present, so the NPY files can
stay locally and only the compact ones be shipped (a 416 GB cache shrinks to ~30 GB).  Idempotent
and safe to interrupt: siblings are written atomically and existing ones are skipped.
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..data.composition import BITS_SUFFIX, compact_path

QUALITY = 95


def convert(path: Path) -> tuple[int, int]:
    """(source bytes, compact bytes); compact = 0 when it already existed."""
    out = compact_path(path)
    if out.exists():
        return 0, 0
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    if out.name.endswith(BITS_SUFFIX):
        if array.dtype != np.bool_:
            raise ValueError(f"{path}: masks must be bool, got {array.dtype}")
        packed = np.packbits(np.asarray(array), axis=-1)
        with open(tmp, "wb") as stream:
            np.savez(stream, packed=packed, shape=np.asarray(array.shape), dtype=np.asarray(str(array.dtype)))
    else:
        import torch
        from torchvision.io import encode_jpeg
        if array.dtype != np.uint8 or array.shape[-3] != 3:
            raise ValueError(f"{path}: RGB caches must be uint8 (..., 3, H, W), got {array.dtype} {array.shape}")
        frames = torch.from_numpy(np.ascontiguousarray(array)).reshape(-1, *array.shape[-3:])
        torch.save({"shape": list(array.shape), "jpeg": [encode_jpeg(f, quality=QUALITY) for f in frames]}, tmp)
    os.replace(tmp, out)
    return path.stat().st_size, out.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cache", type=Path, help="composition cache folder (holds experiment.json)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    files = sorted(args.cache.glob("*.npy")) + sorted((args.cache / "renders").glob("*.npy"))
    todo = [f for f in files if not compact_path(f).exists()]
    print(f"{len(files)} NPY pixel files, {len(files) - len(todo)} already compact, {len(todo)} to convert",
          flush=True)
    start, src, dst = time.perf_counter(), 0, 0
    with Pool(args.workers) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(convert, todo, chunksize=4), 1):
            src, dst = src + a, dst + b
            if i % 200 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - start)
                print(f"  {i}/{len(todo)}  {rate:.1f} files/s  {src / 1e9:.1f} GB -> {dst / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
