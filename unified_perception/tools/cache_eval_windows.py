"""Prebuild the local cache of slow evaluation windows (``data.eval_window_cache``).

    python -m unified_perception.tools.cache_eval_windows --config configs/gym_sixcam_track_v1.yaml

Fills exactly the windows the trainer's val_aligned pass reads (the same capped clip list), or
with ``--split train`` every training aligned window (served when ``data.train_window_cache``
is on), in parallel worker processes.  Safe to interrupt and rerun: finished windows are skipped.
"""

from __future__ import annotations

import argparse
import time

from torch.utils.data import DataLoader, Dataset

from ..config import load_config
from ..data.streaming import aligned_component, evaluation_lanes

VAL_ALIGNED_CAP = 512  # windows, as in engine/train.py


def _as_list(batch):
    return batch


class _Fill(Dataset):
    def __init__(self, load, keys):
        self.load, self.keys = load, keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        path = self.load.path(self.keys[i])
        if not path.exists():
            try:
                self.load(self.keys[i])
            except (RuntimeError, OSError, ValueError) as exc:     # e.g. network-drive hiccups
                print(f"  failed {self.keys[i]!r}: {str(exc).splitlines()[0][:160]}", flush=True)
                return 1
        return 0


def window_keys(cfg, split: str, component) -> list:
    """The aligned window keys a run reads: every training window, or the capped validation lanes."""
    if split == "val":
        return list(dict.fromkeys(item[1] for batch in evaluation_lanes([component], cfg.data.batch_size,
                                                                        VAL_ALIGNED_CAP) for item in batch))
    return [key for group in component.groups for key in group]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.data.eval_window_cache:
        raise SystemExit("data.eval_window_cache is not set in this config")
    component = aligned_component(cfg, args.split, cache=True)
    keys = window_keys(cfg, args.split, component)
    todo = [k for k in keys if not component.load.path(k).exists()]
    print(f"{args.split} aligned: {len(keys)} windows, {len(keys) - len(todo)} cached, {len(todo)} to build "
          f"-> {component.load.root}", flush=True)
    start = time.perf_counter()
    loader = DataLoader(_Fill(component.load, todo), batch_size=1, num_workers=args.workers,
                        collate_fn=_as_list)
    failed = 0
    for i, batch in enumerate(loader, 1):
        failed += sum(batch)
        if i % 25 == 0 or i == len(todo):
            rate = i / (time.perf_counter() - start)
            print(f"  {i}/{len(todo)}  {rate:.2f} windows/s  ({failed} failed)", flush=True)
    if failed:
        print(f"{failed} windows failed to load; rerun to retry them", flush=True)


if __name__ == "__main__":
    main()
