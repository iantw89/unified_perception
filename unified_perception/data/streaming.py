"""Streaming batches: each batch lane walks consecutive windows of one clip.

The model keeps its temporal memory and track state between calls, so a lane
that feeds windows ``[s, s+T), [s+T, s+2T), ...`` of one clip processes an
arbitrarily long sequence.  Every sample carries ``stream_reset`` (True on the
first window of a lane's clip, where the model must forget that lane's state)
and ``stream_pad`` (evaluation filler for lanes that already finished; ignore).

A *component* lists its clips as ordered groups of window keys and loads one
window per key.  Clips that cannot be streamed (e.g. fixed-length composition
recipes) are singleton groups, which reset every window.

Training (:class:`StreamingBatchSampler`) cuts every clip into consecutive pieces
at random places each pass (no window skipped; lengths from a per-component
:class:`ChunkPolicy`, 0 = whole clip), shuffles them, and lanes walk one piece at a
time; components are chosen so their share of *windows* matches the mixture.
Evaluation (:func:`evaluation_lanes`) runs every clip whole, exactly once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass
class StreamComponent:
    name: str
    groups: list[list[Any]]           # ordered window keys per clip
    load: Callable[[Any], dict]
    fraction: float = 1.0


def chained_windows(samples, seq_len: int) -> list[list[Any]]:
    """Group ``(seq, start, *rest)`` samples into non-overlapping consecutive chains."""
    by_clip: dict[tuple, set[int]] = {}
    for sample in samples:
        seq, start, *rest = sample
        by_clip.setdefault((seq, *[tuple(r) if isinstance(r, list) else r for r in rest]), set()).add(int(start))
    groups = []
    for clip in sorted(by_clip, key=str):
        starts = by_clip[clip]
        for first in sorted(starts):
            if first - seq_len in starts:
                continue  # inside a chain that started earlier
            chain, start = [], first
            while start in starts:
                chain.append((clip[0], start, *clip[1:]))
                start += seq_len
            groups.append(chain)
    return groups


class StreamingDataset(Dataset):
    """Loads ``(component, key, reset, pad[, piece_seed])`` items yielded by the samplers.

    With ``camera_dropout_prob`` > 0, training items carry ``(piece_seed, window_index)``: the
    piece seed decides once whether the lane drops cameras for this piece, and each window
    then draws an independent random view subset per frame (data/augment.py).
    """

    def __init__(self, components: list[StreamComponent], camera_dropout_prob: float = 0.0,
                 camera_dropout_min_views: int = 1):
        self.components = components
        self.dropout_prob, self.dropout_min_views = camera_dropout_prob, camera_dropout_min_views

    def __len__(self) -> int:
        return sum(len(g) for c in self.components for g in c.groups)

    def __getitem__(self, item):
        component, key, reset, pad = item[:4]
        sample = self.components[component].load(key)
        if len(item) > 4 and self.dropout_prob > 0:
            from .augment import frame_camera_masks, mask_cameras
            piece_seed, window = item[4] if isinstance(item[4], tuple) else (item[4], 0)
            T, N = sample["imgs"].shape[:2]
            if N > self.dropout_min_views and random.Random(piece_seed).random() < self.dropout_prob:
                rng = random.Random(piece_seed * 100_003 + window + 1)
                sample = mask_cameras(sample, frame_camera_masks(int(T), int(N), rng, self.dropout_min_views))
        sample["stream_reset"] = bool(reset)
        sample["stream_pad"] = bool(pad)
        return sample


@dataclass
class ChunkPolicy:
    """Piece lengths in windows (0 = whole clip) and each length's share of windows."""
    windows: list[int]
    window_share: list[float]

    def __post_init__(self):
        if not self.windows or len(self.windows) != len(self.window_share):
            raise ValueError("chunk lengths and window shares must be non-empty and aligned")
        if any(w < 0 for w in self.windows) or any(s < 0 for s in self.window_share)                 or sum(self.window_share) <= 0:
            raise ValueError("chunk lengths must be >= 1 window (0 = whole clip) with positive shares")
        total = float(sum(self.window_share))
        shares = dict(zip(self.windows, (s / total for s in self.window_share)))
        self.whole_share = shares.pop(0, 0.0)
        finite = [(w, s) for w, s in shares.items() if s > 0]
        # A finite length l covering share s of the windows is drawn with probability ~ s / l.
        self.lengths = [w for w, _ in finite]
        self.draw = np.asarray([s / w for w, s in finite], dtype=float)
        if len(self.draw):
            self.draw /= self.draw.sum()

    @property
    def streaming(self) -> bool:
        return self.windows != [1] and any(w != 1 for w, s in zip(self.windows, self.window_share) if s > 0)

    def partition(self, groups: list[list], rng: random.Random) -> list[list]:
        """Cut every clip into consecutive pieces at fresh random places; every window once."""
        pieces = []
        for group in groups:
            # Each clip is kept whole with probability whole_share, so every window has that
            # chance of being seen in a whole-recording piece (also with a single long clip).
            if not self.lengths or rng.random() < self.whole_share:
                pieces.append(list(group))
                continue
            start = 0
            while start < len(group):
                length = int(rng.choices(self.lengths, weights=self.draw)[0])
                pieces.append(list(group[start:start + length]))
                start += length
        rng.shuffle(pieces)
        return pieces


class StreamingBatchSampler(Sampler[list]):
    """Training lanes over piece queues; ``num_batches`` batches per epoch.

    Per component, a *pass* cuts all clips with that component's
    :class:`ChunkPolicy` and shuffles the pieces; a new pass with new cut points
    starts when the previous one is used up, so every window is visited once per
    pass.  Lanes persist across epochs: an hours-long piece simply continues in
    the next epoch (the trainer's evaluation saves/restores the model's streaming
    state).

    Lane allocation follows the configured fractions *per batch*, not just on
    average: each component owns ``floor(fraction * batch_size)`` lanes outright
    (e.g. gym 0.5 of 4 lanes = 2 lanes in every batch), and the remaining lanes
    are shared, a free shared lane taking the component whose shared lane-time is
    furthest below its leftover quota.  (The previous rule drew a component per
    free lane with weights fraction / mean piece length, estimated from one random
    cut: with a single long gym clip the realized gym share ranged 9-65% across
    seeds, and two thirds of the steps could have no gym lane at all.)
    """

    def __init__(self, components: list[StreamComponent], batch_size: int, num_batches: int,
                 policies: list[ChunkPolicy], seed: int = 0):
        pairs = [(c, p) for c, p in zip(components, policies) if c.groups and c.fraction > 0]
        if not pairs:
            raise ValueError("No streamable training clips")
        self.components = [c for c, _ in pairs]
        self.policies = [p for _, p in pairs]
        self.batch_size, self.num_batches = batch_size, num_batches
        self.rng = random.Random(seed)
        self.queues: list[list[list]] = [[] for _ in self.components]
        fractions = np.asarray([c.fraction for c in self.components], dtype=float)
        quotas = fractions / fractions.sum() * batch_size
        owned = np.floor(quotas + 1e-9).astype(int)
        #: lane -> owning component index, or None for a shared lane.
        self.lane_owner: list[Optional[int]] = [i for i, n in enumerate(owned) for _ in range(n)]
        self.lane_owner += [None] * (batch_size - len(self.lane_owner))
        leftover = quotas - owned
        self.shared_target = leftover / leftover.sum() if leftover.sum() > 1e-9 else None
        self.shared_used = np.zeros(len(self.components))   # windows committed in shared lanes
        self.lanes: list[list] = [[] for _ in range(batch_size)]

    def __len__(self) -> int:
        return self.num_batches

    def _shared_component(self) -> int:
        """The component furthest below its share of the shared lanes (random tie-break)."""
        total = self.shared_used.sum()
        deficit = self.shared_target * max(total, 1.0) - self.shared_used
        deficit[self.shared_target <= 0] = -np.inf
        best = np.flatnonzero(deficit >= deficit.max() - 1e-9)
        return int(best[self.rng.randrange(len(best))])

    def _next_piece(self, lane: int) -> list:
        index = self.lane_owner[lane]
        shared = index is None
        if shared:
            index = self._shared_component()
        if not self.queues[index]:
            self.queues[index] = self.policies[index].partition(self.components[index].groups, self.rng)
        piece = self.queues[index].pop()
        if shared:
            self.shared_used[index] += len(piece)
        seed = self.rng.getrandbits(48)  # dropout lane decision per piece, view draws per window
        return [(index, key, (seed, i)) for i, key in enumerate(piece)]

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for lane_index, lane in enumerate(self.lanes):
                reset = not lane
                if reset:
                    lane.extend(self._next_piece(lane_index))
                component, key, seed = lane.pop(0)
                batch.append((component, key, reset, False, seed))
            yield batch


def chunk_policies(cfg, components: list[StreamComponent]) -> list[ChunkPolicy]:
    d = cfg.data
    domain = ChunkPolicy(list(d.stream_domain_chunks), list(d.stream_domain_window_share))
    aux = ChunkPolicy(list(d.stream_aux_chunks), list(d.stream_aux_window_share))
    return [domain if c.name in d.stream_domain_components else aux for c in components]


def streaming_enabled(cfg) -> bool:
    d = cfg.data
    return (ChunkPolicy(list(d.stream_domain_chunks), list(d.stream_domain_window_share)).streaming
            or ChunkPolicy(list(d.stream_aux_chunks), list(d.stream_aux_window_share)).streaming)


def evaluation_lanes(components: list[StreamComponent], batch_size: int,
                     max_windows_total: Optional[int] = None) -> list[list]:
    """Deterministic batches running every clip whole (balanced over lanes, padded)."""
    clips = [(ci, g) for ci, c in enumerate(components) for g in c.groups]
    if max_windows_total is not None:
        kept, total = [], 0
        for clip in clips:
            if total >= max_windows_total:
                break
            kept.append(clip)
            total += len(clip[1])
        clips = kept
    lanes: list[list] = [[] for _ in range(batch_size)]
    for ci, group in sorted(clips, key=lambda c: -len(c[1])):
        lane = min(lanes, key=len)
        lane.extend((ci, key, i == 0, False) for i, key in enumerate(group))
    length = max((len(lane) for lane in lanes), default=0)
    batches = []
    for step in range(length):
        batch = []
        for lane in lanes:
            if step < len(lane):
                batch.append(lane[step])
            elif lane:
                ci, key, _, _ = lane[-1]
                batch.append((ci, key, False, True))
            else:  # a lane with no clip at all repeats another lane's window, ignored
                ci, key, _, _ = next(l for l in lanes if l)[0]
                batch.append((ci, key, True, True))
        batches.append(batch)
    return batches


# --------------------------------------------------------------------------- #
# Components for the existing datasets                                         #
# (loaders are classes, not lambdas, so DataLoader workers can pickle them)    #
# --------------------------------------------------------------------------- #
class _GymWindow:
    def __init__(self, provider):
        self.provider = provider

    def __call__(self, key):
        from .gym_pseudo3d import _to_sample
        return _to_sample(self.provider.load_window(key))


class _IndexedWindow:
    """Loads ``dataset[index[(seq, start)]]``, optionally post-processed."""

    def __init__(self, dataset, samples, aligned: bool = False):
        self.dataset, self.aligned = dataset, aligned
        self.index = {tuple(sample[:2]): i for i, sample in enumerate(samples)}

    def __call__(self, key):
        sample = self.dataset[self.index[tuple(key[:2])]]
        if self.aligned:
            from .gym_pseudo3d import decorate_aligned
            sample = decorate_aligned(sample)
        return sample


def gym_component(cfg, split: str, fraction: float = 1.0) -> StreamComponent:
    from .gym_pseudo3d import GymPseudo3DProvider
    provider = GymPseudo3DProvider(cfg, split)
    T = cfg.data.seq_len
    groups = []
    for name, (_, poses, _) in provider.sequences.items():
        starts = list(range(0, len(poses["frames"]) - T + 1, T))
        for views in provider.view_subsets:
            groups.append([(name, start, views) for start in starts])
    return StreamComponent("gym", groups, _GymWindow(provider), fraction)


def aligned_component(cfg, split: str, fraction: float = 1.0, cache: bool = False) -> StreamComponent:
    """Aligned single-person mocap windows; ``cache`` serves them through :class:`CachedWindows`
    under ``data.eval_window_cache`` (when set)."""
    from .gym_pseudo3d import AlignedSingleDataset
    base = AlignedSingleDataset(cfg, split).base
    groups = chained_windows(base.samples, cfg.data.seq_len)
    load = _IndexedWindow(base, base.samples, aligned=True)
    if cache and cfg.data.eval_window_cache:
        load = CachedWindows(load, eval_cache_root(cfg, "aligned", split),
                             required=cfg.data.window_cache_required)
    return StreamComponent("aligned", groups, load, fraction)


class CachedWindows:
    """Disk cache in front of a component loader: images stored as JPEG (quality 95), every
    other field as-is, one file per window key.

    For sources whose windows are slow to produce (aligned mocap decodes several videos from
    a network drive: ~2-3 s per window vs ~0.1 s from this cache).  The first load of a key
    returns the JPEG round trip too, so every pass sees identical pixels.  Files are written
    atomically, so concurrent DataLoader workers and interrupted runs are safe.
    """

    def __init__(self, load: Callable, root, quality: int = 95, required: bool = False):
        from pathlib import Path
        self.load, self.root, self.quality, self.required = load, Path(root), quality, required

    def path(self, key):
        import hashlib
        digest = hashlib.sha1(repr(key).encode()).hexdigest()[:24]
        return self.root / digest[:2] / f"{digest}.pt"

    def __call__(self, key):
        import os
        from torchvision.io import decode_jpeg, encode_jpeg
        path = self.path(key)
        if path.exists():
            try:
                data = torch.load(path, weights_only=True)
            except Exception:  # unreadable (e.g. disk trouble): rebuild it
                data = None
            if data is not None and data.get("key") == repr(key):
                sample = data["sample"]
                sample["imgs"] = torch.stack(decode_jpeg(data["jpeg"])).view(data["shape"])
                return sample
        if self.required:
            raise FileNotFoundError(f"Window {key!r} is not in the cache {self.root} "
                                    "(data.window_cache_required: prebuild it with cache_eval_windows)")
        sample = self.load(key)
        imgs = sample["imgs"]
        jpeg = [encode_jpeg(image, quality=self.quality) for image in imgs.reshape(-1, *imgs.shape[-3:])]
        rest = {k: v for k, v in sample.items() if k != "imgs"}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        torch.save({"key": repr(key), "shape": list(imgs.shape), "jpeg": jpeg, "sample": rest}, tmp)
        os.replace(tmp, path)
        sample["imgs"] = torch.stack(decode_jpeg(jpeg)).view(imgs.shape)
        return sample


def eval_cache_root(cfg, name: str, split: str):
    """Cache folder for one evaluation source, named by a fingerprint of everything that
    shapes its windows (a changed setting starts a fresh folder instead of serving stale data)."""
    import hashlib
    import json
    from pathlib import Path
    d = cfg.data
    spec = {"source": name, "split": split, "dirs": list(d.gym_mix_aligned_dirs), "seq_len": d.seq_len,
            "image_hw": [d.image_height, d.image_width], "fps": d.human_prior_target_fps,
            "num_cameras": d.num_cameras, "max_people": d.max_people, "split_seed": d.split_seed,
            "val_fraction": d.val_fraction, "stride": d.stride, "format": 1}
    if d.image_long_side:
        # Native-aspect caches: keyed by the long side and the source folder *names*, so the same
        # cache resolves on another machine (e.g. a cloud pod) whatever the mount points are.
        from pathlib import PureWindowsPath
        spec.update(dirs=[PureWindowsPath(p).name for p in d.gym_mix_aligned_dirs],
                    image_hw=["long_side", d.image_long_side], format=2)
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    root = Path(d.eval_window_cache) / f"{name}_{split}_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(spec, indent=2))
    return root


def windows_component(dataset, name: str, fraction: float, seq_len: int) -> StreamComponent:
    """A MultiViewPoseDataset: provider windows chained by clip; composition recipes are singletons."""
    if dataset.composed is not None:
        return StreamComponent(name, [[i] for i in range(len(dataset))], dataset.__getitem__, fraction)
    groups = chained_windows(dataset.samples, seq_len)
    return StreamComponent(name, groups, _IndexedWindow(dataset, dataset.samples), fraction)


def training_components(cfg) -> list[StreamComponent]:
    from .dataset import MultiViewPoseDataset
    T = cfg.data.seq_len
    if cfg.data.provider != "gym_mocap_mix":
        return [windows_component(MultiViewPoseDataset(cfg, "train"), cfg.data.provider, 1.0, T)]
    gym = cfg.data.gym_sample_fraction
    aligned = cfg.data.gym_mix_aligned_fraction if cfg.data.gym_mix_aligned_dirs else 0.0
    components = [windows_component(MultiViewPoseDataset(cfg, "train"), "mocap", 1 - gym - aligned, T),
                  gym_component(cfg, "train", gym)]
    if aligned:
        components.append(aligned_component(cfg, "train", aligned, cache=cfg.data.train_window_cache))
    return components


def collate_stream_flags(samples) -> dict[str, torch.Tensor]:
    return {
        "stream_reset": torch.tensor([bool(s.get("stream_reset", True)) for s in samples]),
        "stream_pad": torch.tensor([bool(s.get("stream_pad", False)) for s in samples]),
    }
