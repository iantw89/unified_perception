"""Hierarchical shuffle-buffer dataset.

Maintains an in-memory pool of fully-decoded sequences and serves randomly
sampled temporal windows from them.  A background thread asynchronously
decodes the next batch of sequences so that the main iterator thread is
never blocked waiting for I/O.

Separation of responsibilities
--------------------------------
* ``HumanPriorAlignedProvider`` (backends.py) owns all knowledge about
  video files, frame-index arithmetic, mocap parsing, and camera geometry.
  Its ``load_full_sequence`` method exposes a complete decoded sequence as
  uint8-image arrays + GT.

* ``ShuffleBufferDataset`` / ``SequenceBuffer`` (this module) own all
  knowledge about buffering, refilling, and window sampling.  They know
  nothing about video formats, MOYO vs. H36M, etc.

Memory model
------------
The buffer is sized in units of *camera-frames*: one uint8 CHW image at the
configured output resolution.  A sequence with T timesteps and N cameras
contributes ``T * N`` camera-frames to the pool.  Setting ``buffer_frames``
larger keeps more sequences in memory, improving shuffle diversity at the
cost of RAM.  A 4096-frame buffer at 640×640 uses ~5 GB; 1024 frames ~1.2 GB.
"""

from __future__ import annotations

import queue
import random
import threading
import time
import traceback
import uuid
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..config import Config


class PoolExhausted(Exception):
    """Sentinel: the decode pass is complete and all pool windows are consumed."""


# Per-worker persistent buffers keyed by "<dataset_id>:<worker_id>".
# Stored at module level so they survive across epoch boundaries (DataLoader
# calls __iter__ again each epoch but keeps the worker process alive when
# persistent_workers=True).
WORKER_BUFS: Dict[str, "SequenceBuffer"] = {}


class ShuffleBufferDataset(IterableDataset):
    """Streaming dataset backed by a per-worker in-memory sequence buffer.

    Each DataLoader worker:
      1. Receives a disjoint shard of the sequence list.
      2. Maintains an in-memory pool of fully-decoded sequences.
      3. Serves randomly-sampled temporal windows from the pool.
      4. Refills the pool asynchronously via a background decode thread.

    Parameters
    ----------
    provider:
        A ``HumanPriorAlignedProvider`` instance (or any provider that
        implements ``load_full_sequence`` and exposes ``._seqs``).
    seq_ids:
        All unique sequence identifiers to draw from this split.
    cfg:
        Full training config.
    """

    def __init__(self, provider, seq_ids: List[str], cfg: Config) -> None:
        super().__init__()
        self.provider     = provider
        self.seq_ids      = list(seq_ids)
        self.seq_len      = cfg.data.seq_len
        self.num_cameras  = cfg.data.num_cameras
        self.buffer_frames        = cfg.data.buffer_frames
        self.min_buffer_seg       = cfg.data.min_buffer_seg
        _depth = getattr(cfg.data, "prefetch_queue_depth", 0)
        self.prefetch_queue_depth = _depth if _depth > 0 else self.min_buffer_seg
        self.max_samples          = cfg.data.max_samples_per_epoch
        self.mean = np.array(cfg.data.pixel_mean, dtype=np.float32)
        self.std  = np.array(cfg.data.pixel_std,  dtype=np.float32)
        # Unique ID so each Dataset instance gets its own buffer slots.
        self._buf_id: str = uuid.uuid4().hex

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        total = sum(
            max(0, self.provider._seqs[sid].num_frames - self.seq_len + 1)
            for sid in self.seq_ids
            if sid in self.provider._seqs
        )
        if self.max_samples > 0:
            return min(self.max_samples, total)
        return total

    def __iter__(self):
        worker_info = get_worker_info()

        # Shard sequences across workers (round-robin by index).
        seq_ids = list(self.seq_ids)
        if worker_info is not None:
            n   = worker_info.num_workers
            wid = worker_info.id
            seq_ids = [s for i, s in enumerate(seq_ids) if i % n == wid]
        else:
            n, wid = 1, 0

        # Fresh shuffle every epoch so the sequence order varies.
        random.shuffle(seq_ids)

        # Reuse the buffer across epochs so the background thread stays warm
        # and we avoid re-paying the initial-fill cost every epoch.
        key = f"{self._buf_id}:{wid}"
        if key not in WORKER_BUFS:
            WORKER_BUFS[key] = SequenceBuffer(
                provider=self.provider,
                seg_ids=seq_ids,
                seq_len=self.seq_len,
                num_cameras=self.num_cameras,
                buffer_frames=self.buffer_frames,
                min_buffer_seg=self.min_buffer_seg,
                prefetch_queue_depth=self.prefetch_queue_depth,
                mean=self.mean,
                std=self.std,
                worker_id=wid,
            )
        else:
            WORKER_BUFS[key].reset(seq_ids)

        buf = WORKER_BUFS[key]

        # Optional hard cap (max_samples_per_epoch > 0 in config).
        max_per_worker = self.max_samples // n if self.max_samples > 0 else 0
        count = 0

        while max_per_worker == 0 or count < max_per_worker:
            try:
                yield buf.next_window()
                count += 1
            except PoolExhausted:
                # Decode pass complete and pool drained: natural epoch end.
                break


# ---------------------------------------------------------------------------
# Internal: per-worker buffer
# ---------------------------------------------------------------------------

class SequenceBuffer:
    """In-memory pool of decoded segment chunks with async background prefetch.

    Each pool entry pre-generates a shuffled list of all valid sequence start
    indices for that segment chunk.  ``next_window()`` samples proportionally
    to remaining sequence count, then pops the last element of the (already-
    shuffled) pending list (O(1), without replacement).  When all sequences of
    an entry are consumed the segment chunk is evicted naturally.

    The background thread performs exactly **one pass** through the segment
    list per epoch, then stops.  When the pass is done and the pool is drained,
    ``next_window`` raises ``PoolExhausted``, which the caller interprets as
    the natural end of the epoch.  The buffer is reused across epochs via
    ``reset()``, which restarts the decode thread with a freshly shuffled
    segment order without recreating the object.

    This design guarantees that every sequence is served exactly once per
    epoch with equal probability.
    """

    def __init__(
        self,
        provider,
        seg_ids: List[str],
        seq_len: int,
        num_cameras: int,
        buffer_frames: int,
        min_buffer_seg: int,
        mean: np.ndarray,
        std: np.ndarray,
        worker_id: int = 0,
        prefetch_queue_depth: int = 0,
    ) -> None:
        self._provider        = provider
        self._seg_ids         = list(seg_ids)
        self._cursor          = 0
        self._seq_len         = seq_len
        self._num_cameras     = num_cameras
        self._buffer_frames   = buffer_frames
        self._min_buffer_seg  = max(1, min_buffer_seg)
        self._worker_id       = worker_id
        _depth = prefetch_queue_depth if prefetch_queue_depth > 0 else max(1, min_buffer_seg)
        self._prefetch_queue_depth = _depth

        self._mean = torch.as_tensor(mean.reshape(3, 1, 1), dtype=torch.float32)
        self._std  = torch.as_tensor(std.reshape(3, 1, 1),  dtype=torch.float32)

        self._pool: List[Dict[str, Any]] = []
        self._pool_frames = 0

        # Background decode thread fills this queue; main thread drains on
        # each next_window() call.  Depth = prefetch_queue_depth (default:
        # min_buffer_seg) so the decode thread can stage one full round-robin
        # cycle of chunks before blocking, reducing stall latency spikes.
        self._prefetch_q: queue.Queue = queue.Queue(maxsize=self._prefetch_queue_depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._decode_loop, daemon=True, name=f"buf-decode-w{worker_id}"
        )
        self._thread.start()

        self._wait_for_initial_fill()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_window(self) -> Dict[str, Any]:
        """Return one window sampled proportionally to each entry's remaining count.

        If the pool is temporarily empty but the decode thread is still running,
        blocks until the next sequence arrives.  Raises ``PoolExhausted`` when
        the decode pass is complete *and* the pool is drained (epoch done).
        """
        self._drain_prefetch()

        if not self._pool:
            # Pool temporarily empty — block until the decode thread delivers
            # the next sequence, or confirm the epoch is truly over.
            while not self._pool:
                try:
                    seq_data = self._prefetch_q.get(timeout=5.0)
                    self._add_to_pool(seq_data)
                except queue.Empty:
                    if not self._thread.is_alive() and self._prefetch_q.empty():
                        raise PoolExhausted

        # Weighted sample: probability proportional to remaining window count.
        weights = [len(e["pending"]) for e in self._pool]
        (entry,) = random.choices(self._pool, weights=weights, k=1)
        start = entry["pending"].pop()

        # Natural eviction once all windows of a sequence are served.
        if not entry["pending"]:
            self._pool = [e for e in self._pool if e is not entry]
            self._pool_frames -= entry["frames"]

        return self._assemble_window(entry["data"], start)

    def reset(self, seg_ids: List[str]) -> None:
        """Prepare for a new epoch with a freshly shuffled segment order.

        Stops the old decode thread, resets pool state, and starts a new
        decode thread.  Blocks until the initial diversity target is reached.
        """
        # Stop the old thread and drain the queue so any blocked put() can exit.
        self._stop.set()
        while True:
            try:
                self._prefetch_q.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=10.0)

        # Reset state for the new epoch.
        self._pool.clear()
        self._pool_frames = 0
        self._seg_ids = list(seg_ids)
        self._cursor = 0
        self._stop = threading.Event()

        # Restart the decode thread with the new segment order.
        self._thread = threading.Thread(
            target=self._decode_loop,
            daemon=True,
            name=f"buf-decode-w{self._worker_id}",
        )
        self._thread.start()
        self._wait_for_initial_fill(label="epoch reset")

    def close(self) -> None:
        """Signal the background thread to stop (called on worker shutdown)."""
        self._stop.set()
        while True:
            try:
                self._prefetch_q.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_initial_fill(self, label: str = "initial fill") -> None:
        """Block until the pool has enough segment chunks for diversity.

        Waits until ``min(min_buffer_seg, len(seg_ids))`` distinct segments
        are represented in the pool.  Logs timing so slow drives are visible.
        """
        target = max(1, min(self._min_buffer_seg, len(self._seg_ids)))

        def _active() -> int:
            return len({e["data"]["seq_id"] for e in self._pool})

        if _active() >= target:
            return
        t0 = time.monotonic()
        print(f"[buf w{self._worker_id}] waiting for {label} "
              f"(target {target} segments from {len(self._seg_ids)} in shard)…",
              flush=True)
        while _active() < target:
            try:
                seg_chunk = self._prefetch_q.get(timeout=120.0)
            except queue.Empty:
                if not self._thread.is_alive():
                    break
                print(f"[buf w{self._worker_id}] still waiting after 120 s…",
                      flush=True)
                continue
            self._add_to_pool(seg_chunk)
        elapsed = time.monotonic() - t0
        print(f"[buf w{self._worker_id}] {label} done in {elapsed:.1f}s  "
              f"pool={_active()} segs / {self._pool_frames} frames  "
              f"pending_seqs={sum(len(e['pending']) for e in self._pool)}",
              flush=True)

    def _add_to_pool(self, seg_chunk: Dict[str, Any]) -> None:
        """Add a segment chunk to the pool with a pre-shuffled sequence list."""
        if seg_chunk["T"] < self._seq_len:
            return
        starts = list(range(seg_chunk["T"] - self._seq_len + 1))
        random.shuffle(starts)
        entry = {
            "data":    seg_chunk,
            "pending": starts,
            "frames":  seg_chunk["T"] * self._num_cameras,
        }
        self._pool.append(entry)
        self._pool_frames += entry["frames"]

    def _decode_loop(self) -> None:
        """Background thread: one full pass through all segments, then stop.

        Maintains a round-robin pool of ``min_buffer_seg`` active segment
        iterators.  On each tick it advances every active iterator by one
        chunk (of exactly ``chunk_frames`` temporal frames), putting the
        result into the prefetch queue.  When an iterator is exhausted the
        slot is replaced by the next unread segment from the cursor.

        This design guarantees:

        * **Diversity**: at any point chunks in the queue come from
          ``min_buffer_seg`` distinct segments, so no single large segment
          can monopolise the buffer.
        * **Completeness**: iterators run to exhaustion — every chunk from
          every segment is decoded exactly once per epoch.
        * **Balance**: every active segment contributes equal-sized chunks,
          so frame budgets are distributed evenly across segments.
        """
        # Temporal frames per chunk.  Sized so that one chunk from each of
        # min_buffer_seg active segments fills exactly buffer_frames.
        chunk_frames = max(
            self._buffer_frames // (max(1, self._min_buffer_seg) * self._num_cameras),
            self._seq_len,  # guarantee at least one valid sequence window per chunk
        )

        _ok = 0
        _fail = 0

        def _open_next(into: list) -> bool:
            """Open the next unread segment and append its iterator to *into*.

            Returns True when a new iterator was successfully opened.
            """
            nonlocal _fail
            while self._cursor < len(self._seg_ids):
                seg_id = self._seg_ids[self._cursor]
                self._cursor += 1
                try:
                    gen = self._provider.load_full_sequence(
                        seg_id, self._num_cameras, chunk_frames
                    )
                    into.append((seg_id, iter(gen)))
                    return True
                except Exception as _e:
                    _fail += 1
                    if _fail <= 5 or _fail % 50 == 0:
                        print(
                            f"[buf w{self._worker_id}] open failed for {seg_id!r}: "
                            f"{type(_e).__name__}: {_e}",
                            flush=True,
                        )
            return False

        # Seed round-robin pool with up to min_buffer_seg active iterators.
        active: list[tuple[str, Any]] = []
        for _ in range(self._min_buffer_seg):
            if not _open_next(into=active):
                break

        while not self._stop.is_set():
            if not active:
                break

            next_active: list[tuple[str, Any]] = []
            for seg_id, it in active:
                if self._stop.is_set():
                    return

                t_chunk = time.monotonic()
                try:
                    data = next(it)
                except StopIteration:
                    # Segment fully decoded; replace with next unread segment.
                    _open_next(into=next_active)
                    continue
                except Exception as _e:
                    _fail += 1
                    if _fail <= 5 or _fail % 50 == 0:
                        print(
                            f"[buf w{self._worker_id}] decode error for {seg_id!r}: "
                            f"{type(_e).__name__}: {_e}",
                            flush=True,
                        )
                    _open_next(into=next_active)
                    continue

                elapsed = time.monotonic() - t_chunk
                if elapsed > 30.0:
                    print(
                        f"[buf w{self._worker_id}] slow chunk: {seg_id!r} "
                        f"took {elapsed:.1f}s",
                        flush=True,
                    )

                if data["T"] >= self._seq_len:
                    self._prefetch_q.put(data)   # blocks if full (back-pressure)
                    _ok += 1
                else:
                    _fail += 1

                next_active.append((seg_id, it))

            active = next_active

        print(
            f"[buf w{self._worker_id}] decode pass done  "
            f"ok={_ok}  fail={_fail}  "
            f"pool={len(self._pool)} segment chunks remaining",
            flush=True,
        )

    def _drain_prefetch(self) -> None:
        """Move decoded segment chunks from the prefetch queue into the pool.

        Drains until BOTH conditions are satisfied:
        * ``pool_frames >= buffer_frames``  (frame-budget constraint)
        * at least ``min_buffer_seg`` distinct segments in pool (diversity constraint)

        Either condition alone does not stop draining; the background thread
        naturally blocks on ``queue.put`` when both are met.
        """
        while True:
            active_segs = len({e["data"]["seq_id"] for e in self._pool})
            frames_ok = self._pool_frames >= self._buffer_frames
            segs_ok   = active_segs >= self._min_buffer_seg
            if frames_ok and segs_ok:
                break
            try:
                seg_chunk = self._prefetch_q.get_nowait()
                self._add_to_pool(seg_chunk)
            except queue.Empty:
                break

    def _assemble_window(self, seq_data: Dict[str, Any], start: int) -> Dict[str, Any]:
        """Slice a temporal window and return raw uint8 images.

        Float normalisation is deferred to the training/eval loop and applied
        on the GPU after batch collation, reducing IPC transfer size 4×.
        """
        s = start
        e = start + self._seq_len

        # Clone the slice so the pool chunk can be freed once evicted.
        # imgs remains uint8 (T, N, 3, H, W) — normalisation happens on GPU.
        imgs_u8 = seq_data["imgs"][s:e].clone()

        sample = {
            "imgs":         imgs_u8,
            "world2img":    torch.from_numpy(seq_data["world2img"][s:e]),
            "intrinsics":   torch.from_numpy(seq_data["intrinsics"][s:e]),
            "gt_keypoints": torch.from_numpy(seq_data["gt_keypoints"][s:e]),
            "gt_valid":     torch.from_numpy(seq_data["gt_valid"][s:e]),
            "seq":          seq_data["seq_id"],
            "start":        start,
        }
        for key in ("gt_keypoint_frame_valid", "gt_keypoint_valid", "gt_root", "gt_root_valid"):
            if key in seq_data:
                sample[key] = torch.from_numpy(seq_data[key][s:e])
        return sample
