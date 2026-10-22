"""
Test cases:

First: terminology.
We refer to a "segment" as a complete sequence of frames from a set of videos (each element of the set is a single camera's video, all synchronized).
A "segment chunk" is a contiguous sub-segment of frames that are cut for optimization purposes (to fit in memory, to speed decoding, etc.)
A "sequence" is a temporal window of frames that is used as a training sample. A sequence is drawn from a "segment chunk". Sequences can overlap with each other.
A frame is a single timestep of a sequence

1) Shuffle buffer always initially loads "buffer_frames" number of frames. When frames are discarded, the buffer is refilled with new frames up to "buffer_frames". The test checks that the buffer is correctly filled and that the number of frames in the buffer is as expected.
2) Shuffle buffer always loads frames from "min_buffer_seg" number of unique segments. It attempts to keep the number of frames from each segment equal. I.e. if buffer_frames=16 and min_buffer_seg=4, then the buffer should contain 4 frames from each of 4 unique segments. This can be violated only when this is not possible (e.g. there are fewer remaining segments than min_buffer_seg, or there are not enough frames left in the remaining segments to fill the buffer).
3) All sequences in the buffer are returned with equal probability (they are shuffled). After it has been returned, that sequence won't be returned again (sampling without replacement).
4) Once all sequences for a given segment chunk have been returned, the segment chunk is discarded and the buffer is refilled with new frames from new segment chunks.
5) The buffer is refilled with new frames from new segment chunks, and the new segment chunks are drawn randomly from the remaining segment chunks that have not yet been used.
6) Any given segment chunk is only used once. Once all segments are exhausted, the dataset is empty and raises StopIteration.
7) The dataset only keeps in memory the frames that are needed to fill the buffer and the metadata needed to track which frames/sequences/segment chunks have been used. It does not keep all frames in memory.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List

import numpy as np
import pytest
import torch

from unified_perception.data.shuffle_buffer_dataset import PoolExhausted, SequenceBuffer


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockProvider:
    """Fake provider that returns deterministic tiny-image segment chunks.

    Parameters
    ----------
    segments : dict mapping segment_id -> total frame count.
        E.g. {"seg_0": 14} means segment "seg_0" has 14 frames total.
        Chunks are split at ``chunk_frames`` boundaries by load_full_sequence.
    """

    def __init__(
        self,
        segments: Dict[str, int],
        num_joints: int = 1,
        img_h: int = 4,
        img_w: int = 4,
    ) -> None:
        self._segments   = segments
        self._num_joints = num_joints
        self._img_h      = img_h
        self._img_w      = img_w
        # Attribute accessed by ShuffleBufferDataset.__len__
        self._seqs = {
            sid: type("Seq", (), {"num_frames": total})()
            for sid, total in segments.items()
        }
        # Incremented when load_full_sequence is called (not on first yield).
        self.decode_calls: Dict[str, int] = {}

    def load_full_sequence(
        self, seg_id: str, num_cameras: int, chunk_frames: int
    ) -> Iterator[Dict[str, Any]]:
        """Increment decode_calls eagerly, then return a chunk generator."""
        self.decode_calls[seg_id] = self.decode_calls.get(seg_id, 0) + 1
        return self._gen_chunks(seg_id, num_cameras, chunk_frames)

    def _gen_chunks(
        self, seg_id: str, num_cameras: int, chunk_frames: int
    ) -> Iterator[Dict[str, Any]]:
        total = self._segments[seg_id]
        N, H, W = num_cameras, self._img_h, self._img_w
        for start in range(0, total, chunk_frames):
            T = min(chunk_frames, total - start)
            yield {
                "imgs":         torch.zeros(T, N, 3, H, W, dtype=torch.uint8),
                "world2img":    np.zeros((T, N, 4, 4), dtype=np.float32),
                "intrinsics":   np.zeros((T, N, 4, 4), dtype=np.float32),
                "gt_keypoints": np.zeros((T, 1, self._num_joints, 3), dtype=np.float32),
                "gt_valid":     np.ones((T, 1), dtype=bool),
                "seq_id":       seg_id,
                "T":            T,
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buf(
    segments: Dict[str, int],
    seq_len: int = 2,
    num_cameras: int = 2,
    buffer_frames: int = 32,
    min_buffer_seg: int = 2,
) -> tuple[SequenceBuffer, MockProvider]:
    """Create a SequenceBuffer backed by a MockProvider."""
    provider = MockProvider(segments)
    buf = SequenceBuffer(
        provider=provider,
        seg_ids=list(segments.keys()),
        seq_len=seq_len,
        num_cameras=num_cameras,
        buffer_frames=buffer_frames,
        min_buffer_seg=min_buffer_seg,
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
        worker_id=0,
    )
    return buf, provider


def _drain_all(buf: SequenceBuffer) -> List[Dict[str, Any]]:
    """Consume every sequence from buf; return all items."""
    items: List[Dict[str, Any]] = []
    while True:
        try:
            items.append(buf.next_window())
        except PoolExhausted:
            break
    return items


def _wait_for_decode(buf: SequenceBuffer, timeout: float = 10.0) -> None:
    """Keep draining the prefetch queue until both fill conditions are met or
    the decode thread is fully done — whichever comes first.

    Uses repeated drain + sleep instead of a single get_nowait() call so we
    don't race against the decode thread putting items into the queue.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        buf._drain_prefetch()
        frames_ok = buf._pool_frames >= buf._buffer_frames
        segs_ok = (
            len({e["data"]["seq_id"] for e in buf._pool}) >= buf._min_buffer_seg
        )
        if frames_ok and segs_ok:
            break
        if not buf._thread.is_alive() and buf._prefetch_q.empty():
            buf._drain_prefetch()   # one final sweep
            break
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Test 1 – Initial fill
# ---------------------------------------------------------------------------

def test_initial_fill_reaches_buffer_frames():
    """Buffer fills to >= buffer_frames camera-frames during construction."""
    segments = {f"seg_{i}": 8 for i in range(10)}
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=40, min_buffer_seg=1)
    _wait_for_decode(buf)

    total_available = sum(segments.values()) * 2
    assert buf._pool_frames >= min(40, total_available), (
        f"pool_frames={buf._pool_frames} < expected {min(40, total_available)}"
    )


def test_initial_fill_loads_all_when_buffer_exceeds_total():
    """If buffer_frames > total available frames, all frames are loaded."""
    segments = {"seg_a": 4, "seg_b": 4}   # 16 camera-frames total
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=1000, min_buffer_seg=1)
    _wait_for_decode(buf)

    assert buf._pool_frames == 16


# ---------------------------------------------------------------------------
# Test 2 – min_buffer_seg diversity
# ---------------------------------------------------------------------------

def test_min_buffer_seg_loads_enough_distinct_segments():
    """Pool contains frames from at least min_buffer_seg distinct segments."""
    segments = {f"seg_{i}": 6 for i in range(8)}
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=24, min_buffer_seg=4)
    _wait_for_decode(buf)

    active_segs = {e["data"]["seq_id"] for e in buf._pool}
    assert len(active_segs) >= min(4, len(segments)), (
        f"Expected >=4 segments in pool, got {active_segs}"
    )


def test_min_buffer_seg_violated_gracefully_when_too_few_available():
    """When fewer segments exist than min_buffer_seg, all are loaded without hanging."""
    segments = {"seg_0": 5, "seg_1": 5}
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=20, min_buffer_seg=6)
    _wait_for_decode(buf)

    active_segs = {e["data"]["seq_id"] for e in buf._pool}
    assert len(active_segs) == 2


def test_min_buffer_seg_large_segment_does_not_flood_buffer():
    """A segment with many chunks contributes at most ~buffer_frames/min_buffer_seg camera-frames.

    With min_buffer_seg=4 and buffer_frames=32, the per-segment quota is 8 cam-frames.
    Even if "large" has 20 chunks the buffer should still contain all 4 distinct segments.
    """
    num_cameras = 2
    # "large": 20 chunks × 4 frames × 2 cameras = 160 cam-frames if fully read.
    # Small segments each contribute exactly one chunk of 4 frames = 8 cam-frames.
    segments = {
        "large": 80,    # 20 × 4-frame chunks when chunk_frames=4
        "seg_b": 4,
        "seg_c": 4,
        "seg_d": 4,
    }
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=num_cameras,
                       buffer_frames=32, min_buffer_seg=4)
    _wait_for_decode(buf)

    # All 4 distinct segments must be present in the pool.
    active_segs = {e["data"]["seq_id"] for e in buf._pool}
    assert len(active_segs) == 4, (
        f"Expected 4 segments in pool after fill, got {active_segs}"
    )

    # "large" should NOT have consumed more than quota + one_chunk overhead.
    quota = 32 // 4  # = 8 cam-frames
    one_chunk = 4 * num_cameras  # = 8 cam-frames
    large_frames = sum(
        e["frames"] for e in buf._pool if e["data"]["seq_id"] == "large"
    )
    assert large_frames <= quota + one_chunk, (
        f"'large' segment contributed {large_frames} cam-frames "
        f"(max allowed {quota + one_chunk})"
    )


    """When chunk sizes equal buffer_frames/min_buffer_seg, each segment is equally represented."""
    # 4 segments × 4 frames × 2 cameras = 8 camera-frames each = 32 total
    # buffer_frames=32, min_buffer_seg=4 → each contributes exactly 8 cam-frames
    segments = {f"seg_{i}": 4 for i in range(4)}
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=32, min_buffer_seg=4)
    _wait_for_decode(buf)

    frames_per_seg = {}
    for e in buf._pool:
        sid = e["data"]["seq_id"]
        frames_per_seg[sid] = frames_per_seg.get(sid, 0) + e["frames"]

    assert len(frames_per_seg) == 4
    for sid, frames in frames_per_seg.items():
        assert frames == 8, f"Segment {sid}: expected 8 camera-frames, got {frames}"


# ---------------------------------------------------------------------------
# Test 3 – Without replacement
# ---------------------------------------------------------------------------

def test_without_replacement_single_chunk():
    """Each valid sequence start is returned exactly once before PoolExhausted."""
    # T=6, seq_len=2 → valid starts [0,1,2,3,4]
    segments = {"seg_0": 6}
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=100, min_buffer_seg=1)

    items = _drain_all(buf)
    starts = [it["start"] for it in items]
    assert sorted(starts) == list(range(5)), (
        f"Expected starts [0..4], got {sorted(starts)}"
    )


def test_without_replacement_multiple_segments():
    """No (segment, start) pair is returned twice across multiple segments."""
    # 3 segments × 5 frames × seq_len=2 → 4 sequences each = 12 total
    segments = {f"seg_{i}": 5 for i in range(3)}
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=100, min_buffer_seg=1)

    items = _drain_all(buf)
    pairs = [(it["seq"], it["start"]) for it in items]
    assert len(pairs) == len(set(pairs)), "Duplicate (segment, start) pairs returned"
    assert len(pairs) == 12


def test_pool_exhausted_raised_when_done():
    """PoolExhausted is raised on the call after the last sequence."""
    segments = {"seg_0": 3}   # seq_len=2 → exactly 2 sequences
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=100, min_buffer_seg=1)

    buf.next_window()
    buf.next_window()

    with pytest.raises(PoolExhausted):
        buf.next_window()


# ---------------------------------------------------------------------------
# Test 4 – Segment chunk eviction and buffer refill
# ---------------------------------------------------------------------------

def test_exhausted_chunk_evicted_from_pool():
    """A segment chunk is removed from pool once all its sequences are consumed."""
    segments = {"seg_0": 3, "seg_1": 10}   # seg_0: 2 seqs; seg_1: 9 seqs
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=100, min_buffer_seg=1)
    _wait_for_decode(buf)

    consumed = 0
    while consumed < 2:
        if buf.next_window()["seq"] == "seg_0":
            consumed += 1

    pool_segs = {e["data"]["seq_id"] for e in buf._pool}
    assert "seg_0" not in pool_segs, (
        f"Exhausted chunk 'seg_0' still in pool: {pool_segs}"
    )


def test_buffer_refills_from_remaining_segments():
    """After one chunk is exhausted, the buffer pulls in a new one from remaining segments."""
    # 5 segments × one 6-frame chunk; buffer holds only ~1 chunk (12 cam-frames)
    segments = {f"seg_{i}": 6 for i in range(5)}
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=12, min_buffer_seg=1)

    segs_seen = {it["seq"] for it in _drain_all(buf)}
    assert segs_seen == set(segments.keys()), (
        f"Not all segments served: missing {set(segments.keys()) - segs_seen}"
    )


# ---------------------------------------------------------------------------
# Test 5 – New chunks drawn from remaining unused segments only
# ---------------------------------------------------------------------------

def test_refill_only_uses_unused_segments():
    """Segments loaded during refill are always from the not-yet-decoded pool."""
    segments = {f"seg_{i}": 6 for i in range(8)}
    buf, provider = _make_buf(segments, seq_len=2, num_cameras=2,
                               buffer_frames=12, min_buffer_seg=1)
    _drain_all(buf)

    for seg_id in segments:
        n = provider.decode_calls.get(seg_id, 0)
        assert n == 1, f"Segment {seg_id!r} decoded {n} times (expected 1)"


# ---------------------------------------------------------------------------
# Test 6 – Each segment chunk used exactly once
# ---------------------------------------------------------------------------

def test_each_segment_decoded_once():
    """load_full_sequence is called exactly once per segment during an epoch.

    Uses segments larger than chunk_frames so each segment yields multiple
    chunks, verifying that repeated next() calls share the same generator
    opened by a single load_full_sequence invocation.
    """
    # chunk_frames = max(100 // (1*2), 2) = 50; segments have 100 frames → 2 chunks each.
    segments = {f"seg_{i}": 100 for i in range(5)}
    buf, provider = _make_buf(segments, seq_len=2, num_cameras=2,
                               buffer_frames=100, min_buffer_seg=1)
    _drain_all(buf)

    for seg_id in segments:
        n = provider.decode_calls.get(seg_id, 0)
        assert n == 1, f"Segment {seg_id!r} decoded {n} times (expected 1)"


def test_total_sequences_matches_sum_over_chunks():
    """Total sequences returned equals the sum of valid windows across all chunks.

    Uses multi-chunk segments (total_frames > chunk_frames) to verify that
    every chunk from every segment is decoded and served exactly once.
    """
    # chunk_frames = max(200 // (1*2), 2) = 100
    # seg_a: 110 frames → chunks [100, 10] → 99 + 9 = 108 seqs
    # seg_b:  50 frames → chunks [50]      → 49 seqs
    # seg_c: 200 frames → chunks [100,100] → 99 + 99 = 198 seqs
    buf_frames, num_cam, seq_len_ = 200, 2, 2
    chunk_frames = max(buf_frames // (1 * num_cam), seq_len_)  # = 100
    segments = {"seg_a": 110, "seg_b": 50, "seg_c": 200}

    def _expected(total: int) -> int:
        n = 0
        for start in range(0, total, chunk_frames):
            T = min(chunk_frames, total - start)
            if T >= seq_len_:
                n += T - seq_len_ + 1
        return n

    expected = sum(_expected(T) for T in segments.values())  # 108 + 49 + 198 = 355

    buf, _ = _make_buf(segments, seq_len=seq_len_, num_cameras=num_cam,
                       buffer_frames=buf_frames, min_buffer_seg=1)

    items = _drain_all(buf)
    assert len(items) == expected, f"Expected {expected} sequences, got {len(items)}"


def test_pool_exhausted_after_all_segments_used():
    """Once all segment chunks are consumed, the next call raises PoolExhausted."""
    segments = {"seg_0": 2}   # T=seq_len=2 → exactly 1 sequence
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=100, min_buffer_seg=1)

    buf.next_window()

    with pytest.raises(PoolExhausted):
        buf.next_window()


# ---------------------------------------------------------------------------
# Test 7 – Memory efficiency
# ---------------------------------------------------------------------------

def test_pool_frames_bounded_during_consumption():
    """pool_frames stays within buffer_frames + one chunk slack at all times."""
    seg_frames, cameras, buffer_frames = 10, 2, 40
    # Internal chunk_frames = max(buffer_frames // (min_buffer_seg * cameras), seq_len)
    #   = max(40 // (1*2), 2) = 20.  Each segment (10 frames) yields one chunk of
    #   10 frames = 20 cam-frames.  Allow one extra chunk of slack.
    cam_frames_per_chunk = seg_frames * cameras   # = 20
    max_allowed = buffer_frames + cam_frames_per_chunk   # = 60

    segments = {f"seg_{i}": seg_frames for i in range(20)}
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=cameras,
                       buffer_frames=buffer_frames, min_buffer_seg=1)

    peak = 0
    for item in _drain_all(buf):
        peak = max(peak, buf._pool_frames)

    assert peak <= max_allowed, (
        f"pool_frames peaked at {peak}, max allowed {max_allowed}"
    )


def test_evicted_chunks_not_retained_in_pool():
    """Fully-consumed segment chunks are not kept in the pool."""
    segments = {
        "small": 3,    # 2 sequences; exhausted quickly
        "large": 20,   # 19 sequences; stays active
    }
    buf, _ = _make_buf(segments, seq_len=2, num_cameras=2,
                       buffer_frames=100, min_buffer_seg=1)
    _wait_for_decode(buf)

    consumed_small = 0
    while consumed_small < 2:
        if buf.next_window()["seq"] == "small":
            consumed_small += 1

    pool_segs = {e["data"]["seq_id"] for e in buf._pool}
    assert "small" not in pool_segs
    assert "large" in pool_segs