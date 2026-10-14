"""Dataloader throughput and memory benchmark.

Usage
-----
    python tools/benchmark_dataloader.py configs/pipe_cleaner.yaml [options]

Options
-------
    --batches N      Number of batches to consume (default: 100)
    --warmup N       Batches to skip before measuring (default: 5)
    --split SPLIT    Dataset split to benchmark: train | val (default: train)
    --workers N      Override cfg.data.num_workers
    --no-shuffle     Disable shuffle buffer even if enabled in config

Report includes
---------------
  - Batches measured / total consumed
  - Batch shape (B, T, N, H, W)
  - RPS  – rows (camera-frames) per second
  - SPS  – samples (windows) per second
  - BPS  – batches per second
  - Batch latency: mean / p50 / p95 / p99 / max  (ms)
  - Host RSS memory: baseline / sustained mean / peak  (MB)
  - CUDA memory (if available): reserved / allocated peak  (MB)
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# Ensure project root and FFMPEG DLLs are on the path before any imports.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_ffmpeg_bin = os.environ.get("FFMPEG_SHARED_BIN", "")
if sys.platform == "win32" and _ffmpeg_bin and os.path.isdir(_ffmpeg_bin):
    os.add_dll_directory(_ffmpeg_bin)

import torch  # noqa: E402

from unified_perception.config import load_config  # noqa: E402
from unified_perception.engine.train import build_loaders  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / 1_048_576


def _cuda_peak_mb() -> tuple[float, float]:
    """Return (reserved_MB, allocated_MB) peak since last reset, or (0, 0)."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_reserved() / 1_048_576,
        torch.cuda.max_memory_allocated() / 1_048_576,
    )


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _hbar(label: str, value: str, width: int = 28) -> str:
    return f"  {label:<{width}} {value}"


def _section(title: str) -> str:
    return f"\n{'─' * 60}\n  {title}\n{'─' * 60}"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    cfg_path: str,
    *,
    n_batches: int = 1000,
    n_warmup: int = 25,
    split: str = "train",
    num_workers_override: int | None = None,
    disable_shuffle: bool = False,
    batch_size_override: int | None = None,
    buffer_frames_override: int | None = None,
    queue_depth_override: int | None = None,
) -> None:
    print(f"\nLoading config: {cfg_path}")
    cfg = load_config(cfg_path)

    if num_workers_override is not None:
        cfg.data.num_workers = num_workers_override
    if batch_size_override is not None:
        cfg.data.batch_size = batch_size_override
    if buffer_frames_override is not None:
        cfg.data.buffer_frames = buffer_frames_override
    if queue_depth_override is not None:
        cfg.data.prefetch_queue_depth = queue_depth_override

    if disable_shuffle:
        cfg.data.use_shuffle_buffer = False

    print(f"  provider         : {cfg.data.provider}")
    print(f"  split            : {split}")
    print(f"  batch_size       : {cfg.data.batch_size}")
    print(f"  num_workers      : {cfg.data.num_workers}")
    print(f"  use_shuffle_buf  : {cfg.data.use_shuffle_buffer}")
    print(f"  buffer_frames    : {getattr(cfg.data, 'buffer_frames', 'n/a')}")
    print(f"  min_buffer_seg   : {getattr(cfg.data, 'min_buffer_seg', 'n/a')}")
    _qd = getattr(cfg.data, 'prefetch_queue_depth', 0)
    _qd_eff = _qd if _qd > 0 else getattr(cfg.data, 'min_buffer_seg', 1)
    print(f"  prefetch_q_depth : {_qd_eff}  (cfg={_qd})")
    print(f"  seq_len          : {cfg.data.seq_len}")
    print(f"  num_cameras      : {cfg.data.num_cameras}")
    print(f"  warmup batches   : {n_warmup}")
    print(f"  measure batches  : {n_batches}")

    # Build loader ------------------------------------------------------------
    print("\nBuilding data loaders …")
    t_build_start = time.perf_counter()
    loaders = build_loaders(cfg)
    loader = loaders[split]
    t_build = time.perf_counter() - t_build_start
    print(f"  build_loaders()  : {t_build:.2f}s")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rss_baseline = _rss_mb()
    print(f"  RSS baseline     : {rss_baseline:.1f} MB")

    # Warm-up -----------------------------------------------------------------
    print(f"\nWarm-up ({n_warmup} batches) …", flush=True)
    loader_iter = iter(loader)
    for _ in range(n_warmup):
        try:
            next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)

    # Measurement loop --------------------------------------------------------
    print(f"Measuring ({n_batches} batches) …", flush=True)

    latencies_ms: list[float] = []
    rss_samples: list[float] = []
    batch_shape: tuple | None = None

    total_windows = 0    # samples (training windows)
    total_cam_frames = 0 # camera-frames

    t_total_start = time.perf_counter()
    t_prev = t_total_start

    for i in range(n_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        t_now = time.perf_counter()
        lat_ms = (t_now - t_prev) * 1000.0
        latencies_ms.append(lat_ms)
        t_prev = t_now

        imgs = batch["imgs"]   # (B, T, N, 3, H, W)
        B, T, N = imgs.shape[:3]
        total_windows    += B
        total_cam_frames += B * T * N

        if batch_shape is None:
            batch_shape = tuple(imgs.shape)

        if i % 10 == 0:
            rss_samples.append(_rss_mb())

    t_total = time.perf_counter() - t_total_start

    rss_sustained_mean = statistics.mean(rss_samples) if rss_samples else 0.0
    rss_peak = max(rss_samples) if rss_samples else 0.0
    cuda_rsvd_peak, cuda_alloc_peak = _cuda_peak_mb()

    # Derived metrics ---------------------------------------------------------
    bps  = n_batches / t_total
    sps  = total_windows / t_total       # training samples/sec
    rps  = total_cam_frames / t_total    # camera-frames/sec

    lat_mean = statistics.mean(latencies_ms)
    lat_p50  = _percentile(latencies_ms, 50)
    lat_p95  = _percentile(latencies_ms, 95)
    lat_p99  = _percentile(latencies_ms, 99)
    lat_max  = max(latencies_ms)
    lat_min  = min(latencies_ms)

    # Report ------------------------------------------------------------------
    print(_section("THROUGHPUT"))
    print(_hbar("batches measured",     f"{n_batches}"))
    print(_hbar("batch shape (B,T,N,C,H,W)", str(batch_shape)))
    print(_hbar("total time",           f"{t_total:.2f}s"))
    print(_hbar("batches / sec",        f"{bps:.2f}"))
    print(_hbar("samples / sec (SPS)",  f"{sps:.1f}"))
    print(_hbar("cam-frames / sec (RPS)", f"{rps:.1f}"))

    print(_section("BATCH LATENCY"))
    print(_hbar("mean",  f"{lat_mean:.1f} ms"))
    print(_hbar("min",   f"{lat_min:.1f} ms"))
    print(_hbar("p50",   f"{lat_p50:.1f} ms"))
    print(_hbar("p95",   f"{lat_p95:.1f} ms"))
    print(_hbar("p99",   f"{lat_p99:.1f} ms"))
    print(_hbar("max",   f"{lat_max:.1f} ms"))

    print(_section("HOST MEMORY (RSS)"))
    print(_hbar("baseline",       f"{rss_baseline:.1f} MB"))
    print(_hbar("sustained mean", f"{rss_sustained_mean:.1f} MB  (+{rss_sustained_mean - rss_baseline:.1f})"))
    print(_hbar("peak",           f"{rss_peak:.1f} MB  (+{rss_peak - rss_baseline:.1f})"))

    if torch.cuda.is_available():
        print(_section("CUDA MEMORY"))
        print(_hbar("peak reserved",  f"{cuda_rsvd_peak:.1f} MB"))
        print(_hbar("peak allocated", f"{cuda_alloc_peak:.1f} MB"))

    print(f"\n{'═' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark dataloader RPS, latency, and memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", help="Path to a YAML config file.")
    p.add_argument("--batches",  type=int, default=1000,
                   help="Number of batches to measure.")
    p.add_argument("--warmup",   type=int, default=50,
                   help="Batches to consume before measurement starts.")
    p.add_argument("--split",    default="train", choices=["train", "val"],
                   help="Dataset split to use.")
    p.add_argument("--workers",  type=int, default=None,
                   help="Override cfg.data.num_workers.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override cfg.data.batch_size.")
    p.add_argument("--buffer-frames", type=int, default=None,
                   help="Override cfg.data.buffer_frames.")
    p.add_argument("--queue-depth", type=int, default=None,
                   help="Override prefetch_queue_depth (0=auto=min_buffer_seg).")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Force-disable the shuffle buffer.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        args.config,
        n_batches=args.batches,
        n_warmup=args.warmup,
        split=args.split,
        num_workers_override=args.workers,
        disable_shuffle=args.no_shuffle,
        batch_size_override=args.batch_size,
        buffer_frames_override=args.buffer_frames,
        queue_depth_override=args.queue_depth,
    )
