"""Benchmark the real training step on a fixed set of cached batches.

Measures what the trainer does per step (h2d, forward, loss, backward, optimizer) without
data-loading noise: the first ``--batches`` training batches are taken from the real streaming
loader once (consecutive, so lanes keep their track/memory continuity), saved under
``data_cache/bench_batches/``, and replayed.

Reports:
  * seconds per step (wall, GPU synchronized at the end of the timed block);
  * GPU busy fraction (CUDA kernel time / wall, torch.profiler) with ``--profile``;
  * every host<->device synchronization, by source line, with ``--sync-debug``
    (torch.cuda.set_sync_debug_mode), i.e. the places where the CPU waits for the GPU.

Usage:
    python -m unified_perception.tools.bench_step configs/gym_sixcam_track_v1.yaml --steps 20 --sync-debug --profile
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

from ..config import config_to_dict, load_config
from ..engine.step import TrainStep, _h2d_async
from ..engine.train import build_loaders, resolve_device, set_seed
from ..models import SetCriterion, UnifiedPerceptionModel

PACKAGE = Path(__file__).resolve().parents[1]


def cached_batches(cfg, count: int, root: Path) -> list[dict]:
    d = config_to_dict(cfg)["data"]
    key = hashlib.sha1(json.dumps({"data": d, "seed": cfg.train.seed, "n": count}, sort_keys=True,
                                  default=str).encode()).hexdigest()[:12]
    path = root / f"batches_{key}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    loader = build_loaders(cfg)["train"]
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= count:
            break
    root.mkdir(parents=True, exist_ok=True)
    torch.save(batches, path)
    return batches


def _gpm_means(path: str) -> str:
    """Means of the GPM columns of an nvidia-smi dmon log (the first sample is a warm-up)."""
    names = ["engine", "sm_active", "sm_occupancy", "tensor", "hmma", "dram", "fp32", "fp16"]
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) >= 8 and cols[-1] != "-":
            rows.append([float(c) for c in cols[-8:]])
    rows = rows[1:] or rows
    if not rows:
        return "no samples"
    return ", ".join(f"{n} {sum(r[i] for r in rows) / len(rows):.1f}%" for i, n in enumerate(names)) +         f" ({len(rows)} samples)"


def _site(stack) -> str:
    """Innermost frame inside the package (where the synchronizing call is made)."""
    for frame in reversed(stack):
        if str(PACKAGE) in frame.filename:
            return f"{Path(frame.filename).relative_to(PACKAGE.parent)}:{frame.lineno} {frame.line}"
    return f"{stack[-1].filename}:{stack[-1].lineno}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sync-debug", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="optional weights (e.g. a run's last.pt)")
    ap.add_argument("--cache", default="data_cache/bench_batches")
    ap.add_argument("--deferred", action="store_true", help="TrainStep.step_deferred (no per-step reads)")
    ap.add_argument("--launch-sites", action="store_true", help="kernel launches per source line")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--compile-decoder", action="store_true", help="torch.compile the head's decoder")
    ap.add_argument("--live", action="store_true", help="real DataLoader + CUDAPrefetcher instead of cached batches")
    ap.add_argument("--gpm", default=None, metavar="FILE",
                    help="sample Hopper GPM counters (nvidia-smi dmon --gpm-metrics) during the timed steps "
                         "into FILE and print their means")
    ap.add_argument("--cpu-times", action="store_true",
                    help="CPU seconds per step without the profiler: whole process (all threads, incl. the "
                         "autograd backward thread) and the main thread alone")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.train.seed)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    device = resolve_device(cfg.train.device)
    if cfg.train.cuda_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(cfg.train.cuda_memory_fraction)
    live = None
    if args.live:
        # The trainer's real input path: streaming DataLoader workers + CUDAPrefetcher.
        from ..engine.step import CUDAPrefetcher
        live_prefetcher = CUDAPrefetcher(build_loaders(cfg)["train"], device)
        live = iter(live_prefetcher)
        batches = []
    else:
        batches = cached_batches(cfg, args.batches, Path(args.cache))
    model = UnifiedPerceptionModel(cfg).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    trunk_ahead = None
    if live is not None and getattr(cfg.train, "trunk_ahead", False):
        from ..engine.step import TrunkAhead
        trunk_ahead = TrunkAhead(model, device)
        live_prefetcher._on_next = trunk_ahead.launch
    if args.compile_decoder:
        model.head.decoder = torch.compile(model.head.decoder, dynamic=None)
    criterion = SetCriterion(cfg.model, cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
                                  fused=True)
    mean = torch.tensor(cfg.data.pixel_mean, device=device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(cfg.data.pixel_std, device=device).view(1, 1, 1, 3, 1, 1)
    ts = TrainStep(model, criterion, optimizer, mean, std, device, True)
    model.train()

    pinned = [{k: (v.pin_memory() if isinstance(v, torch.Tensor) else v) for k, v in b.items()} for b in batches]
    order = iter(range(10 ** 9))

    pending = []

    def step():
        batch = next(live) if live is not None else _h2d_async(pinned[next(order) % len(pinned)], device)
        ts.h2d(batch)
        ts.normalize()
        if trunk_ahead is not None:
            trunk_ahead.take(batch)
        ts.forward()
        if trunk_ahead is not None:
            trunk_ahead.finish()
        if args.deferred:
            pending.append(ts.step_deferred(cfg.train.grad_clip))
            if len(pending) > 1:
                pending.pop(0).result()       # the previous step's scalars, as the trainer reads them
            return
        ts.compute_loss()
        ts.free_inputs()
        ts.backward()
        ts.optimize(cfg.train.grad_clip)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    sampler = None
    if args.gpm:
        import subprocess
        sampler = subprocess.Popen(["nvidia-smi", "dmon", "--gpm-metrics", "1,2,3,5,7,10,12,13", "-d", "1",
                                    "-f", args.gpm], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    start, cpu0, main0 = time.perf_counter(), time.process_time(), time.thread_time()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    if sampler is not None:
        sampler.terminate()
        sampler.wait()
        print("[bench] GPM means over the timed steps: " + _gpm_means(args.gpm))
    per_step = (time.perf_counter() - start) / args.steps
    cpu_step = (time.process_time() - cpu0) / args.steps
    main_step = (time.thread_time() - main0) / args.steps
    print(f"[bench] {per_step:.3f} s/step over {args.steps} steps "
          f"(peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)")
    if args.cpu_times:
        print(f"[bench] CPU per step: process {cpu_step:.3f} s, main thread {main_step:.3f} s, "
              f"other threads {cpu_step - main_step:.3f} s (batch {cfg.data.batch_size})")

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        n = min(args.steps, 5)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        # Union of kernel intervals (kernels on different streams may overlap).
        spans = sorted((e.time_range.start, e.time_range.end) for e in events)
        busy, cur_s, cur_e = 0.0, None, None
        for s, e in spans:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e - cur_s
        print(f"[bench] GPU busy {busy / 1e6 / wall:.0%} of {wall / n:.3f} s/step under the profiler "
              f"({len(events) / n:.0f} kernels/step)")
        table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=25, max_name_column_width=70)
        print("[bench] top device kernels (all profiled steps):")
        print(table)

    if args.launch_sites:
        # Top-level ATen ops (each is >= 1 kernel launch plus dispatcher/Python overhead) and their
        # CPU time per source line; ops without a Python stack run in the autograd (backward) thread.
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            step()
            torch.cuda.synchronize()
        ops: collections.Counter = collections.Counter()
        kernels: collections.Counter = collections.Counter()
        cpu_us: collections.Counter = collections.Counter()
        for e in prof.events():
            if e.device_type != torch.autograd.DeviceType.CPU or not e.name.startswith("aten::"):
                continue
            parent = e.cpu_parent
            if parent is not None and parent.name.startswith("aten::"):
                continue                      # nested inside another op: counted with it
            site = None
            while parent is not None:
                if "unified_perception" in parent.name and "bench_step" not in parent.name:
                    site = parent.name.replace("\\", "/").split("unified_perception/", 1)[-1]
                    break
                parent = parent.cpu_parent
            site = site or "<backward / no python stack>"
            ops[site] += 1
            kernels[site] += len(e.kernels) + sum(len(c.kernels) for c in e.cpu_children)
            cpu_us[site] += e.cpu_time_total
        print(f"[bench] {sum(ops.values())} top-level ops, {sum(kernels.values())} kernels, "
              f"{sum(cpu_us.values()) / 1e3:.0f} ms CPU in one step; top sites by CPU time:")
        for site, us in cpu_us.most_common(40):
            print(f"  {ops[site]:6d} ops {kernels[site]:6d} k {us / 1e3:8.1f} ms  {site}")

    if args.sync_debug:
        sites: collections.Counter = collections.Counter()

        def hook(message, category, filename, lineno, file=None, line=None):
            if "synchroniz" in str(message):
                sites[_site(traceback.extract_stack()[:-1])] += 1

        old = warnings.showwarning
        warnings.showwarning = hook
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            step()
        finally:
            torch.cuda.set_sync_debug_mode(0)
            warnings.showwarning = old
        print(f"[bench] {sum(sites.values())} synchronizations in one step, by site:")
        for site, n in sites.most_common():
            print(f"  {n:5d}  {site}")


if __name__ == "__main__":
    sys.exit(main())
