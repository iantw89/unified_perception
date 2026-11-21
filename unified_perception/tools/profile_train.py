#!/usr/bin/env python
"""Training-step profiler for unified_perception.

Instruments the forward / backward / optimizer loop with torch.profiler and
writes the following outputs to --out-dir / <timestamp>/ :

  chrome_trace.json   Timeline + flame chart for CPU and GPU.
                      Open at  https://ui.perfetto.dev  (drag & drop).

  stacks_cpu.txt      CPU flame-graph stacks (folded format).
  stacks_cuda.txt     CUDA flame-graph stacks.
                      Render:  flamegraph.pl < stacks_cpu.txt > cpu.svg
                      Web:     drag and drop at https://www.speedscope.app

  tb/                 TensorBoard profiler trace.
                      View:    tensorboard --logdir <out-dir>
                      Then open the "PyTorch Profiler" tab.

  report.txt          Per-phase wall-clock timing, top-20 CUDA kernels,
                      top-20 CPU ops, GPU memory summary, recommendations.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, cast

import torch

from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
)

from unified_perception.config import load_config
from unified_perception.engine.step import CUDAPrefetcher, TrainStep
from unified_perception.engine.train import build_loaders, resolve_device
from unified_perception.models import SetCriterion, UnifiedPerceptionModel


# =========================================================================== #
# CLI                                                                          #
# =========================================================================== #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config",          help="YAML config path")
    p.add_argument("--out-dir",       default="runs/profile", help="Root output dir (timestamped sub-dir is created)")
    p.add_argument("--steps",   "-n", type=int, default=3,   help="Number of steps to actively profile")
    p.add_argument("--warmup",        type=int, default=3,    help="Profiler JIT-warmup steps (traced but discarded)")
    p.add_argument("--wait",          type=int, default=2,    help="Profiler wait steps (not traced)")
    p.add_argument("--prewarm",       type=int, default=5,    help="Loader pre-warm steps before profiler starts")
    p.add_argument("--no-stacks",     action="store_true",    help="Skip Python-stack capture (faster, no flame graphs)")
    p.add_argument("--no-memory",     action="store_true",    help="Skip memory profiling (faster)")
    p.add_argument("--compile",       default=True, action="store_true",    help="torch.compile the model before profiling (warm-up steps absorb compile time)")
    return p.parse_args()


# =========================================================================== #
# Report generation                                                            #
# =========================================================================== #

def _pct(part: float, total: float) -> str:
    return f"{100.0 * part / total:5.1f}%" if total > 0 else "  n/a"


def _cuda_us(e) -> float:
    """Total CUDA time (µs) for a profiler event.
    PyTorch ≥ 2.6 renamed ``cuda_time_total`` → ``device_time_total`` on FunctionEventAvg.
    """
    v = getattr(e, "cuda_time_total", None)
    if v is None:
        v = getattr(e, "device_time_total", 0)
    return float(v)


def _p(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, int(len(s) * pct / 100) - 1)
    return s[idx]


def generate_report(
    prof: profile,
    phase_wall: Dict[str, List[float]],
    out_dir: Path,
    device: torch.device,
    cfg,
    config_path: str,
    num_active: int,
) -> None:
    lines: List[str] = []
    W = 74

    def h1(title: str) -> None:
        lines.extend(["", "=" * W, f"  {title}", "=" * W])

    def h2(title: str) -> None:
        lines.extend(["", f"  {title}", "  " + "-" * (W - 2)])

    # ── Header ─────────────────────────────────────────────────────────────
    lines.append("=" * W)
    lines.append("  unified_perception  —  Training Step Profiler")
    lines.append("=" * W)
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Config    : {config_path}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    lines.append(f"  Device    : {device}  ({gpu_name})")
    lines.append(
        f"  Batch     : B={cfg.data.batch_size}  T={cfg.data.seq_len}"
        f"  N={cfg.data.num_cameras}  {cfg.data.image_height}×{cfg.data.image_width}"
    )
    lines.append(f"  Active    : {num_active} profiled steps")

    # ── Phase wall-clock timing table ──────────────────────────────────────
    h1("PHASE TIMING  (wall-clock, per step)")
    col = "{:>9}"
    hdr = f"  {'Phase':<22} {col.format('mean ms')} {col.format('p50 ms')} {col.format('p95 ms')} {col.format('max ms')}  {'%step':>6}"
    lines.append(hdr)
    lines.append("  " + "-" * (W - 2))

    # Order matches the numeric prefixes used in record_function labels.
    PHASE_ORDER = [
        ("1_data_load",       "data_load"),
        ("2_host_to_device",  "host_to_device"),
        ("3_normalize_fwd",   "normalize + fwd"),
        ("6_backward",        "loss + backward"),
        ("7_optimizer",       "optimizer"),
    ]

    total_mean = sum(
        statistics.mean(phase_wall[k]) for k, _ in PHASE_ORDER if phase_wall.get(k)
    )

    for key, label in PHASE_ORDER:
        ts = phase_wall.get(key, [])
        if not ts:
            continue
        m   = statistics.mean(ts)
        p50 = statistics.median(ts)
        p95 = _p(ts, 95)
        mx  = max(ts)
        lines.append(
            f"  {label:<22} {m:>9.2f} {p50:>9.2f} {p95:>9.2f} {mx:>9.2f}  {_pct(m, total_mean):>6}"
        )

    lines.append("  " + "-" * (W - 2))
    lines.append(f"  {'TOTAL STEP':<22} {total_mean:>9.2f}")

    # ── Per-phase CPU + CUDA time from profiler ─────────────────────────────
    h1("PHASE CPU / CUDA TIME  (profiler, ms per step, all active steps)")
    avgs     = prof.key_averages()
    by_key   = {e.key: e for e in avgs}

    PROF_LABELS = [
        "1_data_load",
        "2_host_to_device",
        "3_normalize",
        "4_forward",
        "5_loss",
        "6_backward",
        "7_optimizer",
    ]
    col2 = "{:>14}"
    lines.append(
        f"  {'Phase':<22} {col2.format('cpu ms/step')} {col2.format('cuda ms/step')}"
        f"  {'self_cuda %':>12}"
    )
    lines.append("  " + "-" * (W - 2))

    for lbl in PROF_LABELS:
        e = by_key.get(lbl)
        if e is None:
            continue
        cpu_ms  = e.cpu_time_total  / 1_000 / num_active
        cuda_ms = _cuda_us(e)        / 1_000 / num_active
        sc_pct  = _pct(e.self_cuda_time_total, _cuda_us(e)) if _cuda_us(e) > 0 else "   n/a"
        name    = lbl.lstrip("0123456789_")
        lines.append(
            f"  {name:<22} {cpu_ms:>14.3f} {cuda_ms:>14.3f}  {sc_pct:>12}"
        )

    # ── Top CUDA kernels ────────────────────────────────────────────────────
    h1("TOP 20 CUDA KERNELS  (self_cuda_time_total, all active steps)")
    cuda_sorted = sorted(
        [e for e in avgs if e.self_cuda_time_total > 0],
        key=lambda e: e.self_cuda_time_total,
        reverse=True,
    )
    total_cuda_us = sum(e.self_cuda_time_total for e in cuda_sorted)
    lines.append(
        f"  Total CUDA: {total_cuda_us/1e6:.3f} s total"
        f"  ({total_cuda_us/1e3/num_active:.2f} ms/step)"
    )
    lines.append(
        f"  {'Kernel / Op':<46} {'ms/step':>9} {'%cuda':>7}  {'calls/step':>10}"
    )
    lines.append("  " + "-" * (W - 2))
    for e in cuda_sorted[:20]:
        ms_per = e.self_cuda_time_total / 1_000 / num_active
        pct    = 100.0 * e.self_cuda_time_total / total_cuda_us if total_cuda_us > 0 else 0.0
        cps    = e.count / num_active
        lines.append(
            f"  {e.key[:46]:<46} {ms_per:>9.3f} {pct:>6.1f}%  {cps:>10.1f}"
        )

    # ── Top CPU ops ─────────────────────────────────────────────────────────
    h1("TOP 20 CPU OPS  (self_cpu_time_total, all active steps)")
    cpu_sorted = sorted(
        [e for e in avgs if e.self_cpu_time_total > 0],
        key=lambda e: e.self_cpu_time_total,
        reverse=True,
    )
    total_cpu_us = sum(e.self_cpu_time_total for e in cpu_sorted)
    lines.append(
        f"  Total CPU: {total_cpu_us/1e6:.3f} s total"
        f"  ({total_cpu_us/1e3/num_active:.2f} ms/step)"
    )
    lines.append(
        f"  {'Op':<46} {'ms/step':>9} {'%cpu':>7}  {'calls/step':>10}"
    )
    lines.append("  " + "-" * (W - 2))
    for e in cpu_sorted[:20]:
        ms_per = e.self_cpu_time_total / 1_000 / num_active
        pct    = 100.0 * e.self_cpu_time_total / total_cpu_us if total_cpu_us > 0 else 0.0
        cps    = e.count / num_active
        lines.append(
            f"  {e.key[:46]:<46} {ms_per:>9.3f} {pct:>6.1f}%  {cps:>10.1f}"
        )

    # ── GPU Memory ──────────────────────────────────────────────────────────
    h1("GPU MEMORY")
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_resv  = torch.cuda.max_memory_reserved()  / 1e9
        lines.append(f"  Peak allocated : {peak_alloc:.3f} GB")
        lines.append(f"  Peak reserved  : {peak_resv:.3f} GB")

    # ── Output-file guide ───────────────────────────────────────────────────
    h1("OUTPUT FILES  —  how to view each")

    chrome = out_dir / "chrome_trace.json"
    s_cpu  = out_dir / "stacks_cpu.txt"
    s_cuda = out_dir / "stacks_cuda.txt"
    tb_dir = out_dir / "tb"

    lines.extend([
        "",
        "  TIMELINE + FLAME CHART  (both CPU and GPU in one file)",
        f"    {chrome}",
        "    → Open  https://ui.perfetto.dev  and drag the file in.",
        "      Slices view  : one lane per thread/stream, full time axis.",
        "      Flame chart  : click the 'Flamegraph' button top-left.",
        "      Call stacks are shown when hovering any slice.",
        "",
        "  CPU FLAME GRAPH  (folded stacks, rendered with flamegraph.pl)",
        f"    {s_cpu}",
        "    flamegraph.pl --title 'CPU' < stacks_cpu.txt > cpu_flame.svg",
        "    Get flamegraph.pl:  https://github.com/brendangregg/FlameGraph",
        "    Or drag .txt file to  https://www.speedscope.app  (no install).",
        "",
        "  GPU FLAME GRAPH",
        f"    {s_cuda}",
        "    flamegraph.pl --title 'GPU' --countname us < stacks_cuda.txt > gpu_flame.svg",
        "    (same speedscope.app tip applies)",
        "",
        "  TENSORBOARD PROFILER  (GPU kernel timeline, memory view, operator view)",
        f"    tensorboard --logdir \"{out_dir}\"",
        "    Then open the 'PyTorch Profiler' tab in your browser.",
        "    Key views: Overview · Operator · Kernel · Memory · Trace",
    ])

    # ── Recommendations ─────────────────────────────────────────────────────
    h1("RECOMMENDATIONS")
    recs = _make_recommendations(phase_wall, total_mean, cuda_sorted, total_cuda_us)
    for r in recs:
        # Wrap long lines
        prefix = "  • "
        rest   = prefix + r
        while len(rest) > W:
            cut = rest.rfind(" ", 0, W)
            if cut < 4:
                cut = W
            lines.append(rest[:cut])
            rest = "    " + rest[cut + 1:]
        lines.append(rest)

    # ── Write & print ───────────────────────────────────────────────────────
    text = "\n".join(lines) + "\n"
    report_path = out_dir / "report.txt"
    report_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"[profiler] report saved → {report_path}")


def _make_recommendations(
    phase_wall: Dict[str, List[float]],
    total_mean: float,
    cuda_sorted: list,
    total_cuda_us: float,
) -> List[str]:
    recs: List[str] = []

    def mean_pct(key: str) -> float:
        ts = phase_wall.get(key, [])
        if not ts or total_mean <= 0:
            return 0.0
        return 100.0 * statistics.mean(ts) / total_mean

    dl_pct   = mean_pct("1_data_load")
    h2d_pct  = mean_pct("2_host_to_device")
    fwd_pct  = mean_pct("3_normalize_fwd")
    bwd_pct  = mean_pct("6_backward")
    opt_pct  = mean_pct("7_optimizer")

    if dl_pct > 40:
        recs.append(
            f"DATA LOAD is {dl_pct:.0f}% of step time — dominant bottleneck. "
            "Inspect whether the data is on local NVMe (not a network share). "
            "Try: more num_workers, larger buffer_frames / prefetch_queue_depth, "
            "or memory-map the dataset."
        )
    elif dl_pct > 20:
        recs.append(
            f"DATA LOAD is {dl_pct:.0f}% of step time. "
            "GPU is underutilised during this window — consider overlapping H2D "
            "with the next data prefetch using non_blocking=True on .to(device) calls."
        )

    if h2d_pct > 8:
        recs.append(
            f"HOST→DEVICE is {h2d_pct:.0f}% of step time. "
            "The batch is already uint8 (4× smaller than float32). "
            "Try non_blocking=True on all .to(device) calls and structure the loop "
            "so the H2D transfer for step N+1 overlaps with the compute of step N."
        )

    if bwd_pct > fwd_pct * 2.5 and bwd_pct > 10:
        recs.append(
            f"BACKWARD ({bwd_pct:.0f}%) is >2.5× FORWARD ({fwd_pct:.0f}%). "
            "This is normal but consider: (a) gradient checkpointing to trade "
            "recompute for memory if the model is memory-bound, "
            "(b) fused optimizers (torch.optim.AdamW with foreach=True), "
            "(c) profiling the backward with torch.autograd.set_detect_anomaly."
        )

    if opt_pct > 10:
        recs.append(
            f"OPTIMIZER step is {opt_pct:.0f}% of step time — surprisingly high. "
            "Try AdamW(foreach=True) or torch.compile the optimizer step."
        )

    if cuda_sorted:
        top = cuda_sorted[0]
        top_pct = 100.0 * top.self_cuda_time_total / total_cuda_us if total_cuda_us > 0 else 0.0
        if top_pct > 20:
            recs.append(
                f"GPU kernel '{top.key}' dominates CUDA time ({top_pct:.0f}%). "
                "Check if this is amenable to kernel fusion, Flash Attention, "
                "or a more memory-efficient variant."
            )

    # Check for potential CPU/GPU overlap opportunity
    gpu_total_ms  = total_cuda_us / 1_000
    cpu_bound_ms  = (dl_pct + h2d_pct) / 100 * total_mean
    if gpu_total_ms > 0 and cpu_bound_ms / total_mean > 0.3:
        recs.append(
            "There appears to be significant CPU-bound time (data + H2D) alongside "
            "GPU compute. Inspect the Chrome trace in Perfetto for idle gaps on the "
            "CUDA stream — these are the primary overlap opportunities."
        )

    if not recs:
        recs.append(
            "Profile looks reasonably balanced. Examine the Chrome trace in Perfetto "
            "for micro-level opportunities such as small CUDA-stream idle gaps or "
            "unexpected host-side blocking calls."
        )

    return recs


# =========================================================================== #
# Main profiling loop                                                           #
# =========================================================================== #

def run_profile(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  unified_perception training profiler")
    print(f"  config  : {args.config}")
    print(f"  output  : {out_dir}")
    print(f"  steps   : wait={args.wait}  warmup={args.warmup}  active={args.steps}")
    print(f"  prewarm : {args.prewarm} batches (before profiler starts)")
    print(f"  stacks  : {'disabled (--no-stacks)' if args.no_stacks else 'enabled'}")
    print(f"{'='*60}\n")

    cfg    = load_config(args.config)
    device = resolve_device(cfg.train.device)

    # ── Build model, criterion, optimizer ──────────────────────────────────
    model     = UnifiedPerceptionModel(cfg).to(device).train()
    criterion = SetCriterion(cfg.model, cfg.loss)
    if args.compile:
        torch._dynamo.config.capture_scalar_outputs = True
        print("[profiler] torch.compile(model)  (warmup steps absorb first-run compile time)...")
        model = cast(torch.nn.Module, torch.compile(model))
        criterion = cast(torch.nn.Module, torch.compile(criterion))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    _norm_mean = torch.tensor(
        cfg.data.pixel_mean, dtype=torch.float32, device=device
    ).view(1, 1, 1, 3, 1, 1)
    _norm_std = torch.tensor(
        cfg.data.pixel_std, dtype=torch.float32, device=device
    ).view(1, 1, 1, 3, 1, 1)

    use_amp = device.type == "cuda"

    ts = TrainStep(model, criterion, optimizer, _norm_mean, _norm_std, device, use_amp)

    # ── Build data loader ──────────────────────────────────────────────────
    loaders     = build_loaders(cfg)
    loader      = loaders["train"]
    if device.type == "cuda":
        loader = CUDAPrefetcher(loader, device)
    loader_iter = iter(loader)

    # ── Pre-warm: fill shuffle buffer before profiling ─────────────────────
    if args.prewarm > 0:
        print(f"[profiler] pre-warming data loader ({args.prewarm} batches)...")
        t_pw = time.perf_counter()
        for _ in range(args.prewarm):
            _ = next(loader_iter)
        print(f"[profiler] pre-warm done in {time.perf_counter()-t_pw:.1f}s\n")

    # ── Profiler schedule ──────────────────────────────────────────────────
    WAIT   = args.wait
    WARMUP = args.warmup
    ACTIVE = args.steps

    prof_schedule = schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1)
    tb_dir        = out_dir / "tb"
    tb_dir.mkdir(exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Per-step wall-clock timers (with CUDA sync for GPU-bound phases).
    phase_wall: Dict[str, List[float]] = defaultdict(list)

    total_steps = WAIT + WARMUP + ACTIVE
    print(f"[profiler] starting {total_steps} profiler steps "
          f"({WAIT} wait + {WARMUP} warmup + {ACTIVE} active)\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=not args.no_memory,
        with_stack=not args.no_stacks,
    ) as prof:
        for step in range(total_steps):
            is_active = step >= WAIT + WARMUP
            tag = "[active]" if is_active else f"[{'warmup' if step >= WAIT else 'wait  '}]"

            # ── 1. Data load ────────────────────────────────────────────────
            t0 = time.perf_counter()
            with record_function("1_data_load"):
                batch = next(loader_iter)
            t1 = time.perf_counter()

            # ── 2. Host → device ────────────────────────────────────────────
            with record_function("2_host_to_device"):
                ts.h2d(batch)
            t2 = time.perf_counter()

            # ── 3. Normalize  4. Forward ────────────────────────────────────
            with record_function("3_normalize"):
                ts.normalize()

            with record_function("4_forward"):
                ts.forward(return_2d=False)

            # Sync after forward so we get accurate split between fwd and bwd.
            if is_active and torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            # ── 5. Loss ─────────────────────────────────────────────────────
            with record_function("5_loss"):
                loss_val = ts.compute_loss()

            if not math.isfinite(loss_val):
                print(f"  step {step:3d}/{total_steps-1} {tag}  loss={loss_val} — NaN/Inf, skipping")
                optimizer.zero_grad(set_to_none=True)
                prof.step()
                continue

            ts.free_inputs()

            # ── 6. Backward ─────────────────────────────────────────────────
            with record_function("6_backward"):
                ts.backward()

            if is_active and torch.cuda.is_available():
                torch.cuda.synchronize()
            t4 = time.perf_counter()

            # ── 7. Optimizer ─────────────────────────────────────────────────
            with record_function("7_optimizer"):
                ts.optimize(cfg.train.grad_clip)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t5 = time.perf_counter()

            # Record wall-clock splits for active steps.
            if is_active:
                phase_wall["1_data_load"].append((t1 - t0) * 1_000)
                phase_wall["2_host_to_device"].append((t2 - t1) * 1_000)
                phase_wall["3_normalize_fwd"].append((t3 - t2) * 1_000)  # normalize + forward
                phase_wall["6_backward"].append((t4 - t3) * 1_000)       # loss + backward
                phase_wall["7_optimizer"].append((t5 - t4) * 1_000)

            step_ms  = (t5 - t0) * 1_000
            dl_ms    = (t1 - t0) * 1_000
            fwd_ms   = (t3 - t2) * 1_000
            bwd_ms   = (t4 - t3) * 1_000
            print(
                f"  step {step:3d}/{total_steps-1} {tag}"
                f"  total={step_ms:7.1f}ms"
                f"  data={dl_ms:6.1f}ms"
                f"  fwd={fwd_ms:6.1f}ms"
                f"  bwd={bwd_ms:6.1f}ms"
                f"  loss={loss_val:.4f}"
            )

            prof.step()

    # ── Export traces ───────────────────────────────────────────────────────
    # chrome_trace.json is the primary export; it is also a valid TensorBoard
    # PyTorch-profiler trace (same JSON format), so we copy it into tb/ with
    # the *.pt.trace.json naming that TensorBoard expects.
    print("\n[profiler] exporting traces...")

    chrome_path = out_dir / "chrome_trace.json"
    prof.export_chrome_trace(str(chrome_path))
    print(f"  chrome_trace  → {chrome_path}")

    tb_trace = tb_dir / f"worker0.{timestamp}.pt.trace.json"
    shutil.copy2(chrome_path, tb_trace)
    print(f"  tb trace      → {tb_trace}")

    if not args.no_stacks:
        for metric, fname in [
            ("self_cpu_time_total",  "stacks_cpu.txt"),
            ("self_cuda_time_total", "stacks_cuda.txt"),
        ]:
            path = out_dir / fname
            try:
                prof.export_stacks(str(path), metric=metric)
                print(f"  {fname:<20} → {path}")
            except Exception as exc:
                print(f"  [warn] {fname} export failed: {exc}")

    # ── Generate text report ────────────────────────────────────────────────
    print("\n[profiler] generating report...")
    generate_report(
        prof       = prof,
        phase_wall = phase_wall,
        out_dir    = out_dir,
        device     = device,
        cfg        = cfg,
        config_path= args.config,
        num_active = ACTIVE,
    )

    print(f"\n{'='*60}")
    print(f"  All outputs in: {out_dir}")
    print(f"  TensorBoard   : tensorboard --logdir \"{out_dir}\"")
    print(f"  Perfetto      : https://ui.perfetto.dev  (load chrome_trace.json)")
    print(f"  Speedscope    : https://www.speedscope.app  (load stacks_*.txt)")
    print(f"{'='*60}\n")


# =========================================================================== #
# Entry point                                                                  #
# =========================================================================== #

if __name__ == "__main__":
    run_profile(_parse_args())
