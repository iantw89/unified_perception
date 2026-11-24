"""Where a training step (and an evaluation batch) spends its time, per model stage, and how well the
GPU is used there.

    python -m unified_perception.tools.profile_stages CONFIG --checkpoint last.pt [--warmup 20]
        [--steps 6] [--eval-batches 6] [--out DIR]

Training steps use the real streaming loader (CUDAPrefetcher) and the deferred step, as the trainer
does.  Stages are marked with torch.profiler record_function by wrapping module / function entry
points at runtime (no model code changes):

  trunk (frozen ViT), backbone (trunk + SimpleFPN + regrouping), feat_proj, self_attn, cross_attn,
  msda (the deformable-attention kernel call inside cross_attn), ffn, identity, ext_reid,
  track_assign, loss, backward, optimizer; evaluation adds eval_forward and eval_metrics.

Reported per stage (inclusive of nested stages): wall share of GPU kernel time, kernel count, CPU
time, and achieved TFLOP/s for the ops PyTorch counts FLOPs for (matmul / conv).  Saturation, without
hardware counters: GPU idle fraction of the step (host-bound time), and the GPU time split by kernel
duration (kernels under ~20 us rarely fill 132 SMs).  The backward pass and the whole step are also
split by kernel class (GEMM, attention, deformable attention, elementwise/reduction, copy/index).
"""

from __future__ import annotations

import argparse
import collections
import functools
import re
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from ..config import load_config
from ..engine.step import CUDAPrefetcher, TrainStep
from ..engine.train import build_loaders, resolve_device, set_seed
from ..models import SetCriterion, UnifiedPerceptionModel

H100_BF16_DENSE_TFLOPS = 989.0


def _wrap(obj, name: str, label: str) -> None:
    fn = getattr(obj, name)

    @functools.wraps(fn)
    def wrapped(*a, **k):
        with record_function(label):
            return fn(*a, **k)
    setattr(obj, name, wrapped)


def instrument(model, criterion) -> None:
    from ..models import transformer
    bb = model.backbone
    _wrap(bb, "forward", "S:backbone")
    if hasattr(bb, "_vit_features"):
        _wrap(bb, "_vit_features", "S:trunk")
    for proj in model.feat_projs:
        _wrap(proj, "forward", "S:feat_proj")
    for layer in model.head.decoder.layers:
        _wrap(layer.self_attn, "forward", "S:self_attn")
        _wrap(layer.cross_attn, "forward", "S:cross_attn")
        _wrap(layer.ffn, "forward", "S:ffn")
    if transformer.MMCVMSDAfn is not None:
        apply = transformer.MMCVMSDAfn.apply

        def msda(*a, **k):
            with record_function("S:msda"):
                return apply(*a, **k)
        transformer.MMCVMSDAfn = type("MSDA", (), {"apply": staticmethod(msda)})
    head = model.head
    if getattr(head, "identity_head", None) is not None:
        _wrap(head.identity_head, "forward", "S:identity")
    ext = head.__dict__.get("external_reid")
    if ext is not None:
        _wrap(ext, "forward", "S:ext_reid")
    tm = getattr(head, "track_manager", None)
    if tm is not None:
        for name in ("train_step", "infer_step"):
            if hasattr(tm, name):
                _wrap(tm, name, "S:track_assign")


KERNEL_CLASSES = [
    ("deform_attn", re.compile(r"ms_deformable|deform", re.I)),
    ("attention", re.compile(r"flash|fmha|sdpa|attention", re.I)),
    ("gemm", re.compile(r"gemm|nvjet|cutlass|wgmma|triton_tem|_mm_|matmul|conv|sm90_xmma", re.I)),
    ("copy_index", re.compile(r"copy|index|gather|scatter|cat|memcpy|memset|fill", re.I)),
    ("elementwise_reduce", re.compile(r"elementwise|reduce|vectorized|triton_poi|triton_red|triton_per|"
                                      r"softmax|norm|unrolled", re.I)),
]


def kernel_class(name: str) -> str:
    for cls, pattern in KERNEL_CLASSES:
        if pattern.search(name):
            return cls
    return "other"


def analyse(prof, steps: int, wall: float, title: str) -> str:
    events = prof.events()
    # Device events are kernels / memcpy / memset plus the GPU-side spans of record_function labels
    # ("gpu_user_annotation"), which must not count as kernel time.
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA
               and not e.name.startswith("S:")]
    # Busy = union of kernel intervals.
    spans = sorted((k.time_range.start, k.time_range.end) for k in kernels)
    busy, s0, e0 = 0.0, None, None
    for s, e in spans:
        if e0 is None or s > e0:
            if e0 is not None:
                busy += e0 - s0
            s0, e0 = s, e
        else:
            e0 = max(e0, e)
    if e0 is not None:
        busy += e0 - s0
    gpu_total = sum(k.time_range.elapsed_us() for k in kernels)
    lines = [f"===== {title}: {wall / steps:.3f} s per step under the profiler ({steps} steps)",
             f"GPU busy (kernel union / wall): {busy / 1e6 / wall:.0%};  kernels/step {len(kernels) / steps:.0f};  "
             f"summed kernel time {gpu_total / 1e6 / steps:.3f} s/step"]
    # Kernel duration histogram.
    bins = [(0, 10), (10, 20), (20, 50), (50, 200), (200, 1000), (1000, 1e12)]
    hist = collections.Counter()
    count = collections.Counter()
    for k in kernels:
        d = k.time_range.elapsed_us()
        for lo, hi in bins:
            if lo <= d < hi:
                hist[(lo, hi)] += d
                count[(lo, hi)] += 1
                break
    lines.append("GPU time by kernel duration:  " + "  ".join(
        f"{lo:g}-{hi:g}us {hist[(lo, hi)] / max(gpu_total, 1):.0%} ({count[(lo, hi)] // steps}/step)"
        if hi < 1e12 else f">{lo:g}us {hist[(lo, hi)] / max(gpu_total, 1):.0%} ({count[(lo, hi)] // steps}/step)"
        for lo, hi in bins))
    by_class = collections.Counter()
    for k in kernels:
        by_class[kernel_class(k.name)] += k.time_range.elapsed_us()
    lines.append("GPU time by kernel class:  " + "  ".join(
        f"{c} {v / max(gpu_total, 1):.0%}" for c, v in by_class.most_common()))

    # Stages: record_function ranges; kernels attributed by launch correlation to the enclosing range.
    stage_events = [e for e in events if e.name.startswith("S:")]
    stages = collections.defaultdict(lambda: {"gpu": 0.0, "kernels": 0, "cpu": 0.0, "flops": 0.0,
                                              "flop_gpu": 0.0, "classes": collections.Counter()})

    for ev in stage_events:
        st = stages[ev.name]
        st["cpu"] += ev.cpu_time_total
        # every kernel launched inside this range (inclusive of nested stages)
        stack = [ev]
        while stack:
            cur = stack.pop()
            for kk in getattr(cur, "kernels", []):
                st["gpu"] += kk.duration
                st["kernels"] += 1
                st["classes"][kernel_class(kk.name)] += kk.duration
            if cur.flops:
                st["flops"] += cur.flops
                st["flop_gpu"] += sum(kk.duration for kk in getattr(cur, "kernels", []))
            stack.extend(cur.cpu_children)
    # The autograd engine launches backward kernels from its own thread, outside any forward range:
    # everything not claimed by a top-level range is reported as "backward (rest)".
    claimed = sum(stages[n]["gpu"] for n in ("S:forward", "S:loss", "S:optimizer", "S:eval_total")
                  if n in stages)
    claimed_ids = set()
    for ev in stage_events:
        if ev.name in ("S:forward", "S:loss", "S:optimizer", "S:eval_total"):
            stack = [ev]
            while stack:
                cur = stack.pop()
                claimed_ids.update(id(kk) for kk in getattr(cur, "kernels", []))
                stack.extend(cur.cpu_children)
    rest = stages["S:backward (rest)"]
    for ev in events:
        for kk in getattr(ev, "kernels", []):
            if id(kk) not in claimed_ids:
                rest["gpu"] += kk.duration
                rest["kernels"] += 1
                rest["classes"][kernel_class(kk.name)] += kk.duration
    lines.append(f"{'stage':14s} {'GPU s/step':>10s} {'%GPU':>5s} {'kern/step':>9s} {'CPU s/step':>10s} "
                 f"{'TFLOP/s(mm)':>11s} {'%peak':>6s}  top kernel classes")
    for name, st in sorted(stages.items(), key=lambda kv: -kv[1]["gpu"]):
        tf = st["flops"] / (st["flop_gpu"] * 1e-6) / 1e12 if st["flop_gpu"] else float("nan")
        top = ", ".join(f"{c} {v / max(st['gpu'], 1):.0%}" for c, v in st["classes"].most_common(3))
        lines.append(f"{name[2:]:14s} {st['gpu'] / 1e6 / steps:10.3f} {st['gpu'] / max(gpu_total, 1):5.0%} "
                     f"{st['kernels'] / steps:9.0f} {st['cpu'] / 1e6 / steps:10.3f} {tf:11.1f} "
                     f"{tf / H100_BF16_DENSE_TFLOPS:6.0%}  {top}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--checkpoint", default=None, help="trained weights (default: fresh initialization)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--eval-batches", type=int, default=6)
    ap.add_argument("--eval-split", default="val_gym")
    ap.add_argument("--out", default="runs/profile_stages")
    ap.add_argument("--compile-decoder", action="store_true", help="torch.compile each decoder layer")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    if cfg.train.cuda_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(cfg.train.cuda_memory_fraction)
    loaders = build_loaders(cfg)
    model = UnifiedPerceptionModel(cfg).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    criterion = SetCriterion(cfg.model, cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
                                  fused=True)
    mean = torch.tensor(cfg.data.pixel_mean, device=device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(cfg.data.pixel_std, device=device).view(1, 1, 1, 3, 1, 1)
    ts = TrainStep(model, criterion, optimizer, mean, std, device, True)
    instrument(model, criterion)
    if args.compile_decoder:
        for i, layer in enumerate(model.head.decoder.layers):
            model.head.decoder.layers[i] = torch.compile(layer, dynamic=True)
    # Loss / backward / optimizer markers around the TrainStep phases.
    _wrap(ts, "_criterion_outputs", "S:loss")
    orig_backward = torch.Tensor.backward

    def backward(self, *a, **k):
        with record_function("S:backward"):
            return orig_backward(self, *a, **k)
    torch.Tensor.backward = backward
    orig_step = optimizer.step

    def opt_step(*a, **k):
        with record_function("S:optimizer"):
            return orig_step(*a, **k)
    optimizer.step = opt_step

    model.train()
    live = iter(CUDAPrefetcher(loaders["train"], device))
    pending = []

    def step():
        ts.h2d(next(live))
        ts.normalize()
        with record_function("S:forward"):
            ts.forward()
        pending.append(ts.step_deferred(cfg.train.grad_clip))
        if len(pending) > 1:
            pending.pop(0).result()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    plain = (time.perf_counter() - t0) / args.steps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    report = [f"[train] {plain:.3f} s/step without the profiler", analyse(prof, args.steps, wall, "TRAIN")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out / "train_trace.json"))

    # Evaluation: the trainer's run_eval on a few batches of one split.
    if args.eval_batches:
        from ..engine import evaluate as E
        loader = loaders[args.eval_split]
        _wrap(model, "forward", "S:eval_forward")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as eprof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with record_function("S:eval_total"):
                E.run_eval(model, loader, cfg, device, max_batches=args.eval_batches)
            torch.cuda.synchronize()
            ewall = time.perf_counter() - t0
        report.append(analyse(eprof, args.eval_batches, ewall, f"EVAL ({args.eval_split})"))
        eprof.export_chrome_trace(str(out / "eval_trace.json"))
        top = eprof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15, max_name_column_width=60)
        report.append("Eval: top CPU ops\n" + top)
    text = "\n\n".join(report)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
