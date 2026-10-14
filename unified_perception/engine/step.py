"""Single training-step decomposed into named phases.

Both the real training loop (engine/train.py) and the profiler
(tools/profile_train.py) call these same phase methods, so the profiler
instruments the exact same code paths with zero divergence.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _grad_norm(model: nn.Module) -> torch.Tensor:
    """L2 norm across all parameter gradients."""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2) ** 2
    return total ** 0.5


class TrainStep:
    """One training step decomposed into named phases.

    The instance is created once before the training loop and reused across
    all steps and epochs.  Each phase stores its outputs as attributes that
    are consumed by subsequent phases.

    Phase call order per step::

        ts.h2d(batch)           # host → device
        ts.normalize()          # in-place img norm + keypoint projection
        ts.forward(return_2d)   # model forward (AMP)
        loss_val = ts.compute_loss()
        # caller checks math.isfinite(loss_val) and skips if needed
        ts.free_inputs()        # release VRAM before backward
        grad_norm = ts.backward()
        # caller checks math.isfinite(grad_norm) and skips if needed
        ts.optimize(grad_clip)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Any,
        optimizer: torch.optim.Optimizer,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
        use_amp: bool,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self._norm_mean = norm_mean
        self._norm_std = norm_std
        self._device = device
        self._use_amp = use_amp

        # Phase outputs — written by phases, read by later phases.
        self.imgs: torch.Tensor | None = None
        self.world2img: torch.Tensor | None = None
        self.intrinsics: torch.Tensor | None = None
        self.gt_mm: torch.Tensor | None = None
        self.gt_valid: torch.Tensor | None = None
        self.gt_norm: torch.Tensor | None = None
        self.cls_scores: torch.Tensor | None = None
        self.kpts_norm: torch.Tensor | None = None
        self.dn_meta: object = None
        self.dets_2d: list = []
        self.loss: torch.Tensor | None = None
        self.loss_val: float = float("nan")
        self.metrics: dict = {}
        self.grad_norm: float = float("nan")

    # ── Phases ──────────────────────────────────────────────────────────────

    def h2d(self, batch: dict) -> None:
        """Transfer batch tensors to device (float32 for images)."""
        d = self._device
        self.imgs       = batch["imgs"].to(device=d, dtype=torch.float32)
        self.world2img  = batch["world2img"].to(d)
        self.intrinsics = batch["intrinsics"].to(d)
        self.gt_mm      = batch["gt_keypoints"].to(d)
        self.gt_valid   = batch["gt_valid"].to(d)

    def normalize(self) -> None:
        """Normalize images in-place; project keypoints to model space."""
        self.imgs.sub_(self._norm_mean).div_(self._norm_std)
        self.gt_norm = self.model.normalize_keypoints(self.gt_mm)

    def forward(self, return_2d: bool = False) -> None:
        """Run the model forward pass inside AMP autocast."""
        with torch.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=self._use_amp,
        ):
            self.cls_scores, self.kpts_norm, self.dn_meta, self.dets_2d = self.model(
                self.imgs,
                self.world2img,
                self.intrinsics,
                self.gt_norm,
                self.gt_valid,
                return_2d=return_2d,
            )

    def compute_loss(self) -> float:
        """Compute criterion loss; return scalar value (may be non-finite)."""
        self.loss, self.metrics = self.criterion(
            self.cls_scores.float(),
            self.kpts_norm.float(),
            self.gt_norm,
            self.gt_valid,
            self.dn_meta,
        )
        self.loss_val = self.loss.item()
        return self.loss_val

    def free_inputs(self) -> None:
        """Delete input tensors before backward to free VRAM early."""
        del self.imgs, self.world2img, self.intrinsics, self.gt_mm, self.gt_valid, self.gt_norm
        self.imgs       = self.world2img  = self.intrinsics = None
        self.gt_mm      = self.gt_valid   = self.gt_norm    = None

    def backward(self) -> float:
        """Zero grads, reset memory stats, run backward; return grad norm."""
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.loss.backward()
        self.grad_norm = _grad_norm(self.model).item()
        return self.grad_norm

    def optimize(self, grad_clip: float) -> None:
        """Clip gradients and step the optimizer."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()


# ---------------------------------------------------------------------------
# CUDA prefetcher
# ---------------------------------------------------------------------------

def _h2d_async(batch: dict, device: torch.device) -> dict:
    """Move batch tensors to *device* non-blocking; imgs converted to float32.

    Called inside a secondary CUDA stream — callers must synchronise via a
    ``torch.cuda.Event`` before consuming the returned tensors.
    """
    return {
        k: (v.to(
                device=device,
                dtype=torch.float32 if k == "imgs" else None,
                non_blocking=True,
            )
            if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


class CUDAPrefetcher:
    """Wraps a DataLoader to overlap H2D transfers with GPU compute.

    On each iteration the next batch is copied to device on a dedicated CUDA
    stream while the caller processes the current batch (forward + backward +
    optimizer step).  A ``torch.cuda.Event`` gates the compute stream so it
    only reads each batch after its H2D transfer has completed.

    On CPU devices the prefetcher degrades transparently to a plain iterator.

    Usage::

        prefetcher = CUDAPrefetcher(dataloader, device)
        for batch in prefetcher:           # batch tensors already on device
            ts.h2d(batch)                  # near-zero cost (no-op .to())
            ts.normalize()
            ...
    """

    def __init__(self, loader, device: torch.device) -> None:
        self._loader = loader
        self._device = device

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        if self._device.type != "cuda":
            yield from self._loader
            return

        d = self._device
        h2d_stream = torch.cuda.Stream(device=d)
        compute_stream = torch.cuda.current_stream(d)
        loader_it = iter(self._loader)

        # ── prime the first batch ────────────────────────────────────────
        try:
            raw = next(loader_it)
        except StopIteration:
            return

        with torch.cuda.stream(h2d_stream):
            batch = _h2d_async(raw, d)
        evt = torch.cuda.Event()
        h2d_stream.record_event(evt)

        # ── main loop ───────────────────────────────────────────────────
        for raw in loader_it:
            # Kick off H2D for the NEXT batch while the caller will
            # be processing the current one.
            with torch.cuda.stream(h2d_stream):
                nxt = _h2d_async(raw, d)
            nxt_evt = torch.cuda.Event()
            h2d_stream.record_event(nxt_evt)

            # Block the compute stream until the CURRENT batch's H2D is done.
            # (nxt's H2D may still be in flight — that's intentional.)
            compute_stream.wait_event(evt)
            yield batch

            batch, evt = nxt, nxt_evt

        # ── final batch ─────────────────────────────────────────────────
        compute_stream.wait_event(evt)
        yield batch
