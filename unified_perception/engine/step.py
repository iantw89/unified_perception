"""Single training-step decomposed into named phases.

Both the real training loop (engine/train.py) and the profiler
(tools/profile_train.py) call these same phase methods, so the profiler
instruments the exact same code paths with zero divergence.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..constants import ROOT_JOINTS, ROOT_SLICE


def _head(model: nn.Module):
    return getattr(model, "_orig_mod", model).head


def track_identity(batch: dict, gt_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-person identities and existence for track supervision.

    ``instance_ids`` (B,T,P) are stable within a clip (-1 = unknown identity, never
    a track).  Without them, slot order is assumed stable (true for all current
    providers).  ``instance_exists`` marks people present but unobserved, whose
    tracks are kept alive without supervision; it defaults to ``gt_valid``.
    """
    valid = gt_valid.bool()
    ids = batch.get("instance_ids")
    if ids is None:
        slots = torch.arange(valid.shape[-1], device=valid.device).expand_as(valid)
        ids = torch.where(valid, slots, torch.full_like(slots, -1))
    exists = batch.get("instance_exists")
    exists = valid if exists is None else exists.to(device=valid.device, dtype=torch.bool) | valid
    return ids.long(), exists & (ids.to(valid.device) >= 0)


def _grad_norm(model: nn.Module) -> torch.Tensor:
    """L2 norm across all parameter gradients (one fused multi-tensor reduction)."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.nn.utils.get_total_norm(grads, 2.0)


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
        self.camera_valid: torch.Tensor | None = None
        self.gt_mm: torch.Tensor | None = None
        self.gt_valid: torch.Tensor | None = None
        self.gt_frame_valid: torch.Tensor | None = None
        self.gt_keypoint_valid: torch.Tensor | None = None
        self.gt_root: torch.Tensor | None = None
        self.gt_root_valid: torch.Tensor | None = None
        self.gt_root_norm: torch.Tensor | None = None
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
        # uint8 stays uint8: the model normalizes one frame at a time (~3 GB less per window).
        imgs = batch["imgs"]
        self.imgs = imgs.to(d) if imgs.dtype == torch.uint8 else imgs.to(device=d, dtype=torch.float32)
        self.world2img  = batch["world2img"].to(d)
        self.intrinsics = batch["intrinsics"].to(d)
        self.camera_valid = batch.get("camera_valid")
        # Host copy of the view mask: the backbone selects active views by an index built on the
        # host instead of boolean-mask indexing on the device (a synchronization per step).
        self.camera_valid_host = batch.get("camera_valid_host")
        if self.camera_valid is not None:
            if self.camera_valid_host is None and not self.camera_valid.is_cuda:
                self.camera_valid_host = self.camera_valid.bool()
            self.camera_valid = self.camera_valid.to(device=d, dtype=torch.bool)
        self.gt_mm      = batch["gt_keypoints"].to(d)
        self.gt_valid   = batch["gt_valid"].to(d)
        self.gt_frame_valid = batch.get(
            "gt_keypoint_frame_valid",
            torch.ones(self.gt_valid.shape[:2], dtype=torch.bool, device=d),
        ).to(device=d, dtype=torch.bool)
        self.gt_keypoint_valid = batch.get(
            "gt_keypoint_valid",
            self.gt_valid.unsqueeze(-1).expand(*self.gt_valid.shape, self.gt_mm.shape[-2]),
        ).to(device=d, dtype=torch.bool)
        self.gt_keypoint_valid = self.gt_keypoint_valid & self.gt_valid.unsqueeze(-1) & self.gt_frame_valid[:, :, None, None]
        self.gt_root = batch.get("gt_root")
        if self.gt_root is not None:
            self.gt_root = self.gt_root.to(d)
        self.gt_root_valid = batch.get("gt_root_valid", self.gt_valid).to(device=d, dtype=torch.bool)
        self.instance_ids, self.instance_exists = track_identity(batch, self.gt_valid)
        self.instance_ids = self.instance_ids.to(d)
        self.instance_exists = self.instance_exists.to(d)
        # Approximate-GT extras (data/golden.py); absent for exact-GT datasets.
        self.gt_sigma = batch.get("gt_keypoint_sigma")
        self.gt_root_sigma = batch.get("gt_root_sigma")
        self.gt_kp2d = batch.get("gt_kp2d")
        self.gt_kp2d_valid = batch.get("gt_kp2d_valid")
        self.gt_match_root = batch.get("gt_match_root")
        self.gt_match_root_valid = batch.get("gt_match_root_valid")
        for name in ("gt_sigma", "gt_root_sigma", "gt_kp2d", "gt_kp2d_valid", "gt_match_root",
                     "gt_match_root_valid"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(d))
        reset = batch.get("stream_reset")
        self.stream_reset = None if reset is None else reset.to(device=d, dtype=torch.bool)


    def normalize(self) -> None:
        """Normalize images in-place; project keypoints to model space."""
        if self.imgs.dtype != torch.uint8:
            self.imgs.sub_(self._norm_mean).div_(self._norm_std)
        self.gt_norm = self.model.normalize_keypoints(self.gt_mm)
        if self.gt_root is not None:
            self.gt_root_norm = self.model.normalize_keypoints(self.gt_root)
        else:
            self.gt_root_norm = self.gt_norm[..., ROOT_SLICE, :].mean(dim=-2)
        if self.gt_match_root is not None:
            self.gt_match_root = self.model.normalize_keypoints(self.gt_match_root)

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
                # DN anchors use the mean of both hips. Skip unlabeled frames
                # and people with either hip unannotated; their main-query root
                # loss remains available from independent root targets.
                self.gt_valid
                & self.gt_frame_valid[:, :, None]
                & self.gt_keypoint_valid[..., ROOT_SLICE].all(dim=-1),
                return_2d=return_2d,
                gt_keypoint_valid=self.gt_keypoint_valid,
                camera_valid=self.camera_valid,
                track_targets={
                    "gt": self.gt_norm, "valid": self.gt_valid,
                    "keypoint_valid": self.gt_keypoint_valid, "frame_valid": self.gt_frame_valid,
                    "root": self.gt_root_norm, "root_valid": self.gt_root_valid,
                    "instance_ids": self.instance_ids, "exists": self.instance_exists,
                    **self._extras(),
                },
                stream_reset=self.stream_reset,
                camera_valid_host=self.camera_valid_host,
            )

    def _extras(self) -> dict:
        extras = {"sigma_mm": self.gt_sigma, "root_sigma_mm": self.gt_root_sigma}
        if self.gt_kp2d is not None:
            extras.update(kp2d=self.gt_kp2d, kp2d_valid=self.gt_kp2d_valid, world2img=self.world2img)
        if self.gt_match_root is not None:
            extras.update(match_root=self.gt_match_root, match_root_valid=self.gt_match_root_valid.bool())
        return {k: v for k, v in extras.items() if v is not None}

    def compute_loss(self) -> float:
        """Compute criterion loss; return scalar value (may be non-finite)."""
        self.loss, self.metrics = self.criterion(
            self.cls_scores.float(),
            self.kpts_norm.float(),
            self.gt_norm,
            self.gt_valid,
            self.dn_meta,
            gt_keypoint_frame_valid=self.gt_frame_valid,
            gt_root=self.gt_root_norm,
            gt_root_valid=self.gt_root_valid,
            gt_keypoint_valid=self.gt_keypoint_valid,
            track_output=_head(self.model).track_output,
            gt_keypoint_sigma=self.gt_sigma,
            gt_root_sigma=self.gt_root_sigma,
            gt_kp2d=self.gt_kp2d,
            gt_kp2d_valid=self.gt_kp2d_valid,
            world2img=self.world2img,
            position_range=getattr(self.model, "_orig_mod", self.model).position_range,
            gt_match_root=self.gt_match_root,
            gt_match_root_valid=self.gt_match_root_valid,
        )
        self.loss_val = self.loss.item()
        return self.loss_val

    # ── Deferred step (no host synchronization) ─────────────────────────────

    def deferred_supported(self) -> bool:
        """Deferred steps need a fused optimizer on CUDA (skip on device via ``found_inf``)."""
        return (self._device.type == "cuda"
                and all(g.get("fused", False) for g in self.optimizer.param_groups))

    def step_deferred(self, grad_clip: float) -> "PendingStep":
        """Loss, backward, clip and optimizer update without waiting for the device.

        A non-finite loss or gradient norm sets the fused optimizer's ``found_inf`` flag, so the
        update is skipped on the device (parameters, moments and step counts untouched) instead
        of by a host-side check.  Loss, gradient norm, the skip flag and every tensor metric are
        copied to pinned host memory behind an event; read them with :meth:`PendingStep.result`
        after the next step is queued, when the copy has long completed.
        """
        self.loss, self.metrics = self._criterion_outputs()
        self.free_inputs()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()   # allocator bookkeeping, no device sync
        self.loss.backward()
        norm = _grad_norm(self.model)
        found_inf = (~torch.isfinite(self.loss.detach()) | ~torch.isfinite(norm)).float()
        torch.nn.utils.clip_grads_with_norm_(
            [p for p in self.model.parameters() if p.grad is not None], grad_clip, norm)
        self.optimizer.found_inf = found_inf
        try:
            self.optimizer.step()
        finally:
            del self.optimizer.found_inf
        values = {"loss": self.loss.detach(), "grad_norm": norm, "skipped": found_inf}
        host = {}
        for key, value in self.metrics.items():
            if isinstance(value, torch.Tensor):
                values.setdefault(key, value)
            else:
                host[key] = value
        return PendingStep(values, host)

    def _criterion_outputs(self):
        return self.criterion(
            self.cls_scores.float(),
            self.kpts_norm.float(),
            self.gt_norm,
            self.gt_valid,
            self.dn_meta,
            gt_keypoint_frame_valid=self.gt_frame_valid,
            gt_root=self.gt_root_norm,
            gt_root_valid=self.gt_root_valid,
            gt_keypoint_valid=self.gt_keypoint_valid,
            track_output=_head(self.model).track_output,
            gt_keypoint_sigma=self.gt_sigma,
            gt_root_sigma=self.gt_root_sigma,
            gt_kp2d=self.gt_kp2d,
            gt_kp2d_valid=self.gt_kp2d_valid,
            world2img=self.world2img,
            position_range=getattr(self.model, "_orig_mod", self.model).position_range,
            gt_match_root=self.gt_match_root,
            gt_match_root_valid=self.gt_match_root_valid,
        )

    def free_inputs(self) -> None:
        """Delete input tensors before backward to free VRAM early."""
        del self.imgs, self.world2img, self.intrinsics, self.gt_mm, self.gt_valid
        del self.gt_frame_valid, self.gt_keypoint_valid, self.gt_root, self.gt_root_valid
        del self.gt_root_norm, self.gt_norm
        self.instance_ids = self.instance_exists = self.stream_reset = None
        self.gt_sigma = self.gt_root_sigma = self.gt_kp2d = self.gt_kp2d_valid = None
        self.gt_match_root = self.gt_match_root_valid = None
        self.imgs       = self.world2img  = self.intrinsics = None
        self.camera_valid = self.camera_valid_host = None
        self.gt_mm      = self.gt_valid   = self.gt_norm    = None
        self.gt_frame_valid = self.gt_keypoint_valid = None
        self.gt_root = self.gt_root_valid = self.gt_root_norm = None

    def backward(self) -> float:
        """Zero grads, reset memory stats, run backward; return grad norm."""
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.loss.backward()
        self._grad_norm_t = _grad_norm(self.model)
        self.grad_norm = self._grad_norm_t.item()
        return self.grad_norm

    def optimize(self, grad_clip: float) -> None:
        """Clip gradients (reusing the norm from :meth:`backward`) and step the optimizer."""
        norm = getattr(self, "_grad_norm_t", None)
        if norm is None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        else:
            torch.nn.utils.clip_grads_with_norm_(
                [p for p in self.model.parameters() if p.grad is not None], grad_clip, norm)
        self._grad_norm_t = None
        self.optimizer.step()


class PendingStep:
    """Scalars of a queued step, copied device -> pinned host behind an event."""

    def __init__(self, values: dict, host: dict):
        self.names = list(values)
        stacked = torch.stack([v.detach().float().reshape(()) for v in values.values()])
        self._buf = torch.empty(stacked.shape, dtype=torch.float32, pin_memory=True)
        self._buf.copy_(stacked, non_blocking=True)
        self._event = torch.cuda.Event()
        self._event.record()
        self._host = host
        self._result: dict | None = None

    def result(self) -> dict:
        """Wait only for this step's copy (later queued work keeps running) and return floats."""
        if self._result is None:
            self._event.synchronize()
            self._result = {**self._host, **dict(zip(self.names, self._buf.tolist()))}
            for key in [k for k in self._result if k.startswith("num_")]:
                self._result[key] = int(self._result[key])
        return self._result


# ---------------------------------------------------------------------------
# CUDA prefetcher
# ---------------------------------------------------------------------------

def _h2d_async(batch: dict, device: torch.device) -> dict:
    """Move batch tensors to *device* non-blocking (images stay uint8).

    Called inside a secondary CUDA stream — callers must synchronise via a
    ``torch.cuda.Event`` before consuming the returned tensors.
    """
    out = {
        k: (v.to(
                device=device,
                dtype=None,  # images stay uint8; the model normalizes per frame
                non_blocking=True,
            )
            if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    if isinstance(batch.get("camera_valid"), torch.Tensor) and not batch["camera_valid"].is_cuda:
        out["camera_valid_host"] = batch["camera_valid"].bool()
    return out


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

    def _enqueue(self, raw, stream):
        with torch.cuda.stream(stream):
            batch = _h2d_async(raw, self._device)
        evt = torch.cuda.Event()
        stream.record_event(evt)
        return batch, evt

    @staticmethod
    def _hand_over(batch: dict, stream) -> None:
        """Tensors allocated on the side stream are used on ``stream``: keep the caching
        allocator from reusing their memory before that stream is done with them."""
        for value in batch.values():
            for t in (value if isinstance(value, list) else [value]):
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    t.record_stream(stream)

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

        batch, evt = self._enqueue(raw, h2d_stream)

        # ── main loop ───────────────────────────────────────────────────
        for raw in loader_it:
            # Kick off H2D for the NEXT batch while the caller will
            # be processing the current one.
            nxt, nxt_evt = self._enqueue(raw, h2d_stream)

            # Block the compute stream until the CURRENT batch's H2D is done.
            # (nxt's H2D may still be in flight — that's intentional.)
            compute_stream.wait_event(evt)
            self._hand_over(batch, compute_stream)
            yield batch

            batch, evt = nxt, nxt_evt

        # ── final batch ─────────────────────────────────────────────────
        compute_stream.wait_event(evt)
        self._hand_over(batch, compute_stream)
        yield batch
