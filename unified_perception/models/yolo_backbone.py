"""YOLO-pose feature backbone.

Uses ultralytics pose model weights and taps a pre-head feature map from the
detection trunk.  The full pose head is also run so that its predictions are
available for auxiliary supervision or monitoring.
"""

from __future__ import annotations

import contextlib
import numpy as np
import torch
import torch.nn as nn
import torchvision


class YOLOPoseBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        feature_indices: list[int],
        trainable: bool = False,
    ):
        """Multi-scale YOLO feature extractor.

        Args:
            model_name:      Ultralytics model file / name.
            feature_indices: Ordered list of YOLO layer indices to tap.
                             Features are returned in the same order.
            trainable:       Allow backbone gradients.
        """
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLO pose backbone. "
                "Install with: pip install ultralytics"
            ) from exc

        self.yolo = YOLO(model_name).model
        self.feature_indices = sorted(set(feature_indices))
        self._feature_set   = set(self.feature_indices)
        self._max_index     = max(self.feature_indices)
        self._trainable     = trainable

        if not trainable:
            for p in self.yolo.parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        return_detections: bool = False,
        conf_thresh: float = 0.25,
    ):
        """Run YOLO modules, optionally also decoding 2-D pose detections.

        Parameters
        ----------
        x : (B, 3, H, W) float tensor, values in [0, 1].
        return_detections : if True, runs the full model (including the Pose
            head) and returns ``(features, detections)``; if False (default)
            stops after the last tapped feature layer for efficiency.
        conf_thresh : confidence threshold used when decoding detections.

        Returns
        -------
        features : list[Tensor (B, C_l, H_l, W_l)]
            Feature maps at ``self.feature_indices``.
        detections : list of B dicts — only present when ``return_detections``.
            Each dict has numpy arrays ``boxes`` (K,4) xyxy, ``kpts`` (K,J,2),
            ``scores`` (K,). Otherwise None.
        """
        y = []
        tapped: dict[int, torch.Tensor] = {}
        modules = self.yolo.model
        training_state = self.yolo.training
        if not self._trainable:
            self.yolo.eval()
        else:
            self.yolo.train()
        _no_grad_ctx = torch.no_grad() if not self._trainable else contextlib.nullcontext()
        try:
            with _no_grad_ctx:
                for i, m in enumerate(modules):
                    if m.f != -1:
                        x_in = (
                            [x if j == -1 else y[j] for j in m.f]
                            if isinstance(m.f, list)
                            else y[m.f]
                        )
                    else:
                        x_in = x
                    x = m(x_in)
                    if i in self._feature_set:
                        tapped[i] = x
                    # Early exit when detections are not needed (skip FPN + head).
                    if not return_detections and i == self._max_index and not self._trainable:
                        break
                    y.append(x)
        finally:
            self.yolo.train(training_state)

        features = [tapped[i] for i in self.feature_indices]
        if not return_detections:
            return features, None
        # x is now the Pose head's raw output; decode into per-image dicts.
        return features, self._decode_predictions(x, conf_thresh)

    def _decode_predictions(self, pred_raw, conf_thresh: float = 0.25) -> list:
        """Decode raw Pose-head output into per-image detection dicts.

        Parameters
        ----------
        pred_raw : raw output from the last YOLO module (Pose head), which in
            non-export eval mode is ``(pred_tensor, feats_list)`` or just a
            tensor in export mode.
        conf_thresh : minimum class score to retain a detection.

        Returns
        -------
        list of B dicts, each with:
            ``boxes``  (K, 4)    float32  xyxy pixel coords
            ``kpts``   (K, J, 2) float32  xy pixel coords (visibility dropped)
            ``scores`` (K,)      float32  objectness score
        """
        pred = pred_raw[0] if isinstance(pred_raw, (tuple, list)) else pred_raw

        head = self.yolo.model[-1]
        kpt_shape = getattr(head, "kpt_shape", (17, 3))
        nc       = int(getattr(head, "nc", 1))
        num_kpts = kpt_shape[0]
        kpt_dim  = kpt_shape[1]
        feat_dim = 4 + nc + num_kpts * kpt_dim

        pred = pred.cpu().float()
        if pred.ndim == 3 and pred.shape[1] == feat_dim:
            pred = pred.permute(0, 2, 1)   # (B, feat, na) -> (B, na, feat)

        results = []
        for b in range(pred.shape[0]):
            p      = pred[b].numpy()
            scores = p[:, 4:4 + nc].max(axis=1)
            keep   = scores > conf_thresh
            pk     = p[keep]

            if len(pk) == 0:
                results.append({
                    "boxes":  np.zeros((0, 4),           np.float32),
                    "kpts":   np.zeros((0, num_kpts, 2), np.float32),
                    "scores": np.zeros(0,                np.float32),
                })
                continue

            cx, cy, bw, bh = pk[:, 0], pk[:, 1], pk[:, 2], pk[:, 3]
            boxes   = np.stack(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
            ).astype(np.float32)
            kpts_xy = pk[:, 4 + nc:].reshape(-1, num_kpts, kpt_dim)[:, :, :2].astype(np.float32)

            keep_nms = torchvision.ops.nms(
                torch.from_numpy(boxes),
                torch.from_numpy(scores[keep].astype(np.float32)),
                iou_threshold=0.45,
            ).numpy()

            results.append({
                "boxes":  boxes[keep_nms],
                "kpts":   kpts_xy[keep_nms],
                "scores": scores[keep][keep_nms].astype(np.float32),
            })
        return results

    def predict_2d(self, imgs: torch.Tensor, conf_thresh: float = 0.25) -> list:
        """Convenience wrapper: run full YOLO model and return 2-D detections.

        Equivalent to ``forward(imgs, return_detections=True, conf_thresh)[1]``.
        """
        _, dets = self.forward(imgs, return_detections=True, conf_thresh=conf_thresh)
        return dets
