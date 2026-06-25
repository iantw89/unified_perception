"""YOLO-pose feature backbone.

Uses ultralytics pose model weights and taps a pre-head feature map from the
detection trunk.  The full pose head is also run so that its predictions are
available for auxiliary supervision or monitoring.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class YOLOPoseBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        feature_index: int = 16,
        trainable: bool = False,
    ):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLO pose backbone. "
                "Install with: pip install ultralytics"
            ) from exc

        self.yolo = YOLO(model_name).model
        self.feature_index = feature_index

        if not trainable:
            for p in self.yolo.parameters():
                p.requires_grad = False

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the full YOLO model.

        Returns
        -------
        feat : Tensor (B, C, H, W)
            Feature map tapped at ``self.feature_index``.
        pose_preds : dict
            Raw pose-head predictions with keys ``boxes``, ``scores``,
            ``feats``, ``kpts``.  The YOLO model is always called in training
            mode so the pose head returns the raw preds dict directly (no
            decoded-tensor wrapper).
        """
        y = []
        feat = None
        modules = self.yolo.model
        training_state = self.yolo.training
        self.yolo.train()
        try:
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
                if i == self.feature_index:
                    feat = x
                y.append(x)
        finally:
            self.yolo.train(training_state)

        return feat, x  # x is the preds dict from the Pose head
