"""Keypoint detection head (PETR-style), predicting per-person 3D keypoints.

Each object query represents one candidate person.  A query carries a 3D
reference point (the person's anchor, normalised into the capture volume) which
both seeds the query positional embedding and anchors the regressed keypoints.
For every decoder layer the head outputs:

    * a single-logit "is a person" classification score, and
    * J keypoints in normalised [0, 1] volume coordinates.

Predictions are produced for every decoder layer to enable auxiliary losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .misc import inverse_sigmoid, pos2posemb3d
from .transformer import TransformerDecoder


class KeypointPETRHead(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_joints: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_joints = num_joints
        self.num_layers = num_layers

        self.decoder = TransformerDecoder(
            dim=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Query positional embedding from a 3D reference point.
        self.query_embedding = nn.Sequential(
            nn.Linear(embed_dims * 3 // 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        cls_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
        )
        reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_joints * 3),
        )
        self.cls_branches = nn.ModuleList([cls_branch for _ in range(num_layers)])
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_layers)])

    def query_pos_from_reference(self, reference_points):
        """reference_points: (B, Q, 3) in [0,1] -> query_pos (B, Q, C)."""
        return self.query_embedding(pos2posemb3d(reference_points, self.embed_dims // 2))

    def forward(self, tgt, reference_points, memory, memory_pos, key_padding_mask=None):
        """Run the decoder and produce per-layer predictions.

        Args:
            tgt: (B, Q, C) query content embeddings.
            reference_points: (B, Q, 3) normalised anchors in [0, 1].
            memory: (B, M, C) image tokens.
            memory_pos: (B, M, C) image 3D position embeddings.
        Returns:
            cls_scores: (L, B, Q, 1)
            keypoints:  (L, B, Q, J, 3) normalised [0, 1]
            hidden:     (B, Q, C) last-layer query features
        """
        query_pos = self.query_pos_from_reference(reference_points)
        dec = self.decoder(tgt, query_pos, memory, memory_pos, key_padding_mask)
        # dec: (L, B, Q, C)
        L, B, Q, C = dec.shape
        J = self.num_joints

        ref_logit = inverse_sigmoid(reference_points).unsqueeze(2)  # (B, Q, 1, 3)
        cls_out, kpt_out = [], []
        for lvl in range(L):
            feat = dec[lvl]
            cls_out.append(self.cls_branches[lvl](feat))  # (B, Q, 1)
            reg = self.reg_branches[lvl](feat).view(B, Q, J, 3)
            kpts = torch.sigmoid(reg + ref_logit)  # (B, Q, J, 3) in [0,1]
            kpt_out.append(kpts)

        cls_scores = torch.stack(cls_out, dim=0)
        keypoints = torch.stack(kpt_out, dim=0)
        return cls_scores, keypoints, dec[-1]
