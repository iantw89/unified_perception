"""PETR-style transformer decoder.

Object (person) queries attend to multi-view image tokens.  Positional
information is injected DETR-style: query positional embeddings are added to the
query/key of self-attention, and the 3D camera position embedding is added to
the image keys during cross-attention.  Returns the output of every decoder
layer so the head can apply auxiliary losses.

The self-attention additionally supports StreamPETR-style temporal modelling:
``temp_memory`` (propagated features from previous frames) is concatenated to
the self-attention keys/values, and an optional ``attn_mask`` implements the
query-denoising (DN) group masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _with_pos(x, pos):
    return x if pos is None else x + pos


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(ffn_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        query_pos,
        memory,
        memory_pos,
        key_padding_mask=None,
        temp_memory=None,
        temp_pos=None,
        attn_mask=None,
    ):
        # Temporal self-attention: queries attend to themselves and, when
        # streaming, to propagated memory queries appended as extra keys/values.
        q = _with_pos(query, query_pos)
        if temp_memory is not None:
            k = torch.cat([_with_pos(query, query_pos), _with_pos(temp_memory, temp_pos)], dim=1)
            v = torch.cat([query, temp_memory], dim=1)
        else:
            k = _with_pos(query, query_pos)
            v = query
        q2 = self.self_attn(q, k, value=v, attn_mask=attn_mask, need_weights=False)[0]
        query = self.norm1(query + self.dropout(q2))

        # Cross-attention to image tokens.
        q2 = self.cross_attn(
            _with_pos(query, query_pos),
            _with_pos(memory, memory_pos),
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        query = self.norm2(query + self.dropout(q2))

        # Feed-forward.
        query = self.norm3(query + self.dropout(self.ffn(query)))
        return query


class TransformerDecoder(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        query,
        query_pos,
        memory,
        memory_pos,
        key_padding_mask=None,
        temp_memory=None,
        temp_pos=None,
        attn_mask=None,
    ):
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                memory,
                memory_pos,
                key_padding_mask=key_padding_mask,
                temp_memory=temp_memory,
                temp_pos=temp_pos,
                attn_mask=attn_mask,
            )
            intermediate.append(query)
        return torch.stack(intermediate, dim=0)  # (num_layers, B, Q, C)