"""Dual-head contest baseline using the forked MMSA MulT backbone."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from MMSA.models.singleTask.MULT import MULT


class ContestMULT(nn.Module):
    def __init__(self, hidden_dim: int = 24, dropout: float = 0.2,
                 mask_padding: bool = False, mask_unavailable: bool = False):
        super().__init__()
        args = SimpleNamespace(
            use_bert=False, train_mode="regression", feature_dims=[768, 74, 35],
            dst_feature_dim_nheads=[hidden_dim, 4], nlevels=1,
            attn_dropout=0.1, attn_dropout_a=0.1, attn_dropout_v=0.1,
            relu_dropout=0.1, embed_dropout=0.1, res_dropout=0.1,
            output_dropout=dropout, text_dropout=0.1, attn_mask=False,
            mask_padding=mask_padding,
            mask_unavailable=mask_unavailable,
            conv1d_kernel_size_l=1, conv1d_kernel_size_a=1,
            conv1d_kernel_size_v=1,
        )
        self.backbone = MULT(args)
        fused_dim = hidden_dim * 6
        self.classifier = nn.Sequential(nn.LayerNorm(fused_dim),
                                        nn.Linear(fused_dim, 64), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(64, 3))
        self.regressor = nn.Sequential(nn.LayerNorm(fused_dim),
                                       nn.Linear(fused_dim, 64), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        availability = ({name: batch[f"available_{name}"]
                         for name in ("text", "audio", "vision")}
                        if self.backbone.mask_padding and self.backbone.mask_unavailable and
                        all(f"available_{name}" in batch for name in ("text", "audio", "vision"))
                        else None)
        features = self.backbone(batch["text"], batch["audio"], batch["vision"],
                                 content_mask=batch["content"],
                                 availability_masks=availability)["Feature_fused"]
        logits = self.classifier(features)
        strength = 3.0 * torch.tanh(self.regressor(features).squeeze(-1))
        return logits, strength
