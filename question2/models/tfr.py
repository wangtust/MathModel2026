"""Feature-level adapter of MMSA TFR-Net alignment, generator and gated fusion.

The original TFR-Net runs BERT internally.  Here the contest's fixed 768-D
BERT features are the text input, keeping every baseline on the same features.
The upstream temporal-final branch is skipped because it permutes the batch
axis into the Transformer sequence axis; its default behavior is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from MMSA.models.missingTask.TFR_NET.alignment import Alignment
from MMSA.models.missingTask.TFR_NET.fusion import C_GATE
from MMSA.models.missingTask.TFR_NET.generator import Generator

from .baselines import DualHead


class TFRFeatureAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 24, dropout: float = .2):
        super().__init__()
        args = SimpleNamespace(
            feature_dims=[768, 74, 35], seq_lens=[50, 50, 50],
            dst_feature_dim_nheads=[hidden_dim, 4], nlevels=1,
            alignmentModule="crossmodal_attn", generatorModule="linear",
            skip_temporal_final=True, text_dropout=.1,
            conv1d_kernel_size_l=1, conv1d_kernel_size_a=1,
            conv1d_kernel_size_v=1, attn_dropout=.1,
            attn_dropout_a=.1, attn_dropout_v=.1,
            relu_dropout=.1, embed_dropout=.1, res_dropout=.1,
            attn_mask=False,
        )
        self.alignment = Alignment(args)
        self.generators = nn.ModuleDict({
            "text": Generator(args, modality="text"),
            "audio": Generator(args, modality="audio"),
            "vision": Generator(args, modality="vision"),
        })
        width = 3 * hidden_dim
        self.gates = nn.ModuleDict({name: C_GATE(width, hidden_dim, 1, dropout)
                                    for name in ("text", "audio", "vision")})
        self.head = DualHead(3 * hidden_dim, dropout)

    @staticmethod
    def _content_gate(gate: C_GATE, sequence: torch.Tensor,
                      content: torch.Tensor) -> torch.Tensor:
        """Use official C-GATE layers while excluding CLS/SEP/padding in pooling."""
        lengths = content.sum(1).clamp(min=1)
        sequence = sequence[:, 1:]
        transformed = gate.gru(sequence, lengths)
        importance = torch.sigmoid(gate.cnn(transformed.transpose(1, 2)).transpose(1, 2))
        gated = torch.tanh(transformed) * importance
        encoded = torch.tanh(gate.fc(torch.cat((sequence, gated), dim=-1)))
        positions = torch.arange(encoded.shape[1], device=encoded.device)[None, :]
        encoded = encoded.masked_fill((positions >= lengths[:, None]).unsqueeze(-1),
                                      torch.finfo(encoded.dtype).min)
        return gate.dropout_in(encoded.max(1).values)

    def forward(self, batch: dict, return_aux: bool = False):
        outputs = self.alignment(batch["text"], batch["audio"], batch["vision"])
        sequences = dict(zip(("text", "audio", "vision"), outputs[:3]))
        recovered = {name: self.generators[name](sequence)
                     for name, sequence in sequences.items()}
        pooled = [self._content_gate(self.gates[name], sequences[name], batch["content"])
                  for name in ("text", "audio", "vision")]
        logits, strength = self.head(torch.cat(pooled, dim=1))
        if return_aux:
            return logits, strength, {"recovered": recovered}
        return logits, strength
