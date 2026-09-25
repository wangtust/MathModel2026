"""Independent early-fusion baselines for the same aligned contest features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


MODALITIES = ("text", "audio", "vision")
DIMS = (768, 74, 35)


def availability(batch: dict) -> torch.Tensor:
    length = batch["content"].sum(1).clamp(min=1).float()
    return torch.stack([batch[f"available_{name}"].sum(1).float() / length
                        for name in MODALITIES], dim=1)


class DualHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 3))
        self.regressor = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, features):
        return self.classifier(features), 3.0 * torch.tanh(self.regressor(features).squeeze(-1))


class ConcatMLP(nn.Module):
    """Mean-pool each modality over content and concatenate once per sample."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.fusion = nn.Sequential(nn.LayerNorm(sum(DIMS) + 3),
                                    nn.Linear(sum(DIMS) + 3, 128), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU())
        self.head = DualHead(64, dropout)

    def forward(self, batch: dict):
        mask = batch["content"].unsqueeze(-1)
        length = mask.sum(1).clamp(min=1)
        pooled = [(batch[name] * mask).sum(1) / length for name in MODALITIES]
        features = torch.cat((*pooled, availability(batch)), dim=1)
        return self.head(self.fusion(features))


class GRUFusion(nn.Module):
    """Early-fusion bidirectional GRU packed to the actual content length."""

    def __init__(self, hidden_dim: int = 48, dropout: float = 0.2):
        super().__init__()
        self.project = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, 32), nn.LayerNorm(32),
                                                         nn.GELU())
                                      for name, dim in zip(MODALITIES, DIMS)})
        self.gru = nn.GRU(96, hidden_dim, num_layers=1, bidirectional=True,
                          batch_first=True)
        self.head = DualHead(2 * hidden_dim + 3, dropout)

    def forward(self, batch: dict):
        # Content occupies positions 1..L; position 0 is CLS and L+1 is SEP.
        lengths = batch["content"].sum(1).clamp(min=1)
        sequence = torch.cat([self.project[name](batch[name]) for name in MODALITIES], dim=-1)
        packed = pack_padded_sequence(sequence[:, 1:], lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        fused = torch.cat((hidden[-2], hidden[-1], availability(batch)), dim=1)
        return self.head(fused)
