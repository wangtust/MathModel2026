"""Mask-aware reconstruction and reliability fusion over the MMSA MulT encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .mult import ContestMULT


MODALITIES = ("text", "audio", "vision")
DIMS = (768, 74, 35)


class MRRFNet(nn.Module):
    def __init__(self, hidden_dim: int = 24, dropout: float = 0.2,
                 use_reconstruction: bool = True, use_reliability: bool = True,
                 dual_reconstruction: bool = False, mask_padding: bool = True,
                 mask_unavailable: bool = True):
        super().__init__()
        self.use_reconstruction = use_reconstruction
        self.use_reliability = use_reliability
        self.dual_reconstruction = dual_reconstruction
        if dual_reconstruction and not use_reconstruction:
            raise ValueError("Dual reconstruction requires reconstruction to be enabled")
        # Reuse the official MMSA MulT encoder. Its sequence outputs are aligned
        # with the 50 supplied positions; content/padding is handled below.
        self.encoder = ContestMULT(hidden_dim=hidden_dim, dropout=dropout,
                                   mask_padding=mask_padding,
                                   mask_unavailable=mask_unavailable).backbone
        width = 2 * hidden_dim
        if use_reconstruction:
            self.reconstruct = nn.ModuleDict({
                name: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.GELU(), nn.Linear(width, dim))
                for name, dim in zip(MODALITIES, DIMS)
            })
            self.repair = nn.ModuleDict({
                name: nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width))
                for name, dim in zip(MODALITIES, DIMS)
            })
            if dual_reconstruction:
                # Separate, mask-aware within-modality encoders provide context
                # around an injected gap. The MulT sequences above supply the
                # complementary cross-modal reconstruction branch.
                self.intra_input = nn.ModuleDict({
                    name: nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width))
                    for name, dim in zip(MODALITIES, DIMS)
                })
                self.intra_encoder = nn.ModuleDict({
                    name: nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=width, nhead=4, dim_feedforward=2 * width,
                            dropout=dropout, batch_first=True, norm_first=False),
                        num_layers=1)
                    for name in MODALITIES
                })
                self.intra_reconstruct = nn.ModuleDict({
                    name: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
                    for name, dim in zip(MODALITIES, DIMS)
                })
                self.reconstruction_mix = nn.ModuleDict({
                    name: nn.Sequential(nn.LayerNorm(2 * width + 1),
                                        nn.Linear(2 * width + 1, width), nn.GELU(),
                                        nn.Linear(width, 1))
                    for name in MODALITIES
                })
        if use_reliability:
            self.gate = nn.ModuleDict({
                name: nn.Sequential(nn.LayerNorm(width + 1), nn.Linear(width + 1, 32),
                                    nn.GELU(), nn.Linear(32, 1))
                for name in MODALITIES
            })
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 3))
        self.regressor = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def _dual_reconstruct(self, name: str, cross_sequence: torch.Tensor,
                          batch: dict, available: torch.Tensor,
                          fraction: torch.Tensor) -> torch.Tensor:
        projected = self.intra_input[name](batch[name])
        # An all-missing modality has no keys. Expose one zero placeholder to
        # the attention kernel, then discard its intra-modal result entirely.
        key_padding = ~available.clone()
        empty = ~available.any(dim=1)
        key_padding[empty, 0] = False
        intra_sequence = self.intra_encoder[name](
            projected, src_key_padding_mask=key_padding)
        intra_sequence = torch.where(
            empty[:, None, None], torch.zeros_like(intra_sequence), intra_sequence)
        cross_feature = self.reconstruct[name](cross_sequence)
        intra_feature = self.intra_reconstruct[name](intra_sequence)
        mix_input = torch.cat((cross_sequence, intra_sequence,
                               fraction[:, None, None].expand(-1, cross_sequence.shape[1], 1)),
                              dim=-1)
        learned_cross = torch.sigmoid(self.reconstruction_mix[name](mix_input))
        # When no source frames survive, only the cross-modal branch may speak.
        cross_weight = 1 - fraction[:, None, None] * (1 - learned_cross)
        return cross_weight * cross_feature + (1 - cross_weight) * intra_feature

    def forward(self, batch: dict, return_aux: bool = False):

        availability = {
            name: batch[f"available_{name}"]
            for name in MODALITIES
        }

        result = self.encoder(
            batch["text"],
            batch["audio"],
            batch["vision"],
            content_mask=batch["content"],
            availability_masks=availability
        )
        content = batch["content"]
        length = content.sum(1).clamp(min=1)
        pooled, fractions, recovered = [], [], {}
        for name in MODALITIES:
            seq = result[f"Sequence_{name[0] if name != 'vision' else 'v'}"]
            available = batch[f"available_{name}"] & content
            frac = available.sum(1).float() / length.float()
            fractions.append(frac)
            if self.use_reconstruction:
                recovered[name] = (self._dual_reconstruct(name, seq, batch, available, frac)
                                   if self.dual_reconstruction else self.reconstruct[name](seq))
                missing = content & ~available
                # The reconstructed feature participates in prediction, so its
                # auxiliary loss cannot be optimized in isolation.
                seq = seq + missing.unsqueeze(-1) * self.repair[name](recovered[name])
            pooled.append((seq * content.unsqueeze(-1)).sum(1) / length.unsqueeze(-1))
        fraction = torch.stack(fractions, dim=1)
        if self.use_reliability:
            scores = torch.cat([
                self.gate[name](torch.cat((pooled[i], fraction[:, i:i+1]), dim=1))
                for i, name in enumerate(MODALITIES)
            ], dim=1)
            # Availability is a prior, while the learned gate can still use an
            # entirely absent modality's cross-modal reconstruction.
            weights = torch.softmax(scores + torch.log(fraction + 0.05), dim=1)
        else:
            weights = torch.full_like(fraction, 1.0 / len(MODALITIES))
        fused = torch.stack(pooled, dim=1)
        fused = (fused * weights.unsqueeze(-1)).sum(1)
        logits = self.classifier(fused)
        strength = 3.0 * torch.tanh(self.regressor(fused).squeeze(-1))
        if return_aux:
            return logits, strength, {"recovered": recovered, "weights": weights,
                                      "availability": fraction}
        return logits, strength


def reconstruction_loss(recovered: dict, original: dict, masked: dict) -> torch.Tensor:
    """Average per modality over injected positions with an observed target."""
    if not recovered:
        return original["text"].new_zeros(())
    terms = []
    for name in MODALITIES:
        positions = masked[f"injected_{name}"]
        if not bool(positions.any()):
            continue
        prediction = recovered[name][positions]
        target = original[name][positions]
        terms.append(F.smooth_l1_loss(prediction, target))
    return torch.stack(terms).mean() if terms else original["text"].new_zeros(())
