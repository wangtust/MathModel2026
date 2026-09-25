from __future__ import annotations
from .mult import ContestMULT
from .mrrf import MRRFNet
from .baselines import ConcatMLP, GRUFusion
from .tfr import TFRFeatureAdapter

def build_model(name: str, **kwargs):
    if name == 'mult': return ContestMULT(hidden_dim=kwargs.get('hidden_dim',24))
    if name == 'mrrf': return MRRFNet(hidden_dim=kwargs.get('hidden_dim',24), use_reconstruction=kwargs.get('use_reconstruction',True), use_reliability=kwargs.get('use_reliability',True), dual_reconstruction=kwargs.get('dual_reconstruction',False), mask_padding=kwargs.get('mask_padding',False), mask_unavailable=kwargs.get('mask_unavailable',False))
    if name == 'concat_mlp': return ConcatMLP()
    if name == 'gru': return GRUFusion()
    if name == 'tfr': return TFRFeatureAdapter()
    raise ValueError(f'Unknown model: {name}')
