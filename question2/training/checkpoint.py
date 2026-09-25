from __future__ import annotations
from pathlib import Path
import torch

def save_checkpoint(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)
def load_checkpoint(path, device='cpu'):
    return torch.load(Path(path),map_location=device,weights_only=True)
