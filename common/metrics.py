from __future__ import annotations
import numpy as np

def safe_pearson(y_true, y_pred):
    a=np.asarray(y_true,dtype=float); b=np.asarray(y_pred,dtype=float)
    if len(a)<2 or np.std(a)<=1e-8 or np.std(b)<=1e-8: return None
    return float(np.corrcoef(a,b)[0,1])
