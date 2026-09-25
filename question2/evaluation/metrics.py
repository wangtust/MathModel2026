from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from common.metrics import safe_pearson

def metrics(labels,predicted,strength,estimated):
    labels=np.asarray(labels,dtype=int); predicted=np.asarray(predicted,dtype=int); strength=np.asarray(strength,dtype=float); estimated=np.asarray(estimated,dtype=float)
    return {'accuracy':float(accuracy_score(labels,predicted)), 'macro_f1':float(f1_score(labels,predicted,labels=[0,1,2],average='macro',zero_division=0)), 'mae':float(mean_absolute_error(strength,estimated)), 'pearson':safe_pearson(strength,estimated), 'samples':len(labels)}

def coherent_strength(predicted, raw_strength):
    classes=np.asarray(predicted,dtype=int); raw=np.asarray(raw_strength,dtype=float)
    return np.where(classes==1,0.0,np.where(classes==0,np.minimum(raw,-1e-4),np.maximum(raw,1e-4)))
