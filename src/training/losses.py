from __future__ import annotations
import torch
from torch.nn import functional as F
def masked_huber(pred:torch.Tensor,target:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    valid=mask & torch.isfinite(target)
    if not valid.any(): raise ValueError('batch has no valid targets')
    return F.huber_loss(pred[valid],target[valid],delta=1.0)
