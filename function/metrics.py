# -*- coding: utf-8 -*-
import torch
from typing import Tuple

@torch.no_grad()
def f1_pod_far(pred: torch.Tensor, target: torch.Tensor, threshold: float=0.1) -> Tuple[float,float,float]:
    p = (pred >= threshold); t = (target >= threshold)
    TP = torch.logical_and(p, t).sum().item()
    FP = torch.logical_and(p, ~t).sum().item()
    FN = torch.logical_and(~p, t).sum().item()
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # POD
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    far = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    return f1, recall, far
