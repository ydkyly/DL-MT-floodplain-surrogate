# -*- coding: utf-8 -*-
"""Numerical robustness helpers: check_nan_inf counts or repairs nonfinite values; enforce_nonnegative clips negatives; round_tensor/round_array control decimals; sanitize_pred_for_save combines repair, nonnegative clipping, and rounding."""
from __future__ import annotations
import numpy as np
import torch

def check_nan_inf(x, name: str = "", *, raise_on_nan: bool = False, fix: bool = True):
    """Count and optionally repair NaN/Inf in tensors or NumPy arrays, using the corresponding nan_to_num operation. Return the same data type, repaired when fix=True."""
    if isinstance(x, torch.Tensor):
        nan = torch.isnan(x).sum().item()
        inf = torch.isinf(x).sum().item()
        if nan or inf:
            msg = f"[NaN/Inf] {name}: nan={nan}, inf={inf}, shape={tuple(x.shape)}"
            if fix:
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                print(msg + "  -> fixed by torch.nan_to_num(0)")
            elif raise_on_nan:
                raise ValueError(msg)
            else:
                print(msg)
        return x
    else:
        arr = np.asarray(x)
        nan = np.isnan(arr).sum()
        inf = np.isinf(arr).sum()
        if nan or inf:
            msg = f"[NaN/Inf] {name}: nan={nan}, inf={inf}, shape={arr.shape}"
            if fix:
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                print(msg + "  -> fixed by np.nan_to_num(0)")
            elif raise_on_nan:
                raise ValueError(msg)
            else:
                print(msg)
        return arr

# sanity.py
def enforce_nonnegative(x, tol: float = 1e-6, *, return_count: bool = False):
    """Enforce nonnegative values and zero very small magnitudes below tol. Return (clamped_x, neg_count) if return_count=True; otherwise return clamped_x."""
    if isinstance(x, torch.Tensor):
        if return_count:
            neg_cnt = torch.count_nonzero(x < -tol).item()
            x = torch.clamp(x, min=0.0)
            return x, neg_cnt
        else:
            return torch.clamp(x, min=0.0)
    else:
        arr = np.asarray(x)
        if return_count:
            neg_cnt = int((arr < -tol).sum())
            arr = np.maximum(arr, 0.0)
            return arr, neg_cnt
        else:
            return np.maximum(arr, 0.0)


def round_tensor(x: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    if decimals <= 0: return torch.round(x)
    factor = 10.0 ** decimals
    return torch.round(x * factor) / factor

def round_array(x: np.ndarray, decimals: int = 3) -> np.ndarray:
    return np.round(np.asarray(x), decimals).astype(np.float32)

def sanitize_pred_for_save(pred, *, decimals: int = 3, tol: float = 1e-6):
    """Prepare predictions for saving or plotting: repair NaN/Inf, enforce nonnegative values, and round. Return a numpy.float32 array."""
    pred = check_nan_inf(pred, "pred", raise_on_nan=False, fix=True)
    pred = enforce_nonnegative(pred, tol=tol)
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = round_array(pred, decimals=decimals)
    return pred
